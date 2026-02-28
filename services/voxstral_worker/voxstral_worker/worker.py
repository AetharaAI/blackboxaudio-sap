"""SAP Voxstral Worker — Primary ASR via Mistral Voxtral Mini 4B Realtime.

Consumes from sap:asr:pending, streams audio to vLLM's /v1/realtime WebSocket,
and stores transcript segments. Falls back to Whisper (sap:asr:whisper:pending)
if vLLM is unreachable.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import struct
import uuid

import numpy as np
from sqlalchemy import delete

from sap_common.config import settings
from sap_common.db import async_session_factory
from sap_common.minio_client import download_bytes
from sap_common.models import TranscriptSegment
from sap_common.valkey_streams import StreamWorker

logger = logging.getLogger(__name__)

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://voxstral-vllm:8000")
VLLM_WS_URL = VLLM_BASE_URL.replace("http://", "ws://").replace("https://", "wss://")
CHUNK_DURATION_MS = 480  # Match Voxstral recommended delay
FALLBACK_STREAM = "sap:asr:whisper:pending"


class VoxstralWorker(StreamWorker):
    STREAM = "sap:asr:pending"
    GROUP = "voxstral-workers"

    async def process(self, message_id: str, data: dict) -> None:
        session_id = data["session_id"]
        minio_key = data["minio_key"]
        sr = int(data.get("sample_rate", settings.sample_rate))
        duration_sec = float(data.get("duration_sec", 0))

        logger.info("Voxstral ASR starting for session %s", session_id)

        # Download PCM from MinIO
        pcm_bytes = await download_bytes(minio_key)
        audio = np.frombuffer(pcm_bytes, dtype=np.float32)

        try:
            segments = await self._transcribe_via_vllm(audio, sr, session_id)
        except Exception as e:
            logger.warning(
                "Voxstral vLLM unavailable for session %s (%s), falling back to Whisper",
                session_id, e,
            )
            # Re-publish to Whisper fallback stream
            await self.publish(FALLBACK_STREAM, data)
            return

        if not segments:
            logger.info("Voxstral produced no segments for session %s", session_id)

        # Store segments in DB
        sid = uuid.UUID(session_id)
        async with async_session_factory() as db:
            for seg in segments:
                db.add(TranscriptSegment(
                    session_id=sid,
                    t_start=seg["t_start"],
                    t_end=seg["t_end"],
                    text=seg["text"],
                    confidence=seg.get("confidence", 0.9),
                    is_final=True,
                    word_timestamps=seg.get("words"),
                    source="voxstral",
                ))
            await db.commit()

        # Publish to aligner
        await self.publish("sap:align:pending", {
            "session_id": session_id,
            "source": "asr_final",
            "payload": json.dumps(segments),
        })

        logger.info(
            "Voxstral ASR complete for %s: %d segments",
            session_id, len(segments),
        )

    async def _transcribe_via_vllm(
        self, audio: np.ndarray, sr: int, session_id: str,
    ) -> list[dict]:
        """Stream audio to vLLM Voxstral via the /v1/realtime WebSocket API.

        Returns a list of transcript segment dicts with t_start, t_end, text.
        """
        import aiohttp

        # Convert float32 PCM to 16-bit PCM (16kHz) for Voxstral
        # Voxstral expects 16kHz audio
        target_sr = 16000
        if sr != target_sr:
            # Simple resampling via linear interpolation
            ratio = target_sr / sr
            n_samples = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, n_samples)
            audio_16k = np.interp(indices, np.arange(len(audio)), audio)
        else:
            audio_16k = audio

        # Convert to 16-bit PCM bytes
        pcm_16bit = (audio_16k * 32767).astype(np.int16).tobytes()

        segments = []
        chunk_samples = int(target_sr * CHUNK_DURATION_MS / 1000)
        chunk_bytes = chunk_samples * 2  # 16-bit = 2 bytes per sample

        ws_url = f"{VLLM_WS_URL}/v1/realtime"

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                ws_url,
                timeout=aiohttp.ClientWSCloseTimeout(ws_close=10),
            ) as ws:
                # Send session config
                await ws.send_json({
                    "type": "session.update",
                    "session": {
                        "model": "mistralai/Voxtral-Mini-4B-Realtime-2602",
                        "input_audio_format": "pcm16",
                        "input_audio_transcription": {
                            "model": "mistralai/Voxtral-Mini-4B-Realtime-2602",
                        },
                        "turn_detection": None,
                        "temperature": 0.0,
                    },
                })

                # Stream audio in chunks
                offset = 0
                while offset < len(pcm_16bit):
                    chunk = pcm_16bit[offset:offset + chunk_bytes]
                    await ws.send_json({
                        "type": "input_audio_buffer.append",
                        "audio": _bytes_to_base64(chunk),
                    })
                    offset += chunk_bytes
                    # Small delay to simulate realtime
                    await asyncio.sleep(0.01)

                # Signal end of input
                await ws.send_json({
                    "type": "input_audio_buffer.commit",
                })

                # Collect responses
                full_text = ""
                current_start = 0.0
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        event = json.loads(msg.data)
                        event_type = event.get("type", "")

                        if event_type == "response.audio_transcript.delta":
                            delta = event.get("delta", "")
                            full_text += delta

                        elif event_type == "response.audio_transcript.done":
                            transcript = event.get("transcript", full_text)
                            if transcript.strip():
                                # Estimate timing based on audio length
                                duration = len(audio) / sr
                                segments.append({
                                    "t_start": 0.0,
                                    "t_end": duration,
                                    "text": transcript.strip(),
                                    "confidence": 0.9,
                                })
                            break

                        elif event_type == "response.done":
                            break

                        elif event_type == "error":
                            raise RuntimeError(
                                f"vLLM error: {event.get('error', {}).get('message', 'unknown')}"
                            )

                    elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                        break

        return segments


def _bytes_to_base64(data: bytes) -> str:
    import base64
    return base64.b64encode(data).decode("ascii")


async def main():
    logging.basicConfig(level=logging.INFO)
    worker = VoxstralWorker()
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
