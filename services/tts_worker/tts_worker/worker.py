from __future__ import annotations

import asyncio
import logging
import uuid

from sap_common.minio_client import upload_bytes
from sap_common.valkey_streams import StreamWorker

from tts_worker.synthesizer import ChatterboxClient

logger = logging.getLogger(__name__)


class TTSWorker(StreamWorker):
    """TTS worker that delegates synthesis to Chatterbox TTS Server.

    No local model, no GPU needed — this worker is a thin async client
    that consumes requests from Valkey, calls the Chatterbox API,
    stores results in MinIO, and publishes completion events.

    Serves dual purpose:
    1. SAP output layer — spoken summaries, accessibility, agent voice
    2. Docuseries production — narration generation from scripts
    """

    STREAM = "sap:tts:pending"
    GROUP = "tts-workers"

    def __init__(self):
        super().__init__()
        self.client = ChatterboxClient()

    async def process(self, message_id: str, data: dict) -> None:
        session_id = data.get("session_id", str(uuid.uuid4()))
        text = data.get("text", "")
        voice_mode = data.get("voice_mode", "predefined")
        predefined_voice_id = data.get("predefined_voice_id")
        reference_audio_filename = data.get("reference_audio_filename")
        output_format = data.get("output_format", "wav")
        speed_factor = data.get("speed_factor")
        temperature = data.get("temperature")
        exaggeration = data.get("exaggeration")
        seed = data.get("seed")

        if not text:
            logger.warning("Empty text in TTS request for session %s", session_id)
            return

        logger.info(
            "TTS synthesis: session=%s mode=%s format=%s len=%d",
            session_id, voice_mode, output_format, len(text),
        )

        # Parse optional floats/ints from string values (Valkey stores strings)
        speed_f = float(speed_factor) if speed_factor else None
        temp_f = float(temperature) if temperature else None
        exag_f = float(exaggeration) if exaggeration else None
        seed_i = int(seed) if seed else None

        try:
            audio_bytes = await self.client.synthesize(
                text=text,
                voice_mode=voice_mode,
                predefined_voice_id=predefined_voice_id,
                reference_audio_filename=reference_audio_filename,
                output_format=output_format,
                split_text=True,
                chunk_size=120,
                temperature=temp_f,
                exaggeration=exag_f,
                speed_factor=speed_f,
                seed=seed_i,
            )
        except Exception as e:
            logger.error("Chatterbox API error for session %s: %s", session_id, e)
            await self.update_session_status(session_id, "tts_failed", str(e))
            raise

        if not audio_bytes:
            logger.warning("Chatterbox returned empty audio for session %s", session_id)
            return

        # Upload to MinIO
        voice_label = predefined_voice_id or reference_audio_filename or "default"
        ext = output_format if output_format in ("wav", "mp3", "opus") else "wav"
        minio_key = f"tts/{session_id}/narration_{voice_label}.{ext}"
        content_type = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
        }.get(ext, "audio/wav")

        await upload_bytes(minio_key, audio_bytes, content_type=content_type)

        logger.info(
            "TTS complete: session=%s key=%s size=%d bytes",
            session_id, minio_key, len(audio_bytes),
        )

        # Publish completion
        await self.publish("sap:tts:complete", {
            "session_id": session_id,
            "minio_key": minio_key,
            "voice_mode": voice_mode,
            "voice_id": voice_label,
            "output_format": ext,
            "text_length": str(len(text)),
            "audio_size_bytes": str(len(audio_bytes)),
        })

        await self.update_session_status(session_id, "tts_complete")


async def main():
    logging.basicConfig(level=logging.INFO)
    worker = TTSWorker()
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
