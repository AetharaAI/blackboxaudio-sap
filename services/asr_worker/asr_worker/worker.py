from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid

import numpy as np
from sqlalchemy import delete

from sap_common.config import settings
from sap_common.db import async_session_factory
from sap_common.minio_client import download_bytes
from sap_common.models import TranscriptSegment
from sap_common.valkey_streams import StreamWorker

from asr_worker.transcriber import DualPassTranscriber, ModelRegistry

logger = logging.getLogger(__name__)


class ASRWorker(StreamWorker):
    STREAM = "sap:asr:whisper:pending"  # Whisper is now fallback; Voxstral is primary on sap:asr:pending
    GROUP = "asr-workers"

    def __init__(self):
        super().__init__()
        device = os.environ.get("DEVICE", "cuda")
        compute_type = os.environ.get("COMPUTE_TYPE", "float16")
        self.registry = ModelRegistry(device=device, compute_type=compute_type)
        # Pre-load models at startup
        self.transcriber = DualPassTranscriber(
            self.registry.get_streaming_model(),
            self.registry.get_final_model(),
        )

    async def process(self, message_id: str, data: dict) -> None:
        session_id = data["session_id"]
        minio_key = data["minio_key"]
        sr = int(data.get("sample_rate", settings.sample_rate))

        logger.info("ASR starting for session %s", session_id)

        # Download PCM from MinIO
        pcm_bytes = await download_bytes(minio_key)
        audio = np.frombuffer(pcm_bytes, dtype=np.float32)

        loop = asyncio.get_event_loop()
        sid = uuid.UUID(session_id)

        # Pass 1: Streaming — emit partials
        streaming_segments = await loop.run_in_executor(
            None, lambda: list(self.transcriber.streaming_pass(audio, sr))
        )

        # Store streaming partials
        async with async_session_factory() as db:
            for seg in streaming_segments:
                db.add(TranscriptSegment(
                    session_id=sid,
                    t_start=seg["t_start"],
                    t_end=seg["t_end"],
                    text=seg["text"],
                    confidence=seg["confidence"],
                    is_final=False,
                    word_timestamps=seg.get("words"),
                ))
            await db.commit()

        # Publish streaming partials to aligner
        for seg in streaming_segments:
            await self.publish("sap:align:pending", {
                "session_id": session_id,
                "source": "asr",
                "payload": json.dumps(seg),
            })

        logger.info(
            "ASR streaming pass complete for %s: %d segments",
            session_id, len(streaming_segments),
        )

        # Pass 2: Final — high-accuracy transcription
        final_segments = await loop.run_in_executor(
            None, lambda: self.transcriber.final_pass(audio, sr)
        )

        # Replace streaming partials with final segments
        async with async_session_factory() as db:
            await db.execute(
                delete(TranscriptSegment).where(
                    TranscriptSegment.session_id == sid,
                    TranscriptSegment.is_final == False,
                )
            )
            for seg in final_segments:
                db.add(TranscriptSegment(
                    session_id=sid,
                    t_start=seg["t_start"],
                    t_end=seg["t_end"],
                    text=seg["text"],
                    confidence=seg["confidence"],
                    is_final=True,
                    word_timestamps=seg.get("words"),
                ))
            await db.commit()

        # Publish final to aligner
        await self.publish("sap:align:pending", {
            "session_id": session_id,
            "source": "asr_final",
            "payload": json.dumps(final_segments),
        })

        logger.info(
            "ASR final pass complete for %s: %d segments",
            session_id, len(final_segments),
        )


async def main():
    logging.basicConfig(level=logging.INFO)
    worker = ASRWorker()
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
