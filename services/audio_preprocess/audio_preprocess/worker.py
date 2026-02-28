from __future__ import annotations

import asyncio
import logging
import tempfile
import os

from sap_common.config import settings
from sap_common.minio_client import download_bytes, upload_bytes
from sap_common.valkey_streams import StreamWorker

from audio_preprocess.pipeline import preprocess

logger = logging.getLogger(__name__)


class PreprocessWorker(StreamWorker):
    STREAM = "sap:preprocess:pending"
    GROUP = "preprocess-workers"

    async def process(self, message_id: str, data: dict) -> None:
        session_id = data["session_id"]
        minio_key = data["minio_key"]
        sr = settings.sample_rate

        logger.info("Preprocessing session %s from %s", session_id, minio_key)
        await self.update_session_status(session_id, "preprocessing")

        # Download raw audio from MinIO
        raw_data = await download_bytes(minio_key)

        # Write to temp file for ffmpeg
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
            tmp.write(raw_data)
            tmp_path = tmp.name

        try:
            # Run preprocessing (CPU-bound)
            loop = asyncio.get_event_loop()
            audio, duration = await loop.run_in_executor(
                None, preprocess, tmp_path, sr
            )
        finally:
            os.unlink(tmp_path)

        # Upload preprocessed PCM to MinIO
        pcm_key = f"pcm/{session_id}/audio.f32le"
        await upload_bytes(pcm_key, audio.tobytes(), content_type="application/octet-stream")

        logger.info(
            "Preprocessed session %s: %.1fs, %d samples",
            session_id, duration, len(audio),
        )

        # Fan out to ASR, music analysis, and Flamingo workers
        msg_data = {
            "session_id": session_id,
            "minio_key": pcm_key,
            "duration_sec": str(duration),
            "sample_rate": str(sr),
        }
        await self.publish("sap:asr:pending", msg_data)       # Voxstral primary (falls back to Whisper)
        await self.publish("sap:music:pending", msg_data)      # Essentia + librosa
        await self.publish("sap:flamingo:pending", msg_data)   # Music Flamingo deep analysis

        # Update session status
        await self.update_session_status(session_id, "analyzing")


async def main():
    logging.basicConfig(level=logging.INFO)
    worker = PreprocessWorker()
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
