from __future__ import annotations

import asyncio
import json
import logging
import uuid

import numpy as np
from sqlalchemy import delete

from sap_common.config import settings
from sap_common.db import async_session_factory
from sap_common.minio_client import download_bytes
from sap_common.models import MusicAnalysis, MusicChordSegment
from sap_common.valkey_streams import StreamWorker

from music_worker.tempo import analyze_tempo_and_beats
from music_worker.key_detection import analyze_key
from music_worker.chords import analyze_chords
from music_worker.features import compute_frame_features

logger = logging.getLogger(__name__)


class MusicAnalysisWorker(StreamWorker):
    STREAM = "sap:music:pending"
    GROUP = "music-workers"

    async def process(self, message_id: str, data: dict) -> None:
        session_id = data["session_id"]
        minio_key = data["minio_key"]
        sr = int(data.get("sample_rate", settings.sample_rate))

        logger.info("Music analysis starting for session %s", session_id)

        # Download PCM from MinIO
        pcm_bytes = await download_bytes(minio_key)
        audio = np.frombuffer(pcm_bytes, dtype=np.float32)

        # Run all analyses in parallel (CPU-bound, use executor)
        loop = asyncio.get_event_loop()
        tempo_future = loop.run_in_executor(None, analyze_tempo_and_beats, audio, sr)
        key_future = loop.run_in_executor(None, analyze_key, audio, sr)
        chord_future = loop.run_in_executor(None, analyze_chords, audio, sr)
        features_future = loop.run_in_executor(None, compute_frame_features, audio, sr)

        tempo_result, key_result, chord_segments, frame_features = await asyncio.gather(
            tempo_future, key_future, chord_future, features_future
        )

        logger.info(
            "Music analysis complete for %s: bpm=%.1f key=%s:%s chords=%d",
            session_id,
            tempo_result["tempo_bpm"],
            key_result["key_label"],
            key_result["key_scale"],
            len(chord_segments),
        )

        # Store results in Postgres
        async with async_session_factory() as db:
            sid = uuid.UUID(session_id)

            # Upsert music analysis
            await db.execute(
                delete(MusicAnalysis).where(MusicAnalysis.session_id == sid)
            )
            ma = MusicAnalysis(
                session_id=sid,
                tempo_bpm=tempo_result["tempo_bpm"],
                tempo_confidence=tempo_result["tempo_confidence"],
                key_label=key_result["key_label"],
                key_scale=key_result["key_scale"],
                key_confidence=key_result["key_confidence"],
                beat_times=tempo_result["beat_times"],
                downbeat_times=tempo_result["downbeat_times"],
                time_signature=tempo_result["time_signature"],
            )
            db.add(ma)

            # Upsert chord segments
            await db.execute(
                delete(MusicChordSegment).where(MusicChordSegment.session_id == sid)
            )
            for cs in chord_segments:
                db.add(MusicChordSegment(
                    session_id=sid,
                    t_start=cs["t_start"],
                    t_end=cs["t_end"],
                    label=cs["label"],
                    confidence=cs["confidence"],
                ))

            await db.commit()

        # Publish to aligner
        await self.publish("sap:align:pending", {
            "session_id": session_id,
            "source": "music",
            "payload": json.dumps({
                "tempo": tempo_result,
                "key": key_result,
                "chords": chord_segments,
                "features": frame_features,
            }),
        })


async def main():
    logging.basicConfig(level=logging.INFO)
    worker = MusicAnalysisWorker()
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
