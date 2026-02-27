from __future__ import annotations

import asyncio
import json
import logging
import uuid

from sqlalchemy import select, update

from sap_common.config import settings
from sap_common.db import async_session_factory
from sap_common.models import AudioSession, PerceptionFrame, SessionStatus
from sap_common.valkey_streams import StreamWorker

from aligner.fusion import build_perception_frames

logger = logging.getLogger(__name__)


class AlignerWorker(StreamWorker):
    """Consumes analysis results from ASR and music workers, fuses into Perception Frames.

    Coordination: tracks which pipelines have reported per session via Valkey hashes.
    Waits for both music + asr_final before producing final frames.
    """

    STREAM = "sap:align:pending"
    GROUP = "align-workers"

    async def process(self, message_id: str, data: dict) -> None:
        session_id = data["session_id"]
        source = data.get("source", "")
        payload_str = data.get("payload", "{}")

        if isinstance(payload_str, str):
            payload = json.loads(payload_str)
        else:
            payload = payload_str

        client = await self.get_client()
        tracker_key = f"sap:align:tracker:{session_id}"

        if source == "music":
            # Store music results in tracker
            await client.hset(tracker_key, "music", json.dumps(payload))
            await client.hset(tracker_key, "music_received", "1")
            await client.expire(tracker_key, 3600)
            logger.info("Aligner received music data for %s", session_id)

        elif source == "asr":
            # Streaming ASR partial — store latest
            existing = await client.hget(tracker_key, "asr_partials")
            partials = json.loads(existing) if existing else []
            partials.append(payload)
            await client.hset(tracker_key, "asr_partials", json.dumps(partials))
            await client.expire(tracker_key, 3600)

        elif source == "asr_final":
            # Final ASR — store definitive transcript
            await client.hset(tracker_key, "asr_final", json.dumps(payload))
            await client.hset(tracker_key, "asr_final_received", "1")
            await client.expire(tracker_key, 3600)
            logger.info("Aligner received final ASR data for %s", session_id)

        # Check if we have both music + asr_final
        music_received = await client.hget(tracker_key, "music_received")
        asr_final_received = await client.hget(tracker_key, "asr_final_received")

        if music_received and asr_final_received:
            await self._build_and_store_frames(session_id, tracker_key)

    async def _build_and_store_frames(
        self, session_id: str, tracker_key: str
    ) -> None:
        """Build final perception frames and store/publish them."""
        client = await self.get_client()

        # Retrieve all data from tracker
        music_data = json.loads(await client.hget(tracker_key, "music") or "{}")
        asr_data = json.loads(await client.hget(tracker_key, "asr_final") or "[]")

        # Get session duration from DB
        sid = uuid.UUID(session_id)
        async with async_session_factory() as db:
            session = await db.get(AudioSession, sid)
            duration_sec = session.duration_sec if session else 0.0

            if not duration_sec and music_data.get("features"):
                # Estimate from features
                features = music_data["features"]
                if features:
                    duration_sec = features[-1]["t"] + 0.250

        if not duration_sec:
            logger.warning("No duration for session %s, skipping frame build", session_id)
            return

        # Build perception frames
        frames = build_perception_frames(
            session_id=session_id,
            duration_sec=duration_sec,
            audio_features=music_data.get("features", []),
            tempo_result=music_data.get("tempo", {}),
            key_result=music_data.get("key", {}),
            chord_segments=music_data.get("chords", []),
            transcript_segments=asr_data if isinstance(asr_data, list) else [],
        )

        logger.info(
            "Built %d perception frames for session %s (%.1fs)",
            len(frames), session_id, duration_sec,
        )

        # Store frames in Postgres
        async with async_session_factory() as db:
            # Clear old frames
            from sqlalchemy import delete
            await db.execute(
                delete(PerceptionFrame).where(PerceptionFrame.session_id == sid)
            )

            for frame in frames:
                db.add(PerceptionFrame(
                    session_id=sid,
                    t=frame["t"],
                    frame_data=frame,
                ))

            # Update session status to completed
            await db.execute(
                update(AudioSession)
                .where(AudioSession.id == sid)
                .values(
                    status=SessionStatus.completed,
                    duration_sec=duration_sec,
                )
            )
            await db.commit()

        # Publish frames to results stream for WebSocket relay
        # Send as a batch for efficiency
        await self.publish("sap:results:stream", {
            "session_id": session_id,
            "frame": json.dumps({"frame_count": len(frames)}),
            "is_final": "true",
        })

        # Publish completion status
        await self.update_session_status(session_id, "completed")

        # Clean up tracker
        await client.delete(tracker_key)

        logger.info("Session %s completed", session_id)


async def main():
    logging.basicConfig(level=logging.INFO)
    worker = AlignerWorker()
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
