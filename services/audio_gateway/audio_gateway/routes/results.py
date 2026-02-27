from __future__ import annotations

import io
import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse, JSONResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sap_common.config import settings
from sap_common.db import get_session
from sap_common.minio_client import download_bytes, get_minio_client
from sap_common.models import (
    AudioSession,
    MusicAnalysis,
    MusicChordSegment,
    PerceptionFrame,
    TranscriptSegment,
)
from sap_common.schemas import (
    ChordSegmentSchema,
    MusicAnalysisSchema,
    PerceptionFrameSchema,
    SessionResponse,
    SessionResult,
    TranscriptSegmentSchema,
)

router = APIRouter()


@router.get("/sessions/{session_id}/result")
async def get_result(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_session),
):
    session = await db.get(AudioSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Music analysis
    ma_stmt = select(MusicAnalysis).where(MusicAnalysis.session_id == session_id)
    ma = (await db.execute(ma_stmt)).scalar_one_or_none()

    # Chords
    chords_stmt = (
        select(MusicChordSegment)
        .where(MusicChordSegment.session_id == session_id)
        .order_by(MusicChordSegment.t_start)
    )
    chords = (await db.execute(chords_stmt)).scalars().all()

    # Transcript (final pass only)
    transcript_stmt = (
        select(TranscriptSegment)
        .where(
            TranscriptSegment.session_id == session_id,
            TranscriptSegment.is_final == True,
        )
        .order_by(TranscriptSegment.t_start)
    )
    transcript = (await db.execute(transcript_stmt)).scalars().all()

    # Perception frames
    frames_stmt = (
        select(PerceptionFrame)
        .where(PerceptionFrame.session_id == session_id)
        .order_by(PerceptionFrame.t)
    )
    frames = (await db.execute(frames_stmt)).scalars().all()

    return SessionResult(
        session=SessionResponse(
            id=session.id,
            tenant_id=session.tenant_id,
            user_id=session.user_id,
            status=session.status.value,
            filename=session.filename,
            mime_type=session.mime_type,
            duration_sec=session.duration_sec,
            sample_rate=session.sample_rate,
            error_message=session.error_message,
            created_at=session.created_at,
            updated_at=session.updated_at,
        ),
        music_analysis=MusicAnalysisSchema.model_validate(ma) if ma else None,
        chords=[ChordSegmentSchema.model_validate(c) for c in chords],
        transcript=[TranscriptSegmentSchema.model_validate(t) for t in transcript],
        frames=[
            PerceptionFrameSchema(
                session_id=str(f.session_id),
                t=f.t,
                **(f.frame_data or {}),
            )
            for f in frames
        ],
    )


@router.get("/sessions/{session_id}/export/lyrics.txt")
async def export_lyrics(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_session),
):
    session = await db.get(AudioSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    stmt = (
        select(TranscriptSegment)
        .where(
            TranscriptSegment.session_id == session_id,
            TranscriptSegment.is_final == True,
        )
        .order_by(TranscriptSegment.t_start)
    )
    segments = (await db.execute(stmt)).scalars().all()
    text = "\n".join(seg.text for seg in segments)
    return PlainTextResponse(text, media_type="text/plain")


@router.get("/sessions/{session_id}/export/chords.json")
async def export_chords(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_session),
):
    session = await db.get(AudioSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Music analysis
    ma_stmt = select(MusicAnalysis).where(MusicAnalysis.session_id == session_id)
    ma = (await db.execute(ma_stmt)).scalar_one_or_none()

    # Chords
    chords_stmt = (
        select(MusicChordSegment)
        .where(MusicChordSegment.session_id == session_id)
        .order_by(MusicChordSegment.t_start)
    )
    chords = (await db.execute(chords_stmt)).scalars().all()

    result = {
        "session_id": str(session_id),
        "tempo_bpm": ma.tempo_bpm if ma else None,
        "key": f"{ma.key_label} {ma.key_scale}" if ma else None,
        "time_signature": ma.time_signature if ma else None,
        "chords": [
            {
                "t_start": c.t_start,
                "t_end": c.t_end,
                "label": c.label,
                "confidence": c.confidence,
            }
            for c in chords
        ],
    }
    return JSONResponse(result)


# --- Audio playback/streaming (gap #1) ---

@router.get("/sessions/{session_id}/stream")
async def stream_audio(
    session_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_session),
):
    """Stream the original uploaded audio for browser <audio> playback.

    Supports HTTP Range requests for seeking.
    """
    session = await db.get(AudioSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.minio_raw_key:
        raise HTTPException(status_code=404, detail="No audio file uploaded for this session")

    try:
        audio_bytes = await download_bytes(session.minio_raw_key)
    except Exception:
        raise HTTPException(status_code=404, detail="Audio file not found in storage")

    total = len(audio_bytes)
    content_type = session.mime_type or "audio/wav"

    # Handle Range requests for seeking
    range_header = request.headers.get("range")
    if range_header:
        range_spec = range_header.strip().lower().replace("bytes=", "")
        parts = range_spec.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else total - 1
        end = min(end, total - 1)
        length = end - start + 1
        return StreamingResponse(
            io.BytesIO(audio_bytes[start:end + 1]),
            status_code=206,
            media_type=content_type,
            headers={
                "Content-Range": f"bytes {start}-{end}/{total}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(length),
            },
        )

    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type=content_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(total),
            "Content-Disposition": f'inline; filename="{session.filename or "audio.wav"}"',
        },
    )


# --- Frame pagination (gap #2) ---

@router.get("/sessions/{session_id}/frames")
async def get_frames(
    session_id: uuid.UUID,
    t_start: float = Query(default=0.0, ge=0, description="Start time in seconds"),
    t_end: float | None = Query(default=None, ge=0, description="End time in seconds (None = all)"),
    db: AsyncSession = Depends(get_session),
):
    """Query perception frames within a time window.

    Use t_start/t_end for efficient windowed access during timeline scrubbing.
    At 250ms resolution, a 30s window returns ~120 frames.
    """
    session = await db.get(AudioSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    stmt = (
        select(PerceptionFrame)
        .where(
            PerceptionFrame.session_id == session_id,
            PerceptionFrame.t >= t_start,
        )
        .order_by(PerceptionFrame.t)
    )
    if t_end is not None:
        stmt = stmt.where(PerceptionFrame.t <= t_end)

    frames = (await db.execute(stmt)).scalars().all()

    return {
        "session_id": str(session_id),
        "t_start": t_start,
        "t_end": t_end,
        "count": len(frames),
        "frames": [
            PerceptionFrameSchema(
                session_id=str(f.session_id),
                t=f.t,
                **(f.frame_data or {}),
            )
            for f in frames
        ],
    }


# --- Export beats.json (gap #3) ---

@router.get("/sessions/{session_id}/export/beats.json")
async def export_beats(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_session),
):
    """Export beat positions for DAW integration or rhythm visualization."""
    session = await db.get(AudioSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    ma_stmt = select(MusicAnalysis).where(MusicAnalysis.session_id == session_id)
    ma = (await db.execute(ma_stmt)).scalar_one_or_none()

    result = {
        "session_id": str(session_id),
        "tempo_bpm": ma.tempo_bpm if ma else None,
        "time_signature": ma.time_signature if ma else None,
        "beat_times": ma.beat_times if ma else [],
        "downbeat_times": ma.downbeat_times if ma else [],
        "beat_count": len(ma.beat_times) if ma else 0,
    }
    return JSONResponse(result)


# --- Export frames.jsonl (gap #3) ---

@router.get("/sessions/{session_id}/export/frames.jsonl")
async def export_frames_jsonl(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_session),
):
    """Export all perception frames as newline-delimited JSON.

    Useful for ML pipelines, data analysis, and bulk processing.
    """
    session = await db.get(AudioSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    frames_stmt = (
        select(PerceptionFrame)
        .where(PerceptionFrame.session_id == session_id)
        .order_by(PerceptionFrame.t)
    )
    frames = (await db.execute(frames_stmt)).scalars().all()

    def generate_jsonl():
        for f in frames:
            line = {"session_id": str(f.session_id), "t": f.t, **(f.frame_data or {})}
            yield json.dumps(line) + "\n"

    return StreamingResponse(
        generate_jsonl(),
        media_type="application/x-ndjson",
        headers={
            "Content-Disposition": f'attachment; filename="{session_id}_frames.jsonl"',
        },
    )
