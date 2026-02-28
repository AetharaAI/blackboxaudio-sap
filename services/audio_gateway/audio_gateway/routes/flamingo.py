"""Flamingo analysis API routes — retrieve Music Flamingo analysis results."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sap_common.db import get_session
from sap_common.models import AudioSession, FlamingoAnalysis

router = APIRouter()


class FlamingoResult(BaseModel):
    prompt_key: str
    prompt_text: str
    response: str
    structured_data: dict


class FlamingoResponse(BaseModel):
    session_id: str
    analyses: list[FlamingoResult]


@router.get("/sessions/{session_id}/flamingo")
async def get_flamingo_analysis(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_session),
) -> FlamingoResponse:
    """Get all Music Flamingo analysis results for a session."""
    session = await db.get(AudioSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    result = await db.execute(
        select(FlamingoAnalysis)
        .where(FlamingoAnalysis.session_id == session_id)
        .order_by(FlamingoAnalysis.created_at)
    )
    analyses = result.scalars().all()

    return FlamingoResponse(
        session_id=str(session_id),
        analyses=[
            FlamingoResult(
                prompt_key=a.prompt_key,
                prompt_text=a.prompt_text,
                response=a.response,
                structured_data=a.structured_data or {},
            )
            for a in analyses
        ],
    )


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    session_id: str
    question: str
    status: str


@router.post("/sessions/{session_id}/flamingo/ask")
async def ask_flamingo(
    session_id: uuid.UUID,
    request: AskRequest,
    db: AsyncSession = Depends(get_session),
) -> AskResponse:
    """Submit a custom question about the audio to Music Flamingo.

    This queues the question for async processing via the Flamingo worker.
    """
    session = await db.get(AudioSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not session.minio_pcm_key:
        raise HTTPException(
            status_code=409,
            detail="Audio not yet preprocessed",
        )

    # Publish to Flamingo worker with custom prompt
    import valkey.asyncio as valkey_async
    from sap_common.config import settings

    client = valkey_async.from_url(settings.valkey_url, decode_responses=True)
    try:
        await client.xadd(
            "sap:flamingo:pending",
            {
                "session_id": str(session_id),
                "minio_key": session.minio_pcm_key,
                "sample_rate": str(session.sample_rate),
                "custom_prompt": request.question,
            },
        )
    finally:
        await client.aclose()

    return AskResponse(
        session_id=str(session_id),
        question=request.question,
        status="queued",
    )
