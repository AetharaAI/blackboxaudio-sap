from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sap_common.db import get_session
from sap_common.models import AudioSession
from sap_common.schemas import SessionCreate, SessionResponse

router = APIRouter()


@router.post("/sessions", response_model=SessionResponse, status_code=201)
async def create_session(
    body: SessionCreate,
    db: AsyncSession = Depends(get_session),
):
    session = AudioSession(
        tenant_id=body.tenant_id,
        user_id=body.user_id,
        metadata_=body.metadata,
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return _to_response(session)


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session_info(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_session),
):
    session = await db.get(AudioSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return _to_response(session)


@router.get("/sessions", response_model=list[SessionResponse])
async def list_sessions(
    tenant_id: str | None = None,
    limit: int = 50,
    db: AsyncSession = Depends(get_session),
):
    stmt = (
        select(AudioSession)
        .order_by(AudioSession.created_at.desc())
        .limit(limit)
    )
    if tenant_id:
        stmt = stmt.where(AudioSession.tenant_id == tenant_id)
    result = await db.execute(stmt)
    sessions = result.scalars().all()
    return [_to_response(s) for s in sessions]


def _to_response(session: AudioSession) -> SessionResponse:
    return SessionResponse(
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
        ws_url=f"/ws/audio/{session.id}",
    )
