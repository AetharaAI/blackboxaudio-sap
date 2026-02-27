from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from sap_common.config import settings
from sap_common.db import get_session
from sap_common.minio_client import upload_bytes
from sap_common.models import AudioSession, SessionStatus
from sap_common.valkey_streams import StreamWorker

import valkey.asyncio as valkey_async

router = APIRouter()

ALLOWED_MIME_TYPES = {
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/x-wav",
    "audio/flac",
    "audio/x-flac",
    "audio/aac",
    "audio/mp4",
    "audio/ogg",
    "audio/webm",
    "application/octet-stream",  # fallback for unknown
}

MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB


@router.post("/sessions/{session_id}/upload")
async def upload_audio(
    session_id: uuid.UUID,
    file: UploadFile,
    db: AsyncSession = Depends(get_session),
):
    # Validate session exists
    session = await db.get(AudioSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status not in (SessionStatus.created, SessionStatus.uploading):
        raise HTTPException(
            status_code=409,
            detail=f"Session is in state '{session.status.value}', cannot upload",
        )

    # Validate file type
    content_type = file.content_type or "application/octet-stream"
    if content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio format: {content_type}",
        )

    # Read file data
    data = await file.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large (max 500MB)")
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    # Determine file extension
    filename = file.filename or "audio"
    ext = filename.rsplit(".", 1)[-1] if "." in filename else "bin"

    # Upload to MinIO
    minio_key = f"raw/{session_id}/{filename}"
    await upload_bytes(minio_key, data, content_type=content_type)

    # Update session
    session.status = SessionStatus.uploading
    session.filename = filename
    session.mime_type = content_type
    session.minio_raw_key = minio_key
    await db.commit()

    # Publish to preprocessing queue
    client = valkey_async.from_url(settings.valkey_url, decode_responses=True)
    try:
        await client.xadd(
            "sap:preprocess:pending",
            {
                "session_id": str(session_id),
                "minio_key": minio_key,
                "mime_type": content_type,
            },
        )
    finally:
        await client.aclose()

    return {
        "status": "uploaded",
        "session_id": str(session_id),
        "filename": filename,
        "size_bytes": len(data),
        "minio_key": minio_key,
    }
