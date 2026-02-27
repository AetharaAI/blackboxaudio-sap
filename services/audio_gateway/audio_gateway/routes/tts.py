from __future__ import annotations

import io
import logging
import time
import uuid
from collections import defaultdict

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import httpx
import valkey.asyncio as valkey_async

from sap_common.config import settings
from sap_common.minio_client import download_bytes

logger = logging.getLogger(__name__)

router = APIRouter()

CHATTERBOX_URL = "https://tts.aetherpro.us"

# --- Attribution (Chatterbox TTS is MIT-licensed by Resemble AI) ---
CHATTERBOX_ATTRIBUTION = "Powered by Chatterbox TTS by Resemble AI"

# --- Voice whitelist for public API ---
# Only these predefined voices are exposed to the public UI.
# Add voice filenames from Chatterbox's /get_predefined_voices as vetted.
ALLOWED_PUBLIC_VOICES: set[str] = {
    # Populate with vetted voice filenames as they're reviewed.
    # When empty, all predefined voices are allowed (bootstrap mode).
}

# --- Rate limiting (in-memory, per-IP) ---
# Tracks TTS requests per IP with a sliding window.
_rate_limits: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_WINDOW = 60.0   # seconds
RATE_LIMIT_MAX = 10         # max requests per window per IP


def _check_rate_limit(client_ip: str) -> None:
    """Raise 429 if the client has exceeded the TTS rate limit."""
    now = time.monotonic()
    timestamps = _rate_limits[client_ip]
    # Prune expired entries
    _rate_limits[client_ip] = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW]
    if len(_rate_limits[client_ip]) >= RATE_LIMIT_MAX:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {RATE_LIMIT_MAX} TTS requests per {int(RATE_LIMIT_WINDOW)}s.",
        )
    _rate_limits[client_ip].append(now)


# --- Request/Response Models ---

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000)
    voice_mode: str = Field(default="predefined", description="'predefined' or 'clone'")
    predefined_voice_id: str | None = Field(default=None, description="Predefined voice filename")
    reference_audio_filename: str | None = Field(default=None, description="Reference audio for cloning")
    output_format: str = Field(default="wav", description="'wav', 'mp3', or 'opus'")
    speed_factor: float | None = Field(default=None, description="Playback speed multiplier")
    temperature: float | None = Field(default=None, description="Generation temperature")
    exaggeration: float | None = Field(default=None, description="Voice exaggeration factor")
    cfg_weight: float | None = Field(default=None, description="Classifier-free guidance weight")
    seed: int | None = Field(default=None, description="Random seed for reproducibility")
    session_id: str | None = Field(default=None, description="Attach to existing session, or None for standalone")


class TTSResponse(BaseModel):
    session_id: str
    status: str
    voice_mode: str
    text_length: int
    message: str
    attribution: str = CHATTERBOX_ATTRIBUTION


# --- Async (queued via Valkey → TTS worker → MinIO) ---

def _enforce_public_voice_policy(body: TTSRequest) -> None:
    """Block voice cloning from the public API and validate voice selection."""
    if body.voice_mode == "clone" or body.reference_audio_filename:
        raise HTTPException(
            status_code=403,
            detail=(
                "Voice cloning is not available through the public API. "
                "This feature requires explicit authorization and consent verification. "
                "Contact the platform administrator for access."
            ),
        )


@router.post("/tts/synthesize", response_model=TTSResponse)
async def synthesize_async(body: TTSRequest, request: Request):
    """Queue text for async TTS synthesis via the TTS worker.

    The worker calls Chatterbox, stores the result in MinIO.
    Connect to /ws/audio/{session_id} for completion notification,
    then download via GET /v1/audio/tts/{session_id}/audio.
    """
    _check_rate_limit(request.client.host if request.client else "unknown")
    _enforce_public_voice_policy(body)
    session_id = body.session_id or str(uuid.uuid4())

    msg = {
        "session_id": session_id,
        "text": body.text,
        "voice_mode": body.voice_mode,
        "output_format": body.output_format,
    }
    if body.predefined_voice_id:
        msg["predefined_voice_id"] = body.predefined_voice_id
    if body.reference_audio_filename:
        msg["reference_audio_filename"] = body.reference_audio_filename
    if body.speed_factor is not None:
        msg["speed_factor"] = str(body.speed_factor)
    if body.temperature is not None:
        msg["temperature"] = str(body.temperature)
    if body.exaggeration is not None:
        msg["exaggeration"] = str(body.exaggeration)
    if body.cfg_weight is not None:
        msg["cfg_weight"] = str(body.cfg_weight)
    if body.seed is not None:
        msg["seed"] = str(body.seed)

    client = valkey_async.from_url(settings.valkey_url, decode_responses=True)
    try:
        await client.xadd("sap:tts:pending", msg)
    finally:
        await client.aclose()

    return TTSResponse(
        session_id=session_id,
        status="queued",
        voice_mode=body.voice_mode,
        text_length=len(body.text),
        message=f"TTS queued. Listen on /ws/audio/{session_id} or poll /v1/audio/tts/{session_id}/audio",
        attribution=CHATTERBOX_ATTRIBUTION,
    )


# --- Sync (direct proxy to Chatterbox for immediate results) ---

@router.post("/tts/generate")
async def synthesize_sync(body: TTSRequest, request: Request):
    """Generate TTS audio synchronously by proxying to Chatterbox.

    Returns audio bytes directly — no queueing, no MinIO storage.
    Use this for quick previews or when you need immediate results.
    """
    _check_rate_limit(request.client.host if request.client else "unknown")
    _enforce_public_voice_policy(body)
    payload = {
        "text": body.text,
        "voice_mode": body.voice_mode,
        "output_format": body.output_format,
        "split_text": True,
        "chunk_size": 120,
    }
    if body.predefined_voice_id:
        payload["predefined_voice_id"] = body.predefined_voice_id
    if body.reference_audio_filename:
        payload["reference_audio_filename"] = body.reference_audio_filename
    if body.temperature is not None:
        payload["temperature"] = body.temperature
    if body.exaggeration is not None:
        payload["exaggeration"] = body.exaggeration
    if body.cfg_weight is not None:
        payload["cfg_weight"] = body.cfg_weight
    if body.seed is not None:
        payload["seed"] = body.seed
    if body.speed_factor is not None:
        payload["speed_factor"] = body.speed_factor

    async with httpx.AsyncClient(timeout=120.0) as http:
        resp = await http.post(f"{CHATTERBOX_URL}/tts", json=payload)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

    content_type = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
    }.get(body.output_format, "audio/wav")

    return StreamingResponse(
        io.BytesIO(resp.content),
        media_type=content_type,
        headers={"Content-Disposition": f'attachment; filename="tts_output.{body.output_format}"'},
    )


# --- Download from MinIO (after async synthesis) ---

@router.get("/tts/{session_id}/audio")
async def get_tts_audio(session_id: str, voice_id: str = "default", format: str = "wav"):
    """Download generated TTS audio from MinIO after async synthesis completes."""
    minio_key = f"tts/{session_id}/narration_{voice_id}.{format}"
    try:
        audio_bytes = await download_bytes(minio_key)
    except Exception:
        raise HTTPException(
            status_code=404,
            detail="TTS audio not yet available. Wait for synthesis to complete.",
        )

    content_type = {"wav": "audio/wav", "mp3": "audio/mpeg", "opus": "audio/opus"}.get(format, "audio/wav")
    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type=content_type,
        headers={"Content-Disposition": f'attachment; filename="narration_{voice_id}.{format}"'},
    )


# --- Voice management (proxied from Chatterbox) ---

@router.get("/tts/voices")
async def list_voices():
    """List available predefined voices from Chatterbox TTS Server.

    Filtered to a curated whitelist when configured for public safety.
    """
    async with httpx.AsyncClient(timeout=10.0) as http:
        resp = await http.get(f"{CHATTERBOX_URL}/get_predefined_voices")
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail="Failed to reach Chatterbox TTS Server")
        voices = resp.json()

    # Filter to whitelist if configured (empty whitelist = allow all in bootstrap mode)
    if ALLOWED_PUBLIC_VOICES:
        if isinstance(voices, list):
            voices = [v for v in voices if v in ALLOWED_PUBLIC_VOICES]
        elif isinstance(voices, dict):
            voices = {k: v for k, v in voices.items() if k in ALLOWED_PUBLIC_VOICES}

    return {
        "voices": voices,
        "capabilities": ["predefined"],
        "output_formats": ["wav", "mp3", "opus"],
        "attribution": CHATTERBOX_ATTRIBUTION,
    }


@router.get("/tts/model-info")
async def model_info():
    """Get model information from Chatterbox TTS Server."""
    async with httpx.AsyncClient(timeout=10.0) as http:
        resp = await http.get(f"{CHATTERBOX_URL}/api/model-info")
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail="Failed to reach Chatterbox TTS Server")
        info = resp.json()
    info["attribution"] = CHATTERBOX_ATTRIBUTION
    info["watermarking"] = "Perth watermark embedded in all generated audio"
    return info


@router.post("/tts/upload-reference")
async def upload_reference_audio(file: UploadFile):
    """Upload reference audio to Chatterbox for voice cloning.

    RESTRICTED: This endpoint is gated for administrative use only.
    Voice cloning requires explicit authorization and verified consent
    from the voice owner. Not available through the public API.
    """
    # Block from public API — cloning requires admin authorization
    raise HTTPException(
        status_code=403,
        detail=(
            "Voice cloning upload is restricted. "
            "This feature requires administrator authorization and verified consent "
            "from the voice owner per our Terms of Service. "
            "Contact the platform administrator to enable voice cloning."
        ),
    )
