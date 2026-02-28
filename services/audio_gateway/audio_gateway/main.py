from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import valkey.asyncio as valkey_async
from fastapi import FastAPI

from sap_common.config import settings
from sap_common.db import engine
from sap_common.health import add_health_routes
from sap_common.minio_client import ensure_bucket, get_minio_client
from sap_common.models import Base

from audio_gateway.routes.sessions import router as sessions_router
from audio_gateway.routes.upload import router as upload_router
from audio_gateway.routes.results import router as results_router
from audio_gateway.routes.ws import router as ws_router
from audio_gateway.routes.tts import router as tts_router
from audio_gateway.routes.flamingo import router as flamingo_router
from audio_gateway.ws_manager import ws_manager

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.basicConfig(level=getattr(logging, settings.log_level))
    logger.info("Starting Audio Gateway...")

    # Create tables (dev convenience — use alembic in prod)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Ensure MinIO bucket
    await ensure_bucket()

    # Start WebSocket relay (background task consuming Valkey streams)
    await ws_manager.start()

    yield

    # Shutdown
    await ws_manager.stop()
    await engine.dispose()


app = FastAPI(
    title="SAP Audio Gateway",
    description="Structured Acoustic Perception - Audio Processing API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS is handled by Nginx on the VM. Do not add CORSMiddleware here
# to prevent duplicate CORS headers which cause browsers to reject the request.

app.include_router(sessions_router, prefix="/v1/audio", tags=["sessions"])
app.include_router(upload_router, prefix="/v1/audio", tags=["upload"])
app.include_router(results_router, prefix="/v1/audio", tags=["results"])
app.include_router(ws_router, tags=["websocket"])
app.include_router(tts_router, prefix="/v1/audio", tags=["tts"])
app.include_router(flamingo_router, prefix="/v1/audio", tags=["flamingo"])


# Health checks
async def check_postgres():
    async with engine.connect() as conn:
        await conn.execute(Base.metadata.tables["audio_sessions"].select().limit(0))


async def check_valkey():
    client = valkey_async.from_url(settings.valkey_url)
    try:
        await client.ping()
    finally:
        await client.aclose()


async def check_minio():
    client = get_minio_client()
    await client.bucket_exists(settings.minio_bucket)


add_health_routes(
    app,
    checks={
        "postgres": check_postgres,
        "valkey": check_valkey,
        "minio": check_minio,
    },
)
