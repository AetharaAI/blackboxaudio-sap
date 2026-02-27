from __future__ import annotations

import io
import logging

from miniopy_async import Minio

from sap_common.config import settings

logger = logging.getLogger(__name__)


def get_minio_client() -> Minio:
    return Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )


async def ensure_bucket(client: Minio | None = None) -> None:
    """Create the SAP audio bucket if it doesn't exist."""
    client = client or get_minio_client()
    bucket = settings.minio_bucket
    if not await client.bucket_exists(bucket):
        await client.make_bucket(bucket)
        logger.info("Created MinIO bucket: %s", bucket)


async def upload_bytes(
    key: str,
    data: bytes,
    content_type: str = "application/octet-stream",
    client: Minio | None = None,
) -> str:
    """Upload bytes to MinIO and return the object key."""
    client = client or get_minio_client()
    bucket = settings.minio_bucket
    await ensure_bucket(client)
    await client.put_object(
        bucket,
        key,
        io.BytesIO(data),
        length=len(data),
        content_type=content_type,
    )
    logger.info("Uploaded %s to %s (%d bytes)", key, bucket, len(data))
    return key


async def download_bytes(
    key: str,
    client: Minio | None = None,
) -> bytes:
    """Download an object from MinIO and return its bytes."""
    client = client or get_minio_client()
    bucket = settings.minio_bucket
    response = await client.get_object(bucket, key)
    try:
        data = await response.read()
    finally:
        response.close()
        await response.release()
    return data
