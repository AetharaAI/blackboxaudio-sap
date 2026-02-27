from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import uuid
from abc import abstractmethod

import valkey.asyncio as valkey

from sap_common.config import settings

logger = logging.getLogger(__name__)


class StreamWorker:
    """Base class for Valkey Stream consumers with consumer group support.

    Handles:
    - Consumer group creation (XGROUP CREATE ... MKSTREAM)
    - XREADGROUP with block timeout
    - Message acknowledgment (XACK) after successful processing
    - Pending message recovery on startup
    - Dead letter: after MAX_RETRIES failed attempts, move to DLQ stream
    - Graceful shutdown on SIGTERM/SIGINT
    """

    STREAM: str = ""
    GROUP: str = ""
    MAX_RETRIES: int = 3
    BLOCK_MS: int = 5000
    CLAIM_IDLE_MS: int = 30000

    def __init__(self):
        self.consumer_name = f"{self.GROUP}-{uuid.uuid4().hex[:8]}"
        self._running = False
        self._client: valkey.Valkey | None = None

    async def get_client(self) -> valkey.Valkey:
        if self._client is None:
            self._client = valkey.from_url(
                settings.valkey_url,
                decode_responses=True,
                socket_timeout=self.BLOCK_MS / 1000 + 5,
                socket_connect_timeout=5,
            )
        return self._client

    @abstractmethod
    async def process(self, message_id: str, data: dict) -> None:
        """Override in subclass. Raise to NACK."""

    async def publish(self, stream: str, data: dict) -> str:
        """Publish a message to a Valkey stream."""
        client = await self.get_client()
        # Valkey XADD expects flat string key-value pairs
        flat = {}
        for k, v in data.items():
            flat[k] = v if isinstance(v, str) else json.dumps(v)
        msg_id = await client.xadd(stream, flat)
        return msg_id

    async def update_session_status(
        self, session_id: str, status: str, error: str = ""
    ) -> None:
        """Publish a session status update."""
        await self.publish(
            "sap:session:status",
            {"session_id": session_id, "status": status, "error": error},
        )

    async def _ensure_group(self) -> None:
        client = await self.get_client()
        try:
            await client.xgroup_create(
                self.STREAM, self.GROUP, id="0", mkstream=True
            )
            logger.info(
                "Created consumer group %s on stream %s", self.GROUP, self.STREAM
            )
        except valkey.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    async def _get_attempt_count(self, message_id: str) -> int:
        client = await self.get_client()
        key = f"sap:retries:{self.STREAM}:{message_id}"
        val = await client.get(key)
        return int(val) if val else 0

    async def _set_attempt_count(self, message_id: str, count: int) -> None:
        client = await self.get_client()
        key = f"sap:retries:{self.STREAM}:{message_id}"
        await client.set(key, str(count), ex=3600)  # expire in 1h

    async def _handle_message(self, msg_id: str, data: dict) -> None:
        attempt = await self._get_attempt_count(msg_id)
        client = await self.get_client()
        try:
            await self.process(msg_id, data)
            await client.xack(self.STREAM, self.GROUP, msg_id)
            # Clean up retry counter
            retry_key = f"sap:retries:{self.STREAM}:{msg_id}"
            await client.delete(retry_key)
        except Exception:
            attempt += 1
            logger.exception(
                "Failed processing message %s (attempt %d/%d)",
                msg_id,
                attempt,
                self.MAX_RETRIES,
            )
            if attempt >= self.MAX_RETRIES:
                # Move to dead letter queue
                dlq_data = {**data, "original_id": msg_id, "attempts": str(attempt)}
                await client.xadd(f"{self.STREAM}:dlq", dlq_data)
                await client.xack(self.STREAM, self.GROUP, msg_id)
                # Update session status to failed
                session_id = data.get("session_id", "unknown")
                await self.update_session_status(
                    session_id, "failed", f"Exceeded {self.MAX_RETRIES} retries"
                )
            else:
                await self._set_attempt_count(msg_id, attempt)

    async def run(self) -> None:
        """Main consumer loop."""
        self._running = True
        loop = asyncio.get_event_loop()

        def _stop():
            logger.info("Received shutdown signal, stopping worker...")
            self._running = False

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, _stop)

        await self._ensure_group()
        client = await self.get_client()

        logger.info(
            "Worker %s started, consuming %s in group %s",
            self.consumer_name,
            self.STREAM,
            self.GROUP,
        )

        # Phase 1: recover pending messages
        pending = await client.xreadgroup(
            self.GROUP,
            self.consumer_name,
            {self.STREAM: "0"},
            count=10,
        )
        for stream_name, messages in pending:
            for msg_id, data in messages:
                if data:  # skip acknowledged-but-not-yet-trimmed
                    await self._handle_message(msg_id, data)

        # Phase 2: consume new messages
        while self._running:
            try:
                result = await client.xreadgroup(
                    self.GROUP,
                    self.consumer_name,
                    {self.STREAM: ">"},
                    count=1,
                    block=self.BLOCK_MS,
                )
                if result:
                    for stream_name, messages in result:
                        for msg_id, data in messages:
                            await self._handle_message(msg_id, data)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in consumer loop, retrying in 1s...")
                await asyncio.sleep(1)

        # Cleanup
        if self._client:
            await self._client.aclose()
        logger.info("Worker %s stopped", self.consumer_name)
