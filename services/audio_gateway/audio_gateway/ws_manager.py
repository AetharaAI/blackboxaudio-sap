from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict

import valkey.asyncio as valkey_async
from fastapi import WebSocket

from sap_common.config import settings

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections and relays Valkey stream events to clients.

    Subscribes to:
    - sap:results:stream  (perception frames from aligner)
    - sap:session:status  (status updates from all workers)

    Fans out messages to connected WebSocket clients keyed by session_id.
    """

    def __init__(self):
        self._connections: dict[str, set[WebSocket]] = defaultdict(set)
        self._relay_task: asyncio.Task | None = None
        self._running = False

    def connect(self, session_id: str, ws: WebSocket) -> None:
        self._connections[session_id].add(ws)
        logger.info("WebSocket connected for session %s (total: %d)",
                     session_id, len(self._connections[session_id]))

    def disconnect(self, session_id: str, ws: WebSocket) -> None:
        self._connections[session_id].discard(ws)
        if not self._connections[session_id]:
            del self._connections[session_id]
        logger.info("WebSocket disconnected for session %s", session_id)

    async def broadcast(self, session_id: str, message: dict) -> None:
        """Send a message to all WebSocket clients for a session."""
        websockets = self._connections.get(session_id, set()).copy()
        dead = []
        for ws in websockets:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._connections[session_id].discard(ws)

    async def start(self) -> None:
        """Start the Valkey stream relay background task."""
        self._running = True
        self._relay_task = asyncio.create_task(self._relay_loop())
        logger.info("WebSocket relay started")

    async def stop(self) -> None:
        """Stop the relay task."""
        self._running = False
        if self._relay_task:
            self._relay_task.cancel()
            try:
                await self._relay_task
            except asyncio.CancelledError:
                pass
        logger.info("WebSocket relay stopped")

    async def _relay_loop(self) -> None:
        """Consume from Valkey streams and fan out to WebSocket clients."""
        client = valkey_async.from_url(settings.valkey_url, decode_responses=True)
        group = "gateway-relays"
        consumer = "gateway-0"
        streams = ["sap:results:stream", "sap:session:status", "sap:tts:complete"]

        # Create consumer groups
        for stream in streams:
            try:
                await client.xgroup_create(stream, group, id="0", mkstream=True)
            except valkey_async.ResponseError as e:
                if "BUSYGROUP" not in str(e):
                    raise

        while self._running:
            try:
                result = await client.xreadgroup(
                    group,
                    consumer,
                    {s: ">" for s in streams},
                    count=10,
                    block=1000,
                )
                if not result:
                    continue

                for stream_name, messages in result:
                    for msg_id, data in messages:
                        session_id = data.get("session_id", "")
                        if not session_id:
                            await client.xack(stream_name, group, msg_id)
                            continue

                        # Build WebSocket message based on stream
                        if stream_name == "sap:session:status":
                            ws_msg = {
                                "type": "status",
                                "session_id": session_id,
                                "status": data.get("status", ""),
                                "error": data.get("error", ""),
                            }
                        elif stream_name == "sap:results:stream":
                            frame_data = data.get("frame", "{}")
                            if isinstance(frame_data, str):
                                try:
                                    frame_data = json.loads(frame_data)
                                except json.JSONDecodeError:
                                    frame_data = {}
                            ws_msg = {
                                "type": "frame",
                                "session_id": session_id,
                                "is_final": data.get("is_final", "false") == "true",
                                "data": frame_data,
                            }
                        elif stream_name == "sap:tts:complete":
                            ws_msg = {
                                "type": "tts_complete",
                                "session_id": session_id,
                                "minio_key": data.get("minio_key", ""),
                                "voice": data.get("voice", ""),
                                "audio_size_bytes": data.get("audio_size_bytes", "0"),
                            }
                        else:
                            ws_msg = {"type": "unknown", "data": data}

                        await self.broadcast(session_id, ws_msg)
                        await client.xack(stream_name, group, msg_id)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in WebSocket relay loop")
                await asyncio.sleep(1)

        await client.aclose()


ws_manager = WebSocketManager()
