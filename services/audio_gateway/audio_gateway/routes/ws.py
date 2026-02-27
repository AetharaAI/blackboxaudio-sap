from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from audio_gateway.ws_manager import ws_manager

router = APIRouter()


@router.websocket("/ws/audio/{session_id}")
async def audio_websocket(websocket: WebSocket, session_id: uuid.UUID):
    await websocket.accept()
    session_key = str(session_id)
    ws_manager.connect(session_key, websocket)

    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connected",
            "session_id": session_key,
        })

        # Keep connection alive, handle client messages
        while True:
            try:
                data = await websocket.receive_text()
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except WebSocketDisconnect:
                break
    finally:
        ws_manager.disconnect(session_key, websocket)
