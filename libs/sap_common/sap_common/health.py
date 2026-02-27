from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse


def add_health_routes(
    app: FastAPI,
    checks: dict[str, Callable[[], Coroutine[Any, Any, None]]] | None = None,
) -> None:
    """Add liveness and readiness probe endpoints."""

    checks = checks or {}

    @app.get("/health/live")
    async def liveness():
        return {"status": "ok"}

    @app.get("/health/ready")
    async def readiness():
        results = {}
        all_ok = True
        for name, check_fn in checks.items():
            try:
                await check_fn()
                results[name] = "ok"
            except Exception as e:
                results[name] = str(e)
                all_ok = False
        status_code = 200 if all_ok else 503
        return JSONResponse({"checks": results}, status_code=status_code)
