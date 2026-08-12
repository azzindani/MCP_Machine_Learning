"""Combined HTTP entry point — all 3 tiers in ONE process, ONE port.

Each tier (basic/medium/advanced) keeps its own server.py for stdio /
individual-HTTP use (LM Studio "add one tier" installs, local dev). This
file is Docker/remote-deployment-only: it imports each tier's already-built
FastMCP instance and mounts its HTTP app at its own path prefix inside one
Starlette app, so pandas/numpy/scikit-learn/xgboost load ONCE instead of
three times. Each tier's own /health, /version, and /mcp routes (added via
@mcp.custom_route in its own server.py) come along for free under the
mount prefix — nothing tier-specific is duplicated here.

Lifespans do NOT propagate through Starlette's Mount() automatically, so
each tier's session-manager lifespan is entered explicitly via
AsyncExitStack — verified live before wiring this up for real (see
project memory / commit history for the throwaway prototype that proved
this against 2 real tiers before committing to the pattern).
"""

from __future__ import annotations

import argparse
import os
from contextlib import AsyncExitStack, asynccontextmanager

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse
from starlette.routing import Mount, Route

from servers.ml_advanced.server import mcp as advanced_mcp
from servers.ml_basic.server import mcp as basic_mcp
from servers.ml_medium.server import mcp as medium_mcp

_VERSION = "0.1.0"

_TIERS = {
    "basic": basic_mcp,
    "medium": medium_mcp,
    "advanced": advanced_mcp,
}
_sub_apps = {name: mcp.http_app(path="/mcp") for name, mcp in _TIERS.items()}


@asynccontextmanager
async def _combined_lifespan(app):
    async with AsyncExitStack() as stack:
        for sub_app in _sub_apps.values():
            await stack.enter_async_context(sub_app.lifespan(sub_app))
        yield


async def _root_health(request: Request) -> JSONResponse:
    """Aggregate liveness check. Unauthenticated."""
    return JSONResponse({"status": "ok", "version": _VERSION, "tiers": list(_TIERS)})


async def _root_version(request: Request) -> JSONResponse:
    """Report running version. Unauthenticated."""
    return JSONResponse({"current": _VERSION})


async def _root(request: Request) -> JSONResponse:
    return JSONResponse(
        {
            "server": "MCP_Machine_Learning",
            "tiers": {name: f"/{name}/mcp" for name in _TIERS},
        }
    )


def _redirect(target: str):
    """308 redirect to a tier's real well-known route.

    RFC 8414/9728 clients build discovery URLs by inserting
    `/.well-known/...` between the origin and the resource/issuer path
    (e.g. `/.well-known/oauth-protected-resource/basic/mcp`), landing at the
    OUTER app's root. But Mount() nests each tier's real well-known routes
    under its own prefix (`/basic/.well-known/...`) instead, so the
    client's computed URL 404s without this redirect — confirmed live
    against a real unauthenticated claude.ai connector attempt.
    """

    async def _handler(request: Request) -> RedirectResponse:
        return RedirectResponse(target, status_code=308)

    return _handler


_discovery_redirects = [
    route
    for name in _TIERS
    for route in (
        Route(
            f"/.well-known/oauth-protected-resource/{name}/mcp",
            _redirect(f"/{name}/.well-known/oauth-protected-resource"),
        ),
        Route(
            f"/.well-known/oauth-authorization-server/{name}",
            _redirect(f"/{name}/.well-known/oauth-authorization-server"),
        ),
    )
]

app = Starlette(
    routes=[
        Route("/health", _root_health),
        Route("/version", _root_version),
        Route("/", _root),
        *_discovery_redirects,
        *(Mount(f"/{name}", app=sub_app) for name, sub_app in _sub_apps.items()),
    ],
    lifespan=_combined_lifespan,
)


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="MCP_Machine_Learning unified server")
    parser.add_argument("--host", default=os.environ.get("ML_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("ML_PORT", "8820")))
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
