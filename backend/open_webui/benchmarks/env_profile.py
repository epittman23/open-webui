"""benchmarks/env_profile.py - resolve the serving profile, and what's actually running.

Ported from local-llm's scripts/llama_test.py (outer repo): profile(), port()
and served_model(). Kept in its own small module rather than folded into
runner.py because it is a different kind of dependency from the rest of that
file -- a shell-out to the outer local-llm repo's own operational layer, not
this app's request/grade/record logic.

scripts/llama-env.sh stays outside this app and is the single source of truth
for serving configuration; re-declaring the profile table here would create a
second one that could disagree with it. LLAMA_ENV_SH is the same env var
benchmarks/proc.py already requires for the same reason (it points a Command
at llama-env.sh to launch llama-server); this module shells out to it too,
independently, since resolving a profile does not require starting anything.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

import aiohttp

log = logging.getLogger(__name__)


def _env_sh() -> Path:
    configured = os.environ.get('LLAMA_ENV_SH')
    if not configured:
        raise RuntimeError(
            'LLAMA_ENV_SH is not set. It must point at the outer local-llm '
            "repo's scripts/llama-env.sh -- the serving-profile shell layer "
            'this app resolves a serving profile through.'
        )
    return Path(configured)


async def profile_names() -> list[str]:
    """The defined profile names, via `main.sh profile-names` (one per line).

    An unset LLAMA_ENV_SH resolves to an empty list, the same "unrecorded"
    posture profile() below takes -- a misconfigured/not-yet-wired-up
    deployment should show an empty picker, not a 500 with no readable
    message (Starlette's default error body for an unhandled exception is
    plain text, which breaks every API client's `res.json()` parse).
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            str(_env_sh()), 'profile-names', stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return []
        if proc.returncode == 0:
            return [line.strip() for line in stdout.decode().splitlines() if line.strip()]
    except (OSError, RuntimeError):
        pass
    return []


async def profile(name: str | None) -> dict:
    """Resolve a serving profile by asking llama-env.sh, not by copying it.

    A failure here (the script is missing, times out, or answers with
    something that is not JSON) resolves to {} -- the same "unrecorded"
    posture the original CLI took, rather than an error: a hand-started
    server with no profile is a normal, supported case.
    """
    argv = [str(_env_sh()), 'profile-json'] + ([name] if name else [])
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {}
        if proc.returncode == 0:
            return json.loads(stdout)
    except (OSError, ValueError):
        pass
    return {}


def port(prof: dict) -> int:
    return int(os.environ.get('LLAMA_PORT') or prof.get('port') or 8090)


async def served_model(session: aiohttp.ClientSession, p: int, prof: dict) -> str:
    """What the server is actually serving, falling back to the profile alias.

    The model name is read from GET /v1/models, not taken from the profile:
    the profile says what *would* be served, and the server is already
    serving something -- trusting the profile alone previously labelled a run
    with the wrong one.
    """
    alias = prof.get('alias', '')
    try:
        async with session.get(
            f'http://127.0.0.1:{p}/v1/models', timeout=aiohttp.ClientTimeout(total=5)
        ) as resp:
            data = await resp.json(content_type=None)
        served = (data.get('data') or [{}])[0].get('id', '')
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, IndexError, OSError):
        return alias
    if served and alias and served != alias:
        log.warning(
            "port %s is serving '%s', not profile '%s's '%s' - testing what is running",
            p,
            served,
            prof.get('name', '?'),
            alias,
        )
    return served or alias
