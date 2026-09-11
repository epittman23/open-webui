"""routers/benchmarks/serve.py - start/stop/observe llama-server for benchmarking.

Thin HTTP wrapper over benchmarks.env_profile and benchmarks.proc.Command.
Overrides are passed as environment variables (LLAMA_NGL, LLAMA_CTX, ...),
matching main.sh's own override mechanism (`LLAMA_NGL=22 lllm-serve qwen38`)
exactly -- they are NOT `lllm-serve` CLI flags, which main.sh passes straight
through to `llama-server` unparsed.

Single-flight by design: only one server (and one recorded run) at a time,
same as the original CLI dashboard. The in-memory `_job` handle assumes this
process runs with a single worker, same requirement the fork's other
Serve-like features already have.
"""

from __future__ import annotations

import asyncio
import collections
import json
import os

import aiohttp
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from open_webui.benchmarks import env_profile
from open_webui.benchmarks.proc import Command
from open_webui.constants import ERROR_MESSAGES
from open_webui.models.benchmark_configs import BenchmarkRuns
from open_webui.utils.auth import get_admin_user

router = APIRouter()

_job: Command | None = None

# Log draining is a background task independent of any HTTP connection, not
# something `/stream` does itself -- a `Command.lines()` iterator only ever
# gets read once, so if the previous reader was the last SSE client and it
# disconnected (a browser tab navigating away aborts the fetch), nobody was
# left draining the subprocess's stdout pipe and every line since is gone.
# `_log_buffer` keeps recent history so a client reconnecting (the same tab
# coming back, or a second one) sees it immediately instead of a blank pane,
# and `_subscribers` fans new lines out to every currently-connected client.
_LOG_BUFFER_LINES = 2000
_log_buffer: collections.deque[str] = collections.deque(maxlen=_LOG_BUFFER_LINES)
_subscribers: set[asyncio.Queue] = set()
_drain_task: asyncio.Task | None = None

_DONE = object()  # sentinel pushed to subscriber queues when the job ends


async def _drain_log(job: Command) -> None:
    try:
        async for line in job.lines():
            _log_buffer.append(line)
            for queue in _subscribers:
                queue.put_nowait(line)
    finally:
        for queue in _subscribers:
            queue.put_nowait(_DONE)

_OVERRIDE_ENV = {
    'ngl': 'LLAMA_NGL',
    'ctx': 'LLAMA_CTX',
    'threads': 'LLAMA_THREADS',
    'parallel': 'LLAMA_PARALLEL',
    'ot': 'LLAMA_OT',
    'reasoning': 'LLAMA_REASONING',
    'spec': 'LLAMA_SPEC',
}


class ServeStartForm(BaseModel):
    profile: str | None = None
    ngl: int | None = None
    ctx: int | None = None
    threads: int | None = None
    parallel: int | None = None
    ot: str | None = None
    reasoning: str | None = None
    spec: str | None = None

    model_config = ConfigDict(extra='forbid')


@router.get('/profiles')
async def get_profiles(user=Depends(get_admin_user)):
    return {'profiles': await env_profile.profile_names()}


@router.get('/profile/{name}')
async def get_profile(name: str, user=Depends(get_admin_user)):
    return await env_profile.profile(name)


@router.post('/start')
async def start_serve(form_data: ServeStartForm, user=Depends(get_admin_user)):
    global _job, _drain_task
    if _job is not None and _job.running:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=ERROR_MESSAGES.DEFAULT('a server is already running')
        )
    if await BenchmarkRuns.get_any_active_run() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=ERROR_MESSAGES.DEFAULT('a recorded run is already active on another process'),
        )

    env = dict(os.environ)
    for field, var in _OVERRIDE_ENV.items():
        value = getattr(form_data, field)
        if value is not None:
            env[var] = str(value)

    command = f'lllm-serve {form_data.profile}' if form_data.profile else 'lllm-serve'
    _job = Command(command, env=env)
    await _job.start()
    _log_buffer.clear()
    _drain_task = asyncio.create_task(_drain_log(_job))
    return {'started': True}


@router.post('/stop')
async def stop_serve(user=Depends(get_admin_user)):
    global _job
    if _job is None or not _job.running:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=ERROR_MESSAGES.NOT_FOUND)
    await _job.stop()
    return {'stopped': True}


@router.get('/check')
async def check_serve(user=Depends(get_admin_user)):
    prof = await env_profile.profile(None)
    p = env_profile.port(prof)
    async with aiohttp.ClientSession() as session:
        model = await env_profile.served_model(session, p, prof)
    return {
        'port': p,
        'model': model,
        'profile': prof.get('name'),
        'running': _job is not None and _job.running,
    }


@router.get('/stream')
async def stream_serve(user=Depends(get_admin_user)):
    """Replay buffered history, then tail new lines as they arrive.

    Independent of `_drain_log`'s own read of the subprocess -- this only
    ever reads from `_log_buffer`/a subscriber queue, so a client
    disconnecting (tab navigation) and reconnecting later sees the lines it
    missed (up to `_LOG_BUFFER_LINES`) instead of a blank pane, and does not
    stop the subprocess's stdout from being drained in the meantime.
    """
    if _job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=ERROR_MESSAGES.NOT_FOUND)

    queue: asyncio.Queue = asyncio.Queue()
    _subscribers.add(queue)
    backlog = list(_log_buffer)

    async def event_stream():
        try:
            for line in backlog:
                yield f'data: {json.dumps({"line": line})}\n\n'
            if _job is not None and not _job.running and queue.empty():
                # The job already ended before this client connected, and
                # everything it ever printed is in `backlog` above.
                yield 'event: done\ndata: {}\n\n'
                return
            while True:
                item = await queue.get()
                if item is _DONE:
                    yield 'event: done\ndata: {}\n\n'
                    return
                yield f'data: {json.dumps({"line": item})}\n\n'
        finally:
            _subscribers.discard(queue)

    return StreamingResponse(event_stream(), media_type='text/event-stream')
