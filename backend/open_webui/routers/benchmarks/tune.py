"""routers/benchmarks/tune.py - start/resume/stop/observe a configuration-search sweep.

A sweep runs for minutes to hours, so `/start` and `/resume` launch it as a
background asyncio.Task rather than awaiting it in the request handler.
`/stop` cancels that task directly (`task.cancel()`) rather than reaching
into the `Sweep` object for its own cooperative `request_cancel()` --
`benchmarks.tune`'s own module docstring documents a forced
`asyncio.CancelledError` as a supported, first-class way to stop a sweep
(it still reaches `finish()` and leaves a resumable, closed sweep row), so
this is the simplest correct way to expose "stop" over HTTP without needing
a second handle into the running sweep.

`/status` and `/log` are pure DB reads (`sweep_status()`), so they keep
working even across a backend restart while a sweep is mid-flight, same as
the original CLI dashboard's `/tune/status`.
"""

from __future__ import annotations

import asyncio
import json

import aiohttp
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from open_webui.benchmarks import tune
from open_webui.benchmarks.tune_schedule import GRID_DIR, TuneRefused
from open_webui.constants import ERROR_MESSAGES
from open_webui.utils.auth import get_admin_user

router = APIRouter()

_task: asyncio.Task | None = None


class TuneStartForm(BaseModel):
    profile: str | None = None
    tier: str = 'smoke'
    benchmark: str | None = None
    system: str | None = None
    grid: str | None = None
    budget: str = 'interactive'
    candidates: int | None = None
    round_items: int | None = None
    eta: int = 2
    seed: int = 0
    stages: list[str] = ['explore', 'refine']

    model_config = ConfigDict(extra='forbid')

    def to_options(self, **overrides) -> 'tune.SweepOptions':
        return tune.SweepOptions(**{**self.model_dump(), **overrides})


def _running() -> bool:
    return _task is not None and not _task.done()


@router.post('/start')
async def start_tune(form_data: TuneStartForm, user=Depends(get_admin_user)):
    global _task
    if _running():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=ERROR_MESSAGES.DEFAULT('a sweep is already running')
        )
    opts = form_data.to_options()

    async def _run():
        async with aiohttp.ClientSession() as session:
            return await tune.start_sweep(opts, session)

    _task = asyncio.create_task(_run())
    return {'started': True}


@router.post('/resume')
async def resume_tune(sweep_id: str | None = None, user=Depends(get_admin_user)):
    global _task
    if _running():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=ERROR_MESSAGES.DEFAULT('a sweep is already running')
        )
    defaults = tune.SweepOptions()

    async def _run():
        async with aiohttp.ClientSession() as session:
            return await tune.resume_sweep(sweep_id, defaults, session)

    _task = asyncio.create_task(_run())
    return {'resumed': True}


@router.post('/stop')
async def stop_tune(user=Depends(get_admin_user)):
    if not _running():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=ERROR_MESSAGES.NOT_FOUND)
    _task.cancel()
    return {'stopping': True}


@router.get('/status')
async def get_status(sweep_id: str | None = Query(None), user=Depends(get_admin_user)):
    try:
        result = await tune.sweep_status(sweep_id)
    except TuneRefused as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    result['running'] = _running()
    return result


@router.get('/grids')
async def get_grids(user=Depends(get_admin_user)):
    return {'grids': sorted(p.stem for p in GRID_DIR.glob('*.toml'))}


@router.get('/recent')
async def get_recent(limit: int = 20, user=Depends(get_admin_user)):
    return {'sweeps': await tune.list_recent_sweeps(limit=limit)}


@router.get('/log')
async def stream_status(sweep_id: str | None = Query(None), user=Depends(get_admin_user)):
    """Poll `sweep_status()` every 2s until the sweep ends or the task finishes.

    Not a literal stdout stream (there is no subprocess to tail -- the
    sweep runs in-process as an asyncio.Task) but the same live-updating
    contract the original dashboard's SSE log gave the Tune page.
    """

    async def event_stream():
        while True:
            try:
                report = await tune.sweep_status(sweep_id)
            except TuneRefused as exc:
                yield f'data: {json.dumps({"error": str(exc)})}\n\n'
                return
            yield f'data: {json.dumps(report, default=str)}\n\n'
            if not _running():
                yield 'event: done\ndata: {}\n\n'
                return
            await asyncio.sleep(2)

    return StreamingResponse(event_stream(), media_type='text/event-stream')
