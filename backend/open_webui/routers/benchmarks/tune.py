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
import logging

import aiohttp
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from open_webui.benchmarks import tune
from open_webui.benchmarks.tune_schedule import GRID_DIR, TuneRefused
from open_webui.constants import ERROR_MESSAGES
from open_webui.utils.auth import get_admin_user

log = logging.getLogger(__name__)

router = APIRouter()

_task: asyncio.Task | None = None

# `_task` is fire-and-forget by design (a sweep runs for minutes to hours),
# but that means any exception raised before or during `start_sweep()`/
# `resume_sweep()` -- a bad grid file, a profile that fails to resolve, the
# port already held by a server started from the Serve page, llama-server
# refusing to load -- previously vanished into an unretrieved task
# exception: the POST to /start had already returned 200, so the frontend
# had no way to know the sweep never actually began. `_last_error` is set
# from inside `_run_sweep` and surfaced through /status and /log so a
# failed start is visible instead of silently doing nothing.
_last_error: str | None = None


async def _run_sweep(start):
    global _last_error
    try:
        async with aiohttp.ClientSession() as session:
            return await start(session)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see module docstring above
        log.exception('benchmarks tune sweep failed to start')
        _last_error = str(exc) or exc.__class__.__name__
        raise


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
    global _task, _last_error
    if _running():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=ERROR_MESSAGES.DEFAULT('a sweep is already running')
        )
    opts = form_data.to_options()
    _last_error = None
    _task = asyncio.create_task(_run_sweep(lambda session: tune.start_sweep(opts, session)))
    return {'started': True}


@router.post('/resume')
async def resume_tune(sweep_id: str | None = None, user=Depends(get_admin_user)):
    global _task, _last_error
    if _running():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=ERROR_MESSAGES.DEFAULT('a sweep is already running')
        )
    defaults = tune.SweepOptions()
    _last_error = None
    _task = asyncio.create_task(
        _run_sweep(lambda session: tune.resume_sweep(sweep_id, defaults, session))
    )
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
        # A start/resume attempt that failed before ever creating a sweep
        # row (e.g. no grid resolves, the port is already held) has nothing
        # for sweep_status() to find -- surface the failure instead of a
        # bare 404 the frontend has no sweep-shaped state to show alongside.
        if _last_error is not None:
            return {'running': _running(), 'last_error': _last_error}
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    result['running'] = _running()
    result['last_error'] = _last_error
    result['current_visit'] = tune.current_visit_progress(result['sweep_id']) if result.get('sweep_id') else None
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
                report['running'] = _running()
                report['last_error'] = _last_error
                report['current_visit'] = (
                    tune.current_visit_progress(report['sweep_id']) if report.get('sweep_id') else None
                )
                yield f'data: {json.dumps(report, default=str)}\n\n'
            except TuneRefused as exc:
                if _last_error is not None:
                    yield f'data: {json.dumps({"running": _running(), "last_error": _last_error})}\n\n'
                else:
                    yield f'data: {json.dumps({"error": str(exc)})}\n\n'
            if not _running():
                yield 'event: done\ndata: {}\n\n'
                return
            await asyncio.sleep(2)

    return StreamingResponse(event_stream(), media_type='text/event-stream')
