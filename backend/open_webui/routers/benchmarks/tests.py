"""routers/benchmarks/tests.py - run a benchmark suite against the serving model, live.

`prepare_suite()`/`run_items()` in benchmarks.runner already do the real
work (interleaving, resume, grading, recording) and were already designed
to be driven by a web UI's on_record/should_stop callbacks -- this router
is the asyncio.Queue/SSE plumbing around them, replacing the original
dashboard's in-process background-thread version with a background task.
"""

from __future__ import annotations

import asyncio
import json

import aiohttp
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from open_webui.benchmarks import runner
from open_webui.benchmarks.adapters import ADAPTERS_DIR
from open_webui.benchmarks.suites import SUITES_DIR
from open_webui.constants import ERROR_MESSAGES
from open_webui.utils.auth import get_admin_user

router = APIRouter()

_task: asyncio.Task | None = None
_queue: asyncio.Queue | None = None
_cancel = False


class TestsRunForm(BaseModel):
    suite: str
    profile: str | None = None
    benchmark: str | None = None
    system: str | None = None
    slice: str | None = None
    resume: bool = False

    model_config = ConfigDict(extra='forbid')


@router.get('/options')
async def get_options(user=Depends(get_admin_user)):
    return {
        'tiers': sorted(p.stem for p in SUITES_DIR.glob('*.toml')),
        'benchmarks': sorted(p.stem for p in ADAPTERS_DIR.glob('*.toml')),
        'systems': runner.system_names(),
    }


async def _run_suite(form: TestsRunForm, queue: asyncio.Queue) -> None:
    global _cancel
    passed = attempted = 0
    try:
        async with aiohttp.ClientSession() as session:
            ctx, todo, skipped = await runner.prepare_suite(
                session,
                suite_name=form.suite,
                profile_name=form.profile,
                benchmark=form.benchmark,
                item_slice=form.slice,
                system_name=form.system,
                resume=form.resume,
            )
            total = len(todo)
            await queue.put({'type': 'start', 'suite_run_id': ctx['suite_run_id'], 'total': total, 'skipped': len(skipped)})

            async def on_record(record: dict) -> None:
                nonlocal passed, attempted
                if record['outcome'] != 'skipped':
                    attempted += 1
                    if record['outcome'] == 'pass':
                        passed += 1
                await queue.put(
                    {
                        'type': 'item',
                        'benchmark': record['benchmark'],
                        'item_id': record['item_id'],
                        'outcome': record['outcome'],
                        'reason': record.get('reason', ''),
                        'i': attempted,
                        'total': total,
                        'passed': passed,
                        'attempted': attempted,
                    }
                )

            def should_stop() -> bool:
                return _cancel

            await runner.run_items(session, todo, ctx, on_record=on_record, should_stop=should_stop)
        await queue.put({'type': 'done', 'passed': passed, 'attempted': attempted, 'cancelled': _cancel})
    except runner.SuiteLoadError as exc:
        await queue.put({'type': 'error', 'message': str(exc)})
    except runner.ServerGone as exc:
        await queue.put({'type': 'error', 'message': f'server gone: {exc}'})
    except Exception as exc:  # noqa: BLE001 - reported to the client, not swallowed silently
        await queue.put({'type': 'error', 'message': str(exc)})


@router.post('/run')
async def start_run(form_data: TestsRunForm, user=Depends(get_admin_user)):
    global _task, _queue, _cancel
    if _task is not None and not _task.done():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=ERROR_MESSAGES.DEFAULT('a suite run is already active')
        )
    _queue = asyncio.Queue()
    _cancel = False
    _task = asyncio.create_task(_run_suite(form_data, _queue))
    return {'started': True}


@router.post('/cancel')
async def cancel_run(user=Depends(get_admin_user)):
    global _cancel
    if _task is None or _task.done():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=ERROR_MESSAGES.NOT_FOUND)
    _cancel = True
    return {'cancelling': True}


@router.get('/stream')
async def stream_run(user=Depends(get_admin_user)):
    queue = _queue
    if queue is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=ERROR_MESSAGES.NOT_FOUND)

    async def event_stream():
        while True:
            message = await queue.get()
            yield f'data: {json.dumps(message)}\n\n'
            if message['type'] in ('done', 'error'):
                break

    return StreamingResponse(event_stream(), media_type='text/event-stream')
