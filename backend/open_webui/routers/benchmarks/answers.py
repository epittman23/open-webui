"""routers/benchmarks/answers.py - browse stored answers for a suite run.

`_answer_document()` here is a small markdown renderer that was not part of
the `benchmarks.runner` port (that module ported the reusable run/grade/
record loop only, not the CLI's own answer-formatting command) -- it is
reimplemented directly against `BenchmarkAnswers.answer_for()`'s dict shape
rather than pulled from the CLI, since it is presentation-only and small.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from open_webui.constants import ERROR_MESSAGES
from open_webui.models.benchmark_tests import BenchmarkAnswers, BenchmarkResults
from open_webui.utils.auth import get_admin_user

router = APIRouter()

_FILTER_OUTCOMES = {
    'failures': ['fail_assert', 'fail_error', 'fail_timeout', 'no_code'],
    'pass': ['pass'],
    'all': None,
}


@router.get('/runs')
async def get_runs(limit: int = 20, user=Depends(get_admin_user)):
    return {'runs': await BenchmarkResults.suite_runs(limit=limit)}


@router.get('/')
async def get_answers(
    run: str,
    filter: str = Query('failures', pattern='^(failures|all|pass)$'),
    user=Depends(get_admin_user),
):
    rows = await BenchmarkResults.list_results(suite_run_id=run, outcome_in=_FILTER_OUTCOMES[filter])
    return {
        'rows': [
            {
                'benchmark': r.benchmark,
                'item_id': r.item_id,
                'outcome': r.outcome,
                'reason': r.reason,
                'reasoning_chars': r.reasoning_chars,
            }
            for r in rows
        ]
    }


def _answer_document(record: dict, *, thinking: bool) -> str:
    lines = [
        f"# {record['benchmark']}/{record['item_id']}",
        '',
        f"model: {record['model']}  ",
        f"config: {record.get('config_id') or 'unrecorded'}  ",
        f"suite run: {record['suite_run_id']}  ",
        f"system: {record.get('system_name') or '(none)'}  ",
        f"outcome: **{record['outcome']}**" + (f" - {record['reason']}" if record.get('reason') else ''),
        '',
    ]
    timings = record.get('timings') or {}
    if timings:
        lines += ['## timings', '', '```json', str(timings), '```', '']
    lines += ['## prompt', '', record.get('prompt', ''), '']
    reasoning = record.get('reasoning') or ''
    if reasoning:
        if thinking:
            lines += ['## thinking', '', reasoning, '']
        else:
            lines += ['<details><summary>thinking (hidden)</summary>', '', f'{len(reasoning)} characters', '', '</details>', '']
    lines += ['## response', '', record.get('content', '')]
    return '\n'.join(lines)


@router.get('/one')
async def get_answer(
    run: str,
    benchmark: str,
    item_id: str,
    thinking: bool = False,
    user=Depends(get_admin_user),
):
    record = await BenchmarkAnswers.answer_for(benchmark, item_id, suite_run_id=run)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=ERROR_MESSAGES.NOT_FOUND)
    reasoning = record.get('reasoning') or ''
    return {
        'benchmark': record['benchmark'],
        'item_id': record['item_id'],
        'model': record['model'],
        'config_id': record.get('config_id'),
        'suite_run_id': record['suite_run_id'],
        'system_name': record.get('system_name'),
        'outcome': record['outcome'],
        'reason': record.get('reason'),
        'timings': record.get('timings'),
        'prompt': record.get('prompt', ''),
        'reasoning': reasoning if thinking else None,
        'reasoning_chars': len(reasoning),
        'content': record.get('content', ''),
    }
