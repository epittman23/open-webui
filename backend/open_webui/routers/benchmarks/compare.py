"""routers/benchmarks/compare.py - config/benchmark/failure/serving comparison tables."""

from __future__ import annotations

import re

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict

from open_webui.benchmarks import compare as compare_logic
from open_webui.env import DATA_DIR
from open_webui.models.benchmark_tests import BenchmarkAnswers, BenchmarkResults
from open_webui.routers.benchmarks.answers import _answer_document
from open_webui.utils.auth import get_admin_user

router = APIRouter()

ANSWERS_DIR = DATA_DIR / 'benchmarks' / 'answers'
_SAFE = re.compile(r'[^A-Za-z0-9_.-]+')


@router.get('/')
async def get_compare(
    by: str = Query('config', pattern='^(config|benchmark|failures|serving)$'),
    tier: str | None = None,
    baseline: str | None = None,
    user=Depends(get_admin_user),
):
    if by == 'serving':
        facts = await compare_logic.serving_summary()
        derived_columns, derived, notes = compare_logic.serving_rows(facts)
        return {'derived_columns': derived_columns, 'derived': derived, 'notes': notes}

    records = await compare_logic.fetch_results(tier=tier)
    blocks = await compare_logic.fetch_config_blocks()
    groups = compare_logic.group_records(records, by_benchmark=(by == 'benchmark'))
    notes = compare_logic.caveats(groups, blocks) + await compare_logic.exclusion_note()

    if by == 'benchmark':
        columns, rows = compare_logic.benchmark_rows(groups)
        return {'columns': columns, 'rows': rows, 'notes': notes}
    if by == 'failures':
        return {'rows': compare_logic.failure_rows(groups), 'notes': notes}

    rows = compare_logic.rows_for(groups, blocks, baseline=baseline)
    return {'rows': rows, 'notes': notes}


class ExportAnswersForm(BaseModel):
    run: str

    model_config = ConfigDict(extra='forbid')


@router.post('/export-answers')
async def export_answers(form_data: ExportAnswersForm, user=Depends(get_admin_user)):
    """Recreate the pre-database `logs/answers/<run>/*.md` layout on demand.

    Every graded item's answer already lives in the DB (`benchmark_answer`);
    this writes it back out as flat markdown files for anyone who wants to
    grep/diff/archive a suite run outside the app, same as the original
    CLI's on-disk layout.
    """
    results = await BenchmarkResults.list_results(suite_run_id=form_data.run)
    out_dir = ANSWERS_DIR / _SAFE.sub('_', form_data.run)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for r in results:
        record = await BenchmarkAnswers.answer_for(r.benchmark, r.item_id, suite_run_id=form_data.run)
        if record is None:
            continue
        name = f"{_SAFE.sub('_', r.benchmark)}__{_SAFE.sub('_', r.item_id)}.md"
        (out_dir / name).write_text(_answer_document(record, thinking=True))
        written += 1
    return {'exported': written, 'directory': str(out_dir)}
