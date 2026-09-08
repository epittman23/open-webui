"""routers/benchmarks/report.py - generate and serve the statistical audit report.

benchmarks.report.build_report() returns markdown + in-memory figure bytes
by design (it never touches a filesystem, keeping its read-only-database
discipline) -- this router is the one place that decides where those bytes
land (DATA_DIR/benchmarks/reports/<run>/) and how they're served back.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from open_webui.benchmarks import report as report_logic
from open_webui.constants import ERROR_MESSAGES
from open_webui.env import DATA_DIR
from open_webui.utils.auth import get_admin_user

router = APIRouter()

REPORTS_DIR = DATA_DIR / 'benchmarks' / 'reports'


class ReportForm(BaseModel):
    tier: str | None = None
    model: str | None = None
    benchmark: str | None = None
    figures: bool = True

    model_config = ConfigDict(extra='forbid')


@router.post('/')
async def generate_report(form_data: ReportForm, user=Depends(get_admin_user)):
    try:
        result = await report_logic.build_report(
            tier=form_data.tier, model=form_data.model, benchmark=form_data.benchmark,
            generate_figures=form_data.figures,
        )
    except report_logic.SciPyUnavailable as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except report_logic.NoBenchmarkResults as exc:
        return {'error': str(exc)}

    run_dir = f"{time.strftime('%Y-%m-%d', time.gmtime())}-{uuid.uuid4().hex[:6]}"
    out_dir: Path = REPORTS_DIR / run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'report.md').write_text(result.markdown)
    for fig in result.figures:
        (out_dir / fig.name).write_bytes(fig.data)

    base = f'/api/v1/benchmarks/report/files/{run_dir}'
    return {
        'markdown_url': f'{base}/report.md',
        'figures': [f'{base}/{fig.name}' for fig in result.figures],
        'date': run_dir,
    }


@router.get('/files/{run_dir}/{filename}')
async def get_report_file(run_dir: str, filename: str, user=Depends(get_admin_user)):
    path = (REPORTS_DIR / run_dir / filename).resolve()
    if REPORTS_DIR.resolve() not in path.parents or not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=ERROR_MESSAGES.NOT_FOUND)
    return FileResponse(path)
