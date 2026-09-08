"""routers/benchmarks/live.py - GPU/throughput telemetry for the currently-recording run."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from open_webui.benchmarks import stats
from open_webui.models.benchmark_configs import BenchmarkRuns
from open_webui.models.benchmark_telemetry import BenchmarkGpuSamples, BenchmarkMetricsScrapes
from open_webui.models.benchmark_tests import BenchmarkRequests
from open_webui.utils.auth import get_admin_user

router = APIRouter()


@router.get('/')
async def get_live(user=Depends(get_admin_user)):
    run = await BenchmarkRuns.get_any_active_run()
    if run is None:
        return {'run': None}

    samples = [s.model_dump() for s in await BenchmarkGpuSamples.list_samples(run.run_id)]
    summary = stats.gpu_stats(samples)
    deltas = await BenchmarkMetricsScrapes.get_delta(run.run_id)
    aggregate = await BenchmarkRequests.get_run_request_aggregate(run.run_id)
    requests = aggregate['requests'] if aggregate else 0
    warning = stats.headroom_warning(f'{run.model} ({run.config_id})', summary.get('vram_headroom_mib'))

    return {
        'run': run.model_dump(),
        'summary': summary,
        'deltas': deltas,
        'requests': requests,
        'recent_samples': samples[-12:],
        'warning': warning,
    }
