import logging
import time
from typing import Optional

from open_webui.internal.db import Base, get_async_db_context
from pydantic import BaseModel, ConfigDict
from sqlalchemy import BigInteger, Column, Float, ForeignKey, Integer, Text, func, select
from sqlalchemy.ext.asyncio import AsyncSession

log = logging.getLogger(__name__)


####################
# Benchmark GPU Telemetry DB Schema
#
# Ported from scripts/llama_db.py's `gpu_sample` and `metrics_scrape`
# tables. Written by the telemetry recorder subprocess (see
# benchmarks/telemetry_recorder.py) polling nvidia-smi and the served
# model's own /metrics endpoint for the life of a run -- raw samples are
# kept, not just a running summary, so a statistic computed wrongly can
# always be recomputed from the same rows.
####################


class BenchmarkGpuSample(Base):
    __tablename__ = 'benchmark_gpu_sample'
    sample_id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey('benchmark_run.run_id', ondelete='CASCADE'), nullable=False)
    at = Column(BigInteger, nullable=False)
    temp_c = Column(Integer, nullable=True)
    util_pct = Column(Integer, nullable=True)
    mem_used_mib = Column(Integer, nullable=True)
    mem_total_mib = Column(Integer, nullable=True)
    power_w = Column(Float, nullable=True)
    sm_mhz = Column(Integer, nullable=True)
    throttle = Column(Integer, nullable=True)


class BenchmarkGpuSampleModel(BaseModel):
    sample_id: int
    run_id: int
    at: int
    temp_c: Optional[int] = None
    util_pct: Optional[int] = None
    mem_used_mib: Optional[int] = None
    mem_total_mib: Optional[int] = None
    power_w: Optional[float] = None
    sm_mhz: Optional[int] = None
    throttle: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)


class BenchmarkMetricsScrape(Base):
    __tablename__ = 'benchmark_metrics_scrape'
    run_id = Column(Integer, ForeignKey('benchmark_run.run_id', ondelete='CASCADE'), primary_key=True)
    at = Column(BigInteger, primary_key=True)
    counter = Column(Text, primary_key=True)
    value = Column(Float, nullable=False)


class BenchmarkMetricsScrapeModel(BaseModel):
    run_id: int
    at: int
    counter: str
    value: float

    model_config = ConfigDict(from_attributes=True)


class BenchmarkGpuSampleTable:
    async def add_sample(self, run_id: int, sample: dict, db: Optional[AsyncSession] = None) -> None:
        async with get_async_db_context(db) as db:
            db.add(
                BenchmarkGpuSample(
                    run_id=run_id,
                    at=sample.get('at') or int(time.time()),
                    temp_c=sample.get('temp_c'),
                    util_pct=sample.get('util_pct'),
                    mem_used_mib=sample.get('mem_used_mib'),
                    mem_total_mib=sample.get('mem_total_mib'),
                    power_w=sample.get('power_w'),
                    sm_mhz=sample.get('sm_mhz'),
                    throttle=sample.get('throttle'),
                )
            )
            await db.commit()

    async def list_samples(self, run_id: int, db: Optional[AsyncSession] = None) -> list[BenchmarkGpuSampleModel]:
        async with get_async_db_context(db) as db:
            result = await db.execute(
                select(BenchmarkGpuSample).filter_by(run_id=run_id).order_by(BenchmarkGpuSample.at)
            )
            return [BenchmarkGpuSampleModel.model_validate(row) for row in result.scalars().all()]

    async def get_run_aggregates(self, run_id: int, db: Optional[AsyncSession] = None) -> Optional[dict]:
        """Per-run GPU stat summary. Replaces sqlite's v_run_gpu view."""
        async with get_async_db_context(db) as db:
            result = await db.execute(
                select(
                    func.count().label('samples'),
                    func.avg(BenchmarkGpuSample.temp_c).label('temp_avg'),
                    func.max(BenchmarkGpuSample.temp_c).label('temp_max'),
                    func.avg(BenchmarkGpuSample.util_pct).label('util_avg'),
                    func.max(BenchmarkGpuSample.util_pct).label('util_max'),
                    func.avg(BenchmarkGpuSample.mem_used_mib).label('mem_avg'),
                    func.max(BenchmarkGpuSample.mem_used_mib).label('mem_max'),
                    func.avg(BenchmarkGpuSample.power_w).label('power_avg'),
                    func.max(BenchmarkGpuSample.power_w).label('power_max'),
                    func.avg(BenchmarkGpuSample.sm_mhz).label('sm_avg'),
                    func.min(BenchmarkGpuSample.mem_total_mib - BenchmarkGpuSample.mem_used_mib).label(
                        'vram_headroom_mib'
                    ),
                ).filter(BenchmarkGpuSample.run_id == run_id)
            )
            row = result.first()
            if not row or row.samples == 0:
                return None
            return dict(row._mapping)


class BenchmarkMetricsScrapeTable:
    async def add_metrics(self, run_id: int, at: int, values: dict, db: Optional[AsyncSession] = None) -> None:
        """One /metrics scrape, stored whole rather than as a start/end pair.

        llama.cpp updates its prompt counters when a prompt is processed but
        its generation counters when the task completes, so keeping the full
        series (not just first/last) means a delta can be taken to the last
        scrape after the final completion instead of a scrape that caught
        only half the picture.
        """
        async with get_async_db_context(db) as db:
            for name, value in values.items():
                existing = await db.execute(
                    select(BenchmarkMetricsScrape).filter_by(run_id=run_id, at=at, counter=name)
                )
                row = existing.scalars().first()
                if row:
                    row.value = float(value)
                else:
                    db.add(BenchmarkMetricsScrape(run_id=run_id, at=at, counter=name, value=float(value)))
            await db.commit()

    async def get_delta(self, run_id: int, db: Optional[AsyncSession] = None) -> dict[str, float]:
        """Counters accumulated over the run: the last scrape minus the first.

        A counter that went backwards means the server restarted under the
        recorder, and the whole delta is unknowable rather than negative, so
        it is dropped.
        """
        async with get_async_db_context(db) as db:
            bounds = await db.execute(
                select(
                    BenchmarkMetricsScrape.counter,
                    func.min(BenchmarkMetricsScrape.at).label('first_at'),
                    func.max(BenchmarkMetricsScrape.at).label('last_at'),
                )
                .filter(BenchmarkMetricsScrape.run_id == run_id)
                .group_by(BenchmarkMetricsScrape.counter)
            )
            out: dict[str, float] = {}
            for row in bounds.all():
                first = await db.execute(
                    select(BenchmarkMetricsScrape.value).filter_by(
                        run_id=run_id, counter=row.counter, at=row.first_at
                    )
                )
                last = await db.execute(
                    select(BenchmarkMetricsScrape.value).filter_by(
                        run_id=run_id, counter=row.counter, at=row.last_at
                    )
                )
                first_v = first.scalars().first()
                last_v = last.scalars().first()
                if first_v is None or last_v is None:
                    continue
                delta = last_v - first_v
                if delta >= 0:
                    out[row.counter] = delta
            return out


BenchmarkGpuSamples = BenchmarkGpuSampleTable()
BenchmarkMetricsScrapes = BenchmarkMetricsScrapeTable()
