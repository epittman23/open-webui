import logging
import time
import uuid
from typing import Optional

from open_webui.internal.db import Base, get_async_db_context
from pydantic import BaseModel, ConfigDict
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    Float,
    ForeignKey,
    Integer,
    PrimaryKeyConstraint,
    Text,
    UniqueConstraint,
    select,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

log = logging.getLogger(__name__)


####################
# Benchmark Tune (config-search) DB Schema
#
# Ported from scripts/llama_db.py's tune_sweep/tune_candidate/tune_round/
# tune_visit/tune_pause tables -- the scheduling record for the config
# search engine (backend/open_webui/benchmarks/tune.py). Deliberately not
# another store of measurements: throughput lives in benchmark_request and
# verdicts live in benchmark_result, so this is the schedule that produced
# them plus the keys to join back.
#
# tune_candidate is identified by candidate_sha (a hash of the override
# dict the optimizer chose), not config_id, because a candidate that OOMs
# before the server loads never reaches BenchmarkConfigs.upsert at all --
# infeasibility is a result this search has to be able to record -- and
# config_id is filled in only on a candidate's first successful load.
####################

BUDGET_MODES = ('interactive', 'overnight', 'multiday', 'trials')
VERDICTS = ('adopted', 'rejected', 'indeterminate', 'incomplete')
CANDIDATE_STATUSES = ('pending', 'active', 'eliminated', 'infeasible', 'marginal', 'winner', 'rejected')
VISIT_STATUSES = ('running', 'done', 'infeasible', 'aborted')
PAUSE_TRIGGERS = ('drift', 'regime', 'cliff')
PAUSE_RESOLUTIONS = ('recovered', 'timeout', 'abandoned')


class BenchmarkTuneSweep(Base):
    __tablename__ = 'benchmark_tune_sweep'
    __table_args__ = (
        CheckConstraint(f"budget_mode IN {BUDGET_MODES!r}", name='ck_benchmark_tune_sweep_budget_mode'),
        CheckConstraint(f"verdict IS NULL OR verdict IN {VERDICTS!r}", name='ck_benchmark_tune_sweep_verdict'),
    )
    sweep_id = Column(Text, primary_key=True)
    started_at = Column(BigInteger, nullable=False)
    ended_at = Column(BigInteger, nullable=True)
    ended_reason = Column(Text, nullable=True)
    pid = Column(Integer, nullable=True)
    profile = Column(Text, nullable=False)
    tier = Column(Text, nullable=False)
    benchmark = Column(Text, nullable=True)
    system_name = Column(Text, nullable=True)
    system_sha = Column(Text, nullable=True)
    grid_path = Column(Text, nullable=False)
    grid_sha = Column(Text, nullable=False)
    item_order_sha = Column(Text, nullable=False)
    item_count = Column(Integer, nullable=False)
    budget_mode = Column(Text, nullable=False)
    budget_seconds = Column(Integer, nullable=True)
    budget_visits = Column(Integer, nullable=True)
    eta = Column(Integer, nullable=False)
    round_items = Column(Integer, nullable=False)
    candidates = Column(Integer, nullable=False)
    stages = Column(Text, nullable=False, default='explore,refine')
    objective = Column(Text, nullable=False)
    alpha = Column(Float, nullable=False)
    on_drift = Column(Text, nullable=False)
    seed = Column(Integer, nullable=False)
    baseline_config_id = Column(Text, ForeignKey('benchmark_config.config_id'), nullable=True)
    winner_candidate = Column(Text, nullable=True)
    verdict = Column(Text, nullable=True)
    verdict_reason = Column(Text, nullable=False, default='')


class BenchmarkTuneSweepModel(BaseModel):
    sweep_id: str
    started_at: int
    ended_at: Optional[int] = None
    ended_reason: Optional[str] = None
    pid: Optional[int] = None
    profile: str
    tier: str
    benchmark: Optional[str] = None
    system_name: Optional[str] = None
    system_sha: Optional[str] = None
    grid_path: str
    grid_sha: str
    item_order_sha: str
    item_count: int
    budget_mode: str
    budget_seconds: Optional[int] = None
    budget_visits: Optional[int] = None
    eta: int
    round_items: int
    candidates: int
    stages: str = 'explore,refine'
    objective: str
    alpha: float
    on_drift: str
    seed: int
    baseline_config_id: Optional[str] = None
    winner_candidate: Optional[str] = None
    verdict: Optional[str] = None
    verdict_reason: str = ''

    model_config = ConfigDict(from_attributes=True)


class BenchmarkTuneCandidate(Base):
    __tablename__ = 'benchmark_tune_candidate'
    __table_args__ = (
        CheckConstraint(f"status IN {CANDIDATE_STATUSES!r}", name='ck_benchmark_tune_candidate_status'),
    )
    sweep_id = Column(Text, ForeignKey('benchmark_tune_sweep.sweep_id', ondelete='CASCADE'), primary_key=True)
    candidate_sha = Column(Text, primary_key=True)
    stage = Column(Text, nullable=False)
    overrides = Column(JSON, nullable=False)
    is_baseline = Column(Boolean, nullable=False, default=False)
    config_id = Column(Text, ForeignKey('benchmark_config.config_id'), nullable=True)
    suite_run_id = Column(Text, nullable=False)
    status = Column(Text, nullable=False)
    status_reason = Column(Text, nullable=False, default='')
    score = Column(Float, nullable=True)
    eliminated_round = Column(Integer, nullable=True)
    created_at = Column(BigInteger, nullable=False)


class BenchmarkTuneCandidateModel(BaseModel):
    sweep_id: str
    candidate_sha: str
    stage: str
    overrides: dict
    is_baseline: bool = False
    config_id: Optional[str] = None
    suite_run_id: str
    status: str
    status_reason: str = ''
    score: Optional[float] = None
    eliminated_round: Optional[int] = None
    created_at: int

    model_config = ConfigDict(from_attributes=True)


class BenchmarkTuneRound(Base):
    __tablename__ = 'benchmark_tune_round'
    sweep_id = Column(Text, ForeignKey('benchmark_tune_sweep.sweep_id', ondelete='CASCADE'), primary_key=True)
    round = Column(Integer, primary_key=True)
    stage = Column(Text, nullable=False)
    started_at = Column(BigInteger, nullable=False)
    ended_at = Column(BigInteger, nullable=True)
    item_from = Column(Integer, nullable=False)
    item_to = Column(Integer, nullable=False)
    survivors = Column(Integer, nullable=False)
    baseline_gen_tps = Column(Float, nullable=True)
    baseline_regime = Column(Text, nullable=True)
    drift_ratio = Column(Float, nullable=True)
    decision = Column(Text, nullable=False, default='')
    notes = Column(Text, nullable=False, default='')


class BenchmarkTuneRoundModel(BaseModel):
    sweep_id: str
    round: int
    stage: str
    started_at: int
    ended_at: Optional[int] = None
    item_from: int
    item_to: int
    survivors: int
    baseline_gen_tps: Optional[float] = None
    baseline_regime: Optional[str] = None
    drift_ratio: Optional[float] = None
    decision: str = ''
    notes: str = ''

    model_config = ConfigDict(from_attributes=True)


class BenchmarkTuneVisit(Base):
    __tablename__ = 'benchmark_tune_visit'
    __table_args__ = (
        UniqueConstraint('sweep_id', 'candidate_sha', 'round', 'attempt', name='uq_benchmark_tune_visit_slot'),
        CheckConstraint(f"status IN {VISIT_STATUSES!r}", name='ck_benchmark_tune_visit_status'),
    )
    visit_id = Column(Integer, primary_key=True, autoincrement=True)
    sweep_id = Column(Text, ForeignKey('benchmark_tune_sweep.sweep_id', ondelete='CASCADE'), nullable=False)
    candidate_sha = Column(Text, nullable=False)
    round = Column(Integer, nullable=False)
    attempt = Column(Integer, nullable=False, default=1)
    run_id = Column(Integer, ForeignKey('benchmark_run.run_id', ondelete='SET NULL'), nullable=True)
    config_id = Column(Text, nullable=True)
    started_at = Column(BigInteger, nullable=False)
    ended_at = Column(BigInteger, nullable=True)
    server_pgid = Column(Integer, nullable=True)
    load_ms = Column(Float, nullable=True)
    item_from = Column(Integer, nullable=False)
    item_to = Column(Integer, nullable=False)
    items_done = Column(Integer, nullable=False, default=0)
    since_pause_seconds = Column(Float, nullable=True)
    counts_toward_round = Column(Boolean, nullable=False, default=True)
    status = Column(Text, nullable=False)
    reason = Column(Text, nullable=False, default='')


class BenchmarkTuneVisitModel(BaseModel):
    visit_id: int
    sweep_id: str
    candidate_sha: str
    round: int
    attempt: int = 1
    run_id: Optional[int] = None
    config_id: Optional[str] = None
    started_at: int
    ended_at: Optional[int] = None
    server_pgid: Optional[int] = None
    load_ms: Optional[float] = None
    item_from: int
    item_to: int
    items_done: int = 0
    since_pause_seconds: Optional[float] = None
    counts_toward_round: bool = True
    status: str
    reason: str = ''

    model_config = ConfigDict(from_attributes=True)


class BenchmarkTunePause(Base):
    __tablename__ = 'benchmark_tune_pause'
    __table_args__ = (
        CheckConstraint(f"trigger_kind IN {PAUSE_TRIGGERS!r}", name='ck_benchmark_tune_pause_trigger'),
        CheckConstraint(
            f"resolution IS NULL OR resolution IN {PAUSE_RESOLUTIONS!r}", name='ck_benchmark_tune_pause_resolution'
        ),
    )
    pause_id = Column(Integer, primary_key=True, autoincrement=True)
    sweep_id = Column(Text, ForeignKey('benchmark_tune_sweep.sweep_id', ondelete='CASCADE'), nullable=False)
    round = Column(Integer, nullable=True)
    visit_id = Column(Integer, ForeignKey('benchmark_tune_visit.visit_id', ondelete='SET NULL'), nullable=True)
    started_at = Column(BigInteger, nullable=False)
    ended_at = Column(BigInteger, nullable=True)
    trigger_kind = Column(Text, nullable=False)
    drift_ratio = Column(Float, nullable=True)
    attempt = Column(Integer, nullable=False, default=1)
    throttle_before = Column(Text, nullable=True)
    throttle_after = Column(Text, nullable=True)
    temp_before = Column(Float, nullable=True)
    temp_after = Column(Float, nullable=True)
    power_before = Column(Float, nullable=True)
    power_after = Column(Float, nullable=True)
    probe_tps = Column(Float, nullable=True)
    resolution = Column(Text, nullable=True)


class BenchmarkTunePauseModel(BaseModel):
    pause_id: int
    sweep_id: str
    round: Optional[int] = None
    visit_id: Optional[int] = None
    started_at: int
    ended_at: Optional[int] = None
    trigger_kind: str
    drift_ratio: Optional[float] = None
    attempt: int = 1
    throttle_before: Optional[str] = None
    throttle_after: Optional[str] = None
    temp_before: Optional[float] = None
    temp_after: Optional[float] = None
    power_before: Optional[float] = None
    power_after: Optional[float] = None
    probe_tps: Optional[float] = None
    resolution: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class BenchmarkTuneSweepTable:
    async def open_sweep(self, sweep_id: Optional[str] = None, db: Optional[AsyncSession] = None, **fields) -> str:
        sweep_id = sweep_id or f"{int(time.time())}-{uuid.uuid4().hex[:6]}"
        async with get_async_db_context(db) as db:
            row = BenchmarkTuneSweep(sweep_id=sweep_id, started_at=fields.pop('started_at', None) or int(time.time()), **fields)
            db.add(row)
            await db.commit()
            return sweep_id

    async def close_sweep(
        self,
        sweep_id: str,
        *,
        reason: str,
        verdict: Optional[str] = None,
        winner: Optional[str] = None,
        verdict_reason: str = '',
        db: Optional[AsyncSession] = None,
    ) -> None:
        """Record how a sweep's invocation ended.

        Guarded on verdict, not ended_at: 'incomplete' is resumable, and
        every resume calls this again at its own end. Once a verdict is
        terminal (adopted/rejected/indeterminate), further closes are
        refused.
        """
        async with get_async_db_context(db) as db:
            result = await db.execute(select(BenchmarkTuneSweep).filter_by(sweep_id=sweep_id))
            row = result.scalars().first()
            if row and (row.verdict is None or row.verdict == 'incomplete'):
                row.ended_at = int(time.time())
                row.ended_reason = reason
                row.verdict = verdict
                row.winner_candidate = winner
                row.verdict_reason = verdict_reason
                await db.commit()

    async def get_by_id(self, sweep_id: str, db: Optional[AsyncSession] = None) -> Optional[BenchmarkTuneSweepModel]:
        async with get_async_db_context(db) as db:
            result = await db.execute(select(BenchmarkTuneSweep).filter_by(sweep_id=sweep_id))
            row = result.scalars().first()
            return BenchmarkTuneSweepModel.model_validate(row) if row else None

    async def list_sweeps(self, limit: int = 20, db: Optional[AsyncSession] = None) -> list[BenchmarkTuneSweepModel]:
        async with get_async_db_context(db) as db:
            result = await db.execute(
                select(BenchmarkTuneSweep).order_by(BenchmarkTuneSweep.started_at.desc()).limit(limit)
            )
            return [BenchmarkTuneSweepModel.model_validate(row) for row in result.scalars().all()]

    async def latest_sweep(self, *, open_only: bool = False, db: Optional[AsyncSession] = None) -> Optional[str]:
        async with get_async_db_context(db) as db:
            query = select(BenchmarkTuneSweep.sweep_id)
            if open_only:
                query = query.filter(
                    (BenchmarkTuneSweep.verdict.is_(None)) | (BenchmarkTuneSweep.verdict == 'incomplete')
                )
            query = query.order_by(BenchmarkTuneSweep.started_at.desc()).limit(1)
            result = await db.execute(query)
            return result.scalars().first()


class BenchmarkTuneCandidateTable:
    async def add_candidates(self, sweep_id: str, rows: list[dict], db: Optional[AsyncSession] = None) -> int:
        added = 0
        async with get_async_db_context(db) as db:
            for r in rows:
                stmt = pg_insert(BenchmarkTuneCandidate).values(
                    sweep_id=sweep_id,
                    candidate_sha=r['candidate_sha'],
                    stage=r['stage'],
                    overrides=r['overrides'],
                    is_baseline=bool(r.get('is_baseline')),
                    config_id=r.get('config_id'),
                    suite_run_id=r['suite_run_id'],
                    status=r.get('status', 'pending'),
                    created_at=int(time.time()),
                )
                stmt = stmt.on_conflict_do_nothing(index_elements=['sweep_id', 'candidate_sha'])
                result = await db.execute(stmt)
                added += result.rowcount or 0
            await db.commit()
            return added

    async def set_candidate(self, sweep_id: str, candidate_sha: str, db: Optional[AsyncSession] = None, **fields) -> None:
        allowed = ('status', 'status_reason', 'config_id', 'score', 'eliminated_round', 'stage')
        sets = {k: v for k, v in fields.items() if k in allowed}
        if not sets:
            return
        async with get_async_db_context(db) as db:
            result = await db.execute(
                select(BenchmarkTuneCandidate).filter_by(sweep_id=sweep_id, candidate_sha=candidate_sha)
            )
            row = result.scalars().first()
            if row:
                for k, v in sets.items():
                    setattr(row, k, v)
                await db.commit()

    async def sweep_candidates(
        self, sweep_id: str, *, stage: Optional[str] = None, db: Optional[AsyncSession] = None
    ) -> list[BenchmarkTuneCandidateModel]:
        async with get_async_db_context(db) as db:
            query = select(BenchmarkTuneCandidate).filter_by(sweep_id=sweep_id)
            if stage:
                query = query.filter(BenchmarkTuneCandidate.stage == stage)
            query = query.order_by(
                BenchmarkTuneCandidate.is_baseline.desc(),
                BenchmarkTuneCandidate.created_at,
                BenchmarkTuneCandidate.candidate_sha,
            )
            result = await db.execute(query)
            return [BenchmarkTuneCandidateModel.model_validate(row) for row in result.scalars().all()]


class BenchmarkTuneRoundTable:
    async def open_round(
        self, sweep_id: str, rnd: int, *, stage: str, item_from: int, item_to: int, survivors: int,
        db: Optional[AsyncSession] = None,
    ) -> None:
        async with get_async_db_context(db) as db:
            stmt = pg_insert(BenchmarkTuneRound).values(
                sweep_id=sweep_id, round=rnd, stage=stage, started_at=int(time.time()),
                item_from=item_from, item_to=item_to, survivors=survivors,
            )
            stmt = stmt.on_conflict_do_nothing(index_elements=['sweep_id', 'round'])
            await db.execute(stmt)
            await db.commit()

    async def close_round(self, sweep_id: str, rnd: int, db: Optional[AsyncSession] = None, **fields) -> None:
        allowed = ('baseline_gen_tps', 'baseline_regime', 'drift_ratio', 'decision', 'notes', 'survivors')
        sets = {k: v for k, v in fields.items() if k in allowed}
        async with get_async_db_context(db) as db:
            result = await db.execute(select(BenchmarkTuneRound).filter_by(sweep_id=sweep_id, round=rnd))
            row = result.scalars().first()
            if row:
                for k, v in sets.items():
                    setattr(row, k, v)
                row.ended_at = int(time.time())
                await db.commit()

    async def sweep_rounds(self, sweep_id: str, db: Optional[AsyncSession] = None) -> list[BenchmarkTuneRoundModel]:
        async with get_async_db_context(db) as db:
            result = await db.execute(
                select(BenchmarkTuneRound).filter_by(sweep_id=sweep_id).order_by(BenchmarkTuneRound.round)
            )
            return [BenchmarkTuneRoundModel.model_validate(row) for row in result.scalars().all()]


class BenchmarkTuneVisitTable:
    async def open_visit(
        self, sweep_id: str, candidate_sha: str, rnd: int, *, attempt: int = 1, item_from: int, item_to: int,
        since_pause_seconds: Optional[float] = None, db: Optional[AsyncSession] = None,
    ) -> int:
        async with get_async_db_context(db) as db:
            row = BenchmarkTuneVisit(
                sweep_id=sweep_id, candidate_sha=candidate_sha, round=rnd, attempt=attempt,
                started_at=int(time.time()), item_from=item_from, item_to=item_to,
                since_pause_seconds=since_pause_seconds, status='running',
            )
            db.add(row)
            await db.commit()
            await db.refresh(row)
            return row.visit_id

    async def close_visit(self, visit_id: int, *, status: str, db: Optional[AsyncSession] = None, **fields) -> None:
        allowed = ('run_id', 'config_id', 'server_pgid', 'load_ms', 'items_done', 'counts_toward_round', 'reason')
        sets = {k: v for k, v in fields.items() if k in allowed}
        async with get_async_db_context(db) as db:
            result = await db.execute(select(BenchmarkTuneVisit).filter_by(visit_id=visit_id))
            row = result.scalars().first()
            if row:
                for k, v in sets.items():
                    setattr(row, k, v)
                row.status = status
                row.ended_at = int(time.time())
                await db.commit()

    async def set_visit(self, visit_id: int, db: Optional[AsyncSession] = None, **fields) -> None:
        """Update a visit that is still running (the pgid and run_id land early)."""
        allowed = ('run_id', 'config_id', 'server_pgid', 'load_ms', 'items_done', 'counts_toward_round', 'reason')
        sets = {k: v for k, v in fields.items() if k in allowed}
        if not sets:
            return
        async with get_async_db_context(db) as db:
            result = await db.execute(select(BenchmarkTuneVisit).filter_by(visit_id=visit_id))
            row = result.scalars().first()
            if row:
                for k, v in sets.items():
                    setattr(row, k, v)
                await db.commit()

    async def sweep_visits(
        self, sweep_id: str, *, rnd: Optional[int] = None, db: Optional[AsyncSession] = None
    ) -> list[BenchmarkTuneVisitModel]:
        async with get_async_db_context(db) as db:
            query = select(BenchmarkTuneVisit).filter_by(sweep_id=sweep_id)
            if rnd is not None:
                query = query.filter(BenchmarkTuneVisit.round == rnd)
            query = query.order_by(BenchmarkTuneVisit.started_at, BenchmarkTuneVisit.visit_id)
            result = await db.execute(query)
            return [BenchmarkTuneVisitModel.model_validate(row) for row in result.scalars().all()]


class BenchmarkTunePauseTable:
    async def open_pause(
        self, sweep_id: str, *, trigger: str, rnd: Optional[int] = None, visit_id: Optional[int] = None,
        attempt: int = 1, drift_ratio: Optional[float] = None, throttle_before: Optional[str] = None,
        temp_before: Optional[float] = None, power_before: Optional[float] = None,
        db: Optional[AsyncSession] = None,
    ) -> int:
        async with get_async_db_context(db) as db:
            row = BenchmarkTunePause(
                sweep_id=sweep_id, round=rnd, visit_id=visit_id, started_at=int(time.time()),
                trigger_kind=trigger, drift_ratio=drift_ratio, attempt=attempt,
                throttle_before=throttle_before, temp_before=temp_before, power_before=power_before,
            )
            db.add(row)
            await db.commit()
            await db.refresh(row)
            return row.pause_id

    async def close_pause(self, pause_id: int, *, resolution: str, db: Optional[AsyncSession] = None, **fields) -> None:
        allowed = ('throttle_after', 'temp_after', 'power_after', 'probe_tps')
        sets = {k: v for k, v in fields.items() if k in allowed}
        async with get_async_db_context(db) as db:
            result = await db.execute(select(BenchmarkTunePause).filter_by(pause_id=pause_id))
            row = result.scalars().first()
            if row:
                for k, v in sets.items():
                    setattr(row, k, v)
                row.resolution = resolution
                row.ended_at = int(time.time())
                await db.commit()

    async def sweep_pauses(self, sweep_id: str, db: Optional[AsyncSession] = None) -> list[dict]:
        """Replaces sqlite's v_tune_pause view (adds `seconds` + `interrupted_candidate`)."""
        async with get_async_db_context(db) as db:
            result = await db.execute(
                select(BenchmarkTunePause).filter_by(sweep_id=sweep_id).order_by(BenchmarkTunePause.started_at)
            )
            out = []
            for row in result.scalars().all():
                d = BenchmarkTunePauseModel.model_validate(row).model_dump()
                end = d['ended_at'] if d['ended_at'] is not None else d['started_at']
                d['seconds'] = end - d['started_at']
                interrupted = None
                if row.visit_id is not None:
                    visit = await db.execute(select(BenchmarkTuneVisit).filter_by(visit_id=row.visit_id))
                    visit_row = visit.scalars().first()
                    interrupted = visit_row.candidate_sha if visit_row else None
                d['interrupted_candidate'] = interrupted
                out.append(d)
            return out


BenchmarkTuneSweeps = BenchmarkTuneSweepTable()
BenchmarkTuneCandidates = BenchmarkTuneCandidateTable()
BenchmarkTuneRounds = BenchmarkTuneRoundTable()
BenchmarkTuneVisits = BenchmarkTuneVisitTable()
BenchmarkTunePauses = BenchmarkTunePauseTable()
