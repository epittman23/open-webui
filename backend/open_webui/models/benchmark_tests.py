import logging
import time
from typing import Optional

from open_webui.internal.db import Base, get_async_db_context
from pydantic import BaseModel, ConfigDict
from sqlalchemy import (
    JSON,
    BigInteger,
    CheckConstraint,
    Column,
    Float,
    ForeignKey,
    Integer,
    Text,
    UniqueConstraint,
    case,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

log = logging.getLogger(__name__)


####################
# Benchmark Request / Result / Answer / Exclusion / Schema-Note DB Schema
#
# Ported from scripts/llama_db.py's `request`, `result`, `answer`,
# `suite_exclusion` and `schema_note` tables -- the graded-item side of the
# suite (as opposed to benchmark_configs.py's serving-config/run side).
####################


class BenchmarkRequest(Base):
    __tablename__ = 'benchmark_request'
    request_id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey('benchmark_run.run_id', ondelete='CASCADE'), nullable=True)
    at = Column(BigInteger, nullable=False)
    model = Column(Text, nullable=True)
    label = Column(Text, nullable=True)
    wall_ms = Column(Float, nullable=True)
    params = Column(JSON, nullable=True)
    cache_n = Column(Integer, nullable=True)
    prompt_n = Column(Integer, nullable=True)
    prompt_ms = Column(Float, nullable=True)
    prompt_per_token_ms = Column(Float, nullable=True)
    prompt_per_second = Column(Float, nullable=True)
    predicted_n = Column(Integer, nullable=True)
    predicted_ms = Column(Float, nullable=True)
    predicted_per_token_ms = Column(Float, nullable=True)
    predicted_per_second = Column(Float, nullable=True)
    draft_n = Column(Integer, nullable=True)
    draft_n_accepted = Column(Integer, nullable=True)
    timings = Column(JSON, nullable=True)


class BenchmarkRequestModel(BaseModel):
    request_id: int
    run_id: Optional[int] = None
    at: int
    model: Optional[str] = None
    label: Optional[str] = None
    wall_ms: Optional[float] = None
    params: Optional[dict] = None
    cache_n: Optional[int] = None
    prompt_n: Optional[int] = None
    prompt_ms: Optional[float] = None
    prompt_per_token_ms: Optional[float] = None
    prompt_per_second: Optional[float] = None
    predicted_n: Optional[int] = None
    predicted_ms: Optional[float] = None
    predicted_per_token_ms: Optional[float] = None
    predicted_per_second: Optional[float] = None
    draft_n: Optional[int] = None
    draft_n_accepted: Optional[int] = None
    timings: Optional[dict] = None

    model_config = ConfigDict(from_attributes=True)


OUTCOMES = ('pass', 'fail_assert', 'fail_error', 'fail_timeout', 'no_code', 'skipped')


class BenchmarkResult(Base):
    __tablename__ = 'benchmark_result'
    __table_args__ = (
        UniqueConstraint('suite_run_id', 'benchmark', 'item_id', name='uq_benchmark_result_item'),
        CheckConstraint(f"outcome IN {OUTCOMES!r}", name='ck_benchmark_result_outcome'),
    )
    result_id = Column(Integer, primary_key=True, autoincrement=True)
    suite_run_id = Column(Text, nullable=False)
    run_id = Column(Integer, ForeignKey('benchmark_run.run_id', ondelete='SET NULL'), nullable=True)
    request_id = Column(Integer, ForeignKey('benchmark_request.request_id', ondelete='SET NULL'), nullable=True)
    # Denormalised on purpose: a result keeps its configuration identity even
    # if its run row is pruned. NULL means the server was started by hand and
    # no configuration was recorded.
    config_id = Column(Text, ForeignKey('benchmark_config.config_id'), nullable=True)
    at = Column(BigInteger, nullable=False)
    model = Column(Text, nullable=False)
    profile = Column(Text, nullable=True)
    benchmark = Column(Text, nullable=False)
    item_id = Column(Text, nullable=False)
    dataset_revision = Column(Text, nullable=False)
    tier = Column(Text, nullable=False)
    seed = Column(Integer, nullable=False)
    outcome = Column(Text, nullable=False)
    reason = Column(Text, nullable=False, default='')
    reasoning_chars = Column(Integer, nullable=True)
    wall_ms = Column(Float, nullable=True)
    params = Column(JSON, nullable=True)
    timings = Column(JSON, nullable=True)
    # The system prompt and adapter a result was measured under. NULL on
    # system_name/system_sha means no system prompt was sent; NULL on
    # adapter_sha means unknown (row predates adapter fingerprinting) -- see
    # scripts/llama_db.py migrations 2 and 3 for the full reasoning, ported
    # verbatim into the historical-data backfill notes rather than re-derived.
    system_name = Column(Text, nullable=True)
    system_sha = Column(Text, nullable=True)
    adapter_sha = Column(Text, nullable=True)


class BenchmarkResultModel(BaseModel):
    result_id: int
    suite_run_id: str
    run_id: Optional[int] = None
    request_id: Optional[int] = None
    config_id: Optional[str] = None
    at: int
    model: str
    profile: Optional[str] = None
    benchmark: str
    item_id: str
    dataset_revision: str
    tier: str
    seed: int
    outcome: str
    reason: str = ''
    reasoning_chars: Optional[int] = None
    wall_ms: Optional[float] = None
    params: Optional[dict] = None
    timings: Optional[dict] = None
    system_name: Optional[str] = None
    system_sha: Optional[str] = None
    adapter_sha: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class BenchmarkAnswer(Base):
    __tablename__ = 'benchmark_answer'
    result_id = Column(Integer, ForeignKey('benchmark_result.result_id', ondelete='CASCADE'), primary_key=True)
    prompt = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    reasoning = Column(Text, nullable=False)


class BenchmarkAnswerModel(BaseModel):
    result_id: int
    prompt: str
    content: str
    reasoning: str

    model_config = ConfigDict(from_attributes=True)


class BenchmarkSuiteExclusion(Base):
    __tablename__ = 'benchmark_suite_exclusion'
    benchmark = Column(Text, primary_key=True)
    item_id = Column(Text, primary_key=True)
    dataset_revision = Column(Text, primary_key=True)
    kind = Column(Text, nullable=False)
    reason = Column(Text, nullable=False)
    recorded_at = Column(BigInteger, nullable=False)


class BenchmarkSuiteExclusionModel(BaseModel):
    benchmark: str
    item_id: str
    dataset_revision: str
    kind: str
    reason: str
    recorded_at: int

    model_config = ConfigDict(from_attributes=True)


class BenchmarkSchemaNote(Base):
    __tablename__ = 'benchmark_schema_note'
    note_id = Column(Integer, primary_key=True, autoincrement=True)
    noted_on = Column(Text, nullable=False)
    note = Column(Text, nullable=False, unique=True)


class BenchmarkSchemaNoteModel(BaseModel):
    note_id: int
    noted_on: str
    note: str

    model_config = ConfigDict(from_attributes=True)


class BenchmarkRequestTable:
    async def add_request(
        self,
        run_id: Optional[int],
        *,
        at: int,
        model: str,
        label: str,
        wall_ms: Optional[float],
        params: dict,
        timings: dict,
        db: Optional[AsyncSession] = None,
    ) -> int:
        timing_keys = [
            'cache_n', 'prompt_n', 'prompt_ms', 'prompt_per_token_ms', 'prompt_per_second',
            'predicted_n', 'predicted_ms', 'predicted_per_token_ms', 'predicted_per_second',
            'draft_n', 'draft_n_accepted',
        ]
        async with get_async_db_context(db) as db:
            row = BenchmarkRequest(
                run_id=run_id,
                at=at,
                model=model,
                label=label,
                wall_ms=wall_ms,
                params=params,
                timings=timings,
                **{k: timings.get(k) for k in timing_keys},
            )
            db.add(row)
            await db.commit()
            await db.refresh(row)
            return row.request_id

    async def get_run_request_aggregate(self, run_id: int, db: Optional[AsyncSession] = None) -> Optional[dict]:
        """Per-run request totals: cold/warm counts and summed cold-prefill/gen/draft figures.

        Added for benchmarks/compare.py's serving_summary(), which replaces
        the original llama_db.serving_summary()'s raw SQL aggregate over
        `request` with this query method (a plain SUM/COUNT, so it belongs
        here rather than being reassembled row-by-row in Python).

        Cold means no cache reuse (cache_n NULL or 0); warm reused some
        cached prefix. Only *cold* prompt tokens/ms are summed -- a request
        whose prompt was partly served from a slot measures the cache, not
        the serving flags, which is the same reason Group.prefill_tps()
        excludes it in compare.py.
        """
        async with get_async_db_context(db) as db:
            is_cold = func.coalesce(BenchmarkRequest.cache_n, 0) == 0
            is_warm = func.coalesce(BenchmarkRequest.cache_n, 0) > 0
            query = select(
                func.count().label('requests'),
                func.sum(func.cast(is_cold, Integer)).label('cold_reqs'),
                func.sum(func.cast(is_warm, Integer)).label('warm_reqs'),
                func.sum(case((is_cold, BenchmarkRequest.prompt_n), else_=None)).label('cold_prompt_n'),
                func.sum(case((is_cold, BenchmarkRequest.prompt_ms), else_=None)).label('cold_prompt_ms'),
                func.sum(BenchmarkRequest.predicted_n).label('predicted_n'),
                func.sum(BenchmarkRequest.predicted_ms).label('predicted_ms'),
                func.sum(BenchmarkRequest.draft_n).label('draft_n'),
                func.sum(BenchmarkRequest.draft_n_accepted).label('draft_n_accepted'),
            ).filter(BenchmarkRequest.run_id == run_id)
            result = await db.execute(query)
            row = result.first()
            if not row or not row.requests:
                return None
            return dict(row._mapping)


class BenchmarkResultTable:
    async def add_result(
        self, record: dict, answer: Optional[dict] = None, db: Optional[AsyncSession] = None
    ) -> int:
        """One graded item, and its request and answer, in one transaction.

        Committed per item, not per suite: a `full` run is many hours and
        will be interrupted, and holding results in memory would mean the
        run whose partial results are most worth having produced none.
        """
        async with get_async_db_context(db) as db:
            request_id = None
            if record.get('timings'):
                request_id = await BenchmarkRequests.add_request(
                    record.get('run_id'),
                    at=record['at'],
                    model=record['model'],
                    label=f"{record['benchmark']}/{record['item_id']}",
                    wall_ms=record.get('wall_ms'),
                    params=record.get('params') or {},
                    timings=record['timings'],
                    db=db,
                )

            values = dict(
                suite_run_id=record['suite_run_id'],
                run_id=record.get('run_id'),
                request_id=request_id,
                config_id=record.get('config_id'),
                at=record['at'],
                model=record['model'],
                profile=record.get('profile', ''),
                benchmark=record['benchmark'],
                item_id=record['item_id'],
                dataset_revision=record.get('dataset_revision', ''),
                tier=record['tier'],
                seed=int(record.get('seed') or 0),
                outcome=record['outcome'],
                reason=record.get('reason', ''),
                reasoning_chars=record.get('reasoning_chars'),
                wall_ms=record.get('wall_ms'),
                params=record.get('params') or {},
                timings=record.get('timings') or {},
                # NULL, not "", for no system prompt -- "" would be a third
                # state meaning the same thing while grouping separately.
                system_name=record.get('system_name') or None,
                system_sha=record.get('system_sha') or None,
                # NULL means the adapter was not fingerprinted (a real
                # unknown), not a default.
                adapter_sha=record.get('adapter_sha') or None,
            )
            stmt = pg_insert(BenchmarkResult).values(**values)
            update_cols = {
                'request_id': stmt.excluded.request_id,
                'at': stmt.excluded.at,
                'outcome': stmt.excluded.outcome,
                'reason': stmt.excluded.reason,
                'reasoning_chars': stmt.excluded.reasoning_chars,
                'wall_ms': stmt.excluded.wall_ms,
                'params': stmt.excluded.params,
                'timings': stmt.excluded.timings,
                # Re-running an item under a different --system must not
                # leave the row claiming the prompt the first attempt used.
                'system_name': stmt.excluded.system_name,
                'system_sha': stmt.excluded.system_sha,
                'adapter_sha': stmt.excluded.adapter_sha,
            }
            stmt = stmt.on_conflict_do_update(
                index_elements=['suite_run_id', 'benchmark', 'item_id'], set_=update_cols
            ).returning(BenchmarkResult.result_id)
            result = await db.execute(stmt)
            result_id = result.scalar_one()

            if answer is not None:
                answer_stmt = pg_insert(BenchmarkAnswer).values(
                    result_id=result_id,
                    prompt=answer.get('prompt', ''),
                    content=answer.get('content', ''),
                    reasoning=answer.get('reasoning', ''),
                )
                answer_stmt = answer_stmt.on_conflict_do_update(
                    index_elements=['result_id'],
                    set_={
                        'prompt': answer_stmt.excluded.prompt,
                        'content': answer_stmt.excluded.content,
                        'reasoning': answer_stmt.excluded.reasoning,
                    },
                )
                await db.execute(answer_stmt)

            await db.commit()
            return result_id

    async def list_results(
        self,
        *,
        tier: Optional[str] = None,
        model: Optional[str] = None,
        suite_run_id: Optional[str] = None,
        outcome_in: Optional[list[str]] = None,
        db: Optional[AsyncSession] = None,
    ) -> list[BenchmarkResultModel]:
        async with get_async_db_context(db) as db:
            query = select(BenchmarkResult)
            if tier:
                query = query.filter(BenchmarkResult.tier == tier)
            if model:
                query = query.filter(BenchmarkResult.model == model)
            if suite_run_id:
                query = query.filter(BenchmarkResult.suite_run_id == suite_run_id)
            if outcome_in:
                query = query.filter(BenchmarkResult.outcome.in_(outcome_in))
            query = query.order_by(BenchmarkResult.at, BenchmarkResult.result_id)
            result = await db.execute(query)
            return [BenchmarkResultModel.model_validate(row) for row in result.scalars().all()]

    async def completed(self, suite_run_id: str, db: Optional[AsyncSession] = None) -> set[tuple[str, str]]:
        async with get_async_db_context(db) as db:
            result = await db.execute(
                select(BenchmarkResult.benchmark, BenchmarkResult.item_id).filter_by(suite_run_id=suite_run_id)
            )
            return {(row.benchmark, row.item_id) for row in result.all()}

    async def latest_suite_run(
        self, *, model: Optional[str] = None, tier: Optional[str] = None, db: Optional[AsyncSession] = None
    ) -> Optional[str]:
        async with get_async_db_context(db) as db:
            query = select(BenchmarkResult.suite_run_id)
            if model:
                query = query.filter(BenchmarkResult.model == model)
            if tier:
                query = query.filter(BenchmarkResult.tier == tier)
            query = query.order_by(BenchmarkResult.at.desc(), BenchmarkResult.result_id.desc()).limit(1)
            result = await db.execute(query)
            row = result.scalars().first()
            return row

    async def suite_runs(self, limit: int = 20, db: Optional[AsyncSession] = None) -> list[dict]:
        """Recent suite runs, newest first, with what each of them scored.

        `skipped` is left out of the denominator: an item nobody attempted
        is not an item anybody failed.
        """
        async with get_async_db_context(db) as db:
            passed_case = func.sum(func.cast(BenchmarkResult.outcome == 'pass', Integer))
            attempted_case = func.sum(func.cast(BenchmarkResult.outcome != 'skipped', Integer))
            query = (
                select(
                    BenchmarkResult.suite_run_id,
                    func.min(BenchmarkResult.at).label('started_at'),
                    func.max(BenchmarkResult.at).label('ended_at'),
                    BenchmarkResult.model,
                    BenchmarkResult.tier,
                    attempted_case.label('attempted'),
                    passed_case.label('passed'),
                )
                .group_by(BenchmarkResult.suite_run_id, BenchmarkResult.model, BenchmarkResult.tier)
                .order_by(func.min(BenchmarkResult.at).desc())
                .limit(limit)
            )
            result = await db.execute(query)
            return [dict(row._mapping) for row in result.all()]

    async def get_pass_rate_by_group(self, db: Optional[AsyncSession] = None) -> list[dict]:
        """(model, config_id, tier, benchmark, system_sha, adapter_sha) -> pass rate.

        Replaces sqlite's v_pass_rate view -- a plain Python-side query
        method per this migration's decision to keep derived views out of
        the schema.
        """
        async with get_async_db_context(db) as db:
            passed = func.sum(func.cast(BenchmarkResult.outcome == 'pass', Integer))
            attempted = func.count()
            query = (
                select(
                    BenchmarkResult.model,
                    BenchmarkResult.config_id,
                    BenchmarkResult.tier,
                    BenchmarkResult.benchmark,
                    BenchmarkResult.system_sha,
                    BenchmarkResult.adapter_sha,
                    passed.label('passed'),
                    attempted.label('attempted'),
                )
                .filter(BenchmarkResult.outcome != 'skipped')
                .group_by(
                    BenchmarkResult.model,
                    BenchmarkResult.config_id,
                    BenchmarkResult.tier,
                    BenchmarkResult.benchmark,
                    BenchmarkResult.system_sha,
                    BenchmarkResult.adapter_sha,
                )
            )
            result = await db.execute(query)
            rows = []
            for row in result.all():
                d = dict(row._mapping)
                d['pass_rate'] = (d['passed'] / d['attempted']) if d['attempted'] else None
                rows.append(d)
            return rows


class BenchmarkAnswerTable:
    async def answer_for(
        self,
        benchmark: str,
        item_id: str,
        suite_run_id: Optional[str] = None,
        db: Optional[AsyncSession] = None,
    ) -> Optional[dict]:
        async with get_async_db_context(db) as db:
            query = (
                select(BenchmarkResult, BenchmarkAnswer)
                .join(BenchmarkAnswer, BenchmarkAnswer.result_id == BenchmarkResult.result_id)
                .filter(BenchmarkResult.benchmark == benchmark, BenchmarkResult.item_id == item_id)
            )
            if suite_run_id:
                query = query.filter(BenchmarkResult.suite_run_id == suite_run_id)
            query = query.order_by(BenchmarkResult.at.desc(), BenchmarkResult.result_id.desc()).limit(1)
            result = await db.execute(query)
            row = result.first()
            if not row:
                return None
            result_row, answer_row = row
            record = BenchmarkResultModel.model_validate(result_row).model_dump()
            record.update(
                prompt=answer_row.prompt, content=answer_row.content, reasoning=answer_row.reasoning
            )
            return record


class BenchmarkSuiteExclusionTable:
    async def record_exclusions(self, rows: list[dict], db: Optional[AsyncSession] = None) -> int:
        stamp = int(time.time())
        async with get_async_db_context(db) as db:
            for r in rows:
                stmt = pg_insert(BenchmarkSuiteExclusion).values(
                    benchmark=r['benchmark'],
                    item_id=str(r['item_id']),
                    dataset_revision=r.get('dataset_revision', ''),
                    kind=r.get('kind', 'excluded'),
                    reason=r.get('reason', '')[:400],
                    recorded_at=stamp,
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=['benchmark', 'item_id', 'dataset_revision'],
                    set_={'kind': stmt.excluded.kind, 'reason': stmt.excluded.reason, 'recorded_at': stmt.excluded.recorded_at},
                )
                await db.execute(stmt)
            await db.commit()
            return len(rows)

    async def list_all(self, db: Optional[AsyncSession] = None) -> list[BenchmarkSuiteExclusionModel]:
        async with get_async_db_context(db) as db:
            result = await db.execute(
                select(BenchmarkSuiteExclusion).order_by(
                    BenchmarkSuiteExclusion.benchmark, BenchmarkSuiteExclusion.item_id
                )
            )
            return [BenchmarkSuiteExclusionModel.model_validate(row) for row in result.scalars().all()]


class BenchmarkSchemaNoteTable:
    async def add_note(self, noted_on: str, note: str, db: Optional[AsyncSession] = None) -> None:
        async with get_async_db_context(db) as db:
            stmt = pg_insert(BenchmarkSchemaNote).values(noted_on=noted_on, note=note)
            stmt = stmt.on_conflict_do_nothing(index_elements=['note'])
            await db.execute(stmt)
            await db.commit()

    async def list_notes(self, db: Optional[AsyncSession] = None) -> list[BenchmarkSchemaNoteModel]:
        async with get_async_db_context(db) as db:
            result = await db.execute(select(BenchmarkSchemaNote).order_by(BenchmarkSchemaNote.note_id))
            return [BenchmarkSchemaNoteModel.model_validate(row) for row in result.scalars().all()]


BenchmarkRequests = BenchmarkRequestTable()
BenchmarkResults = BenchmarkResultTable()
BenchmarkAnswers = BenchmarkAnswerTable()
BenchmarkSuiteExclusions = BenchmarkSuiteExclusionTable()
BenchmarkSchemaNotes = BenchmarkSchemaNoteTable()
