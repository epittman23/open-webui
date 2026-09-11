import logging
import time
from typing import Optional

from open_webui.internal.db import Base, get_async_db_context
from pydantic import BaseModel, ConfigDict
from sqlalchemy import (
    JSON,
    BigInteger,
    Column,
    Float,
    ForeignKey,
    Integer,
    Text,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession

log = logging.getLogger(__name__)


####################
# Benchmark Config / Run / Run-Load-Info DB Schema
#
# Ported from scripts/llama_db.py's `config`, `run` and `run_load_info`
# tables (local-llm outer repo). config_id is a sha1[:8] fingerprint of the
# serving-flag text computed outside this app (by the serving shell layer);
# config_text is stored verbatim because it is what the fingerprint covers,
# and the typed columns beside it are parsed out of that same text so a
# column can never disagree with the hash that identifies the row.
####################


class BenchmarkConfig(Base):
    __tablename__ = 'benchmark_config'
    config_id = Column(Text, primary_key=True)
    alias = Column(Text, nullable=False)
    config_text = Column(Text, nullable=False)
    arch = Column(Text, nullable=True)
    ngl = Column(Integer, nullable=True)
    ctx = Column(Integer, nullable=True)
    parallel = Column(Integer, nullable=True)
    threads = Column(Integer, nullable=True)
    moe = Column(Integer, nullable=True)
    override_tensors = Column(Text, nullable=True)
    speculative = Column(Text, nullable=True)
    spec_draft_n_max = Column(Integer, nullable=True)
    cache_k = Column(Text, nullable=True)
    cache_v = Column(Text, nullable=True)
    flash_attn = Column(Text, nullable=True)
    batch = Column(Integer, nullable=True)
    ubatch = Column(Integer, nullable=True)
    reasoning_effort = Column(Text, nullable=True)
    samplers = Column(Text, nullable=True)
    first_seen = Column(BigInteger, nullable=False)


class BenchmarkConfigModel(BaseModel):
    config_id: str
    alias: str
    config_text: str
    arch: Optional[str] = None
    ngl: Optional[int] = None
    ctx: Optional[int] = None
    parallel: Optional[int] = None
    threads: Optional[int] = None
    moe: Optional[int] = None
    override_tensors: Optional[str] = None
    speculative: Optional[str] = None
    spec_draft_n_max: Optional[int] = None
    cache_k: Optional[str] = None
    cache_v: Optional[str] = None
    flash_attn: Optional[str] = None
    batch: Optional[int] = None
    ubatch: Optional[int] = None
    reasoning_effort: Optional[str] = None
    samplers: Optional[str] = None
    first_seen: int

    model_config = ConfigDict(from_attributes=True)


class BenchmarkRun(Base):
    __tablename__ = 'benchmark_run'
    run_id = Column(Integer, primary_key=True, autoincrement=True)
    config_id = Column(Text, ForeignKey('benchmark_config.config_id'), nullable=False)
    model = Column(Text, nullable=False)
    quant = Column(Text, nullable=False)
    build = Column(Text, nullable=False)
    port = Column(Integer, nullable=False)
    pid = Column(Integer, nullable=True)
    started_at = Column(BigInteger, nullable=False)
    ended_at = Column(BigInteger, nullable=True)
    ended_reason = Column(Text, nullable=True)


class BenchmarkRunModel(BaseModel):
    run_id: int
    config_id: str
    model: str
    quant: str
    build: str
    port: int
    pid: Optional[int] = None
    started_at: int
    ended_at: Optional[int] = None
    ended_reason: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class BenchmarkRunLoadInfo(Base):
    __tablename__ = 'benchmark_run_load_info'
    run_id = Column(Integer, ForeignKey('benchmark_run.run_id', ondelete='CASCADE'), primary_key=True)
    n_layer = Column(Integer, nullable=True)
    n_layer_all = Column(Integer, nullable=True)
    layers_gpu = Column(Integer, nullable=True)
    layers_total = Column(Integer, nullable=True)
    layers_derived = Column(Integer, nullable=True)
    n_slots = Column(Integer, nullable=True)
    n_ctx_slot = Column(Integer, nullable=True)
    kv_unified = Column(Text, nullable=True)
    fused_gdn = Column(Text, nullable=True)
    mtp_head = Column(Text, nullable=True)
    buffers = Column(JSON, nullable=True)
    cpu_buffer_mib = Column(Float, nullable=True)
    gpu_buffer_mib = Column(Float, nullable=True)
    unused_tensors = Column(Integer, nullable=True)
    unused_prefixes = Column(JSON, nullable=True)
    warnings = Column(JSON, nullable=True)
    deprecated = Column(JSON, nullable=True)


class BenchmarkRunLoadInfoModel(BaseModel):
    run_id: int
    n_layer: Optional[int] = None
    n_layer_all: Optional[int] = None
    layers_gpu: Optional[int] = None
    layers_total: Optional[int] = None
    layers_derived: Optional[int] = None
    n_slots: Optional[int] = None
    n_ctx_slot: Optional[int] = None
    kv_unified: Optional[str] = None
    fused_gdn: Optional[str] = None
    mtp_head: Optional[str] = None
    buffers: Optional[dict] = None
    cpu_buffer_mib: Optional[float] = None
    gpu_buffer_mib: Optional[float] = None
    unused_tensors: Optional[int] = None
    unused_prefixes: Optional[list] = None
    warnings: Optional[list] = None
    deprecated: Optional[list] = None

    model_config = ConfigDict(from_attributes=True)


class BenchmarkConfigTable:
    async def upsert(
        self,
        config_id: str,
        alias: str,
        config_text: str,
        fields: dict,
        db: Optional[AsyncSession] = None,
    ) -> Optional[BenchmarkConfigModel]:
        """Record a serving configuration, keyed by its fingerprint.

        Left alone on conflict, matching the original's ON CONFLICT DO
        NOTHING: config_text is what the id was fingerprinted over, so an
        existing row under the same id is by definition the same config.
        """
        async with get_async_db_context(db) as db:
            existing = await db.execute(select(BenchmarkConfig).filter_by(config_id=config_id))
            if existing.scalars().first():
                return await self.get_by_id(config_id, db=db)

            row = BenchmarkConfig(
                config_id=config_id,
                alias=alias,
                config_text=config_text,
                first_seen=int(time.time()),
                **fields,
            )
            db.add(row)
            await db.commit()
            await db.refresh(row)
            return BenchmarkConfigModel.model_validate(row)

    async def get_by_id(self, config_id: str, db: Optional[AsyncSession] = None) -> Optional[BenchmarkConfigModel]:
        async with get_async_db_context(db) as db:
            result = await db.execute(select(BenchmarkConfig).filter_by(config_id=config_id))
            row = result.scalars().first()
            return BenchmarkConfigModel.model_validate(row) if row else None

    async def list_all(self, db: Optional[AsyncSession] = None) -> list[BenchmarkConfigModel]:
        async with get_async_db_context(db) as db:
            result = await db.execute(select(BenchmarkConfig))
            return [BenchmarkConfigModel.model_validate(row) for row in result.scalars().all()]


class BenchmarkRunTable:
    async def open_run(
        self,
        config_id: str,
        *,
        model: str,
        quant: str,
        build: str,
        port: int,
        pid: int,
        db: Optional[AsyncSession] = None,
    ) -> int:
        async with get_async_db_context(db) as db:
            row = BenchmarkRun(
                config_id=config_id,
                model=model,
                quant=quant,
                build=build,
                port=port,
                pid=pid,
                started_at=int(time.time()),
            )
            db.add(row)
            await db.commit()
            await db.refresh(row)
            return row.run_id

    async def close_run(self, run_id: int, reason: str = 'clean', db: Optional[AsyncSession] = None) -> None:
        async with get_async_db_context(db) as db:
            result = await db.execute(select(BenchmarkRun).filter_by(run_id=run_id))
            row = result.scalars().first()
            if row and row.ended_at is None:
                row.ended_at = int(time.time())
                row.ended_reason = reason
                await db.commit()

    async def get_active_run(self, port: int, db: Optional[AsyncSession] = None) -> Optional[BenchmarkRunModel]:
        async with get_async_db_context(db) as db:
            result = await db.execute(
                select(BenchmarkRun)
                .filter_by(port=port, ended_at=None)
                .order_by(BenchmarkRun.started_at.desc(), BenchmarkRun.run_id.desc())
            )
            row = result.scalars().first()
            return BenchmarkRunModel.model_validate(row) if row else None

    async def get_any_active_run(self, db: Optional[AsyncSession] = None) -> Optional[BenchmarkRunModel]:
        """Whether ANY server is currently being recorded, regardless of port.

        Used by the Serve router's single-flight guard (409 on a second
        `/serve/start`) -- a DB-backed check, unlike the CLI dashboard's
        in-memory job registry, so it stays correct across a backend restart.
        """
        async with get_async_db_context(db) as db:
            result = await db.execute(
                select(BenchmarkRun).filter_by(ended_at=None).order_by(BenchmarkRun.started_at.desc())
            )
            row = result.scalars().first()
            return BenchmarkRunModel.model_validate(row) if row else None

    async def get_by_id(self, run_id: int, db: Optional[AsyncSession] = None) -> Optional[BenchmarkRunModel]:
        async with get_async_db_context(db) as db:
            result = await db.execute(select(BenchmarkRun).filter_by(run_id=run_id))
            row = result.scalars().first()
            return BenchmarkRunModel.model_validate(row) if row else None

    async def list_runs(
        self, *, limit: int = 20, active_only: bool = False, db: Optional[AsyncSession] = None
    ) -> list[BenchmarkRunModel]:
        async with get_async_db_context(db) as db:
            query = select(BenchmarkRun)
            if active_only:
                query = query.filter(BenchmarkRun.ended_at.is_(None))
            query = query.order_by(BenchmarkRun.run_id.desc()).limit(limit)
            result = await db.execute(query)
            return [BenchmarkRunModel.model_validate(row) for row in result.scalars().all()]

    async def get_latest_run_per_config(self, db: Optional[AsyncSession] = None) -> list[BenchmarkRunModel]:
        """Each configuration's most recent run. Replaces sqlite's v_config_latest view.

        A plain Python group-by over `list_runs`, per this migration's
        decision to keep derived views out of the schema (see benchmarks/
        compare.py for the fuller serving-summary aggregation built on top
        of this).
        """
        async with get_async_db_context(db) as db:
            result = await db.execute(select(BenchmarkRun).order_by(BenchmarkRun.started_at.desc()))
            latest: dict[str, BenchmarkRun] = {}
            for row in result.scalars().all():
                if row.config_id not in latest:
                    latest[row.config_id] = row
            return [BenchmarkRunModel.model_validate(row) for row in latest.values()]


class BenchmarkRunLoadInfoTable:
    async def set_load_info(self, run_id: int, info: dict, db: Optional[AsyncSession] = None) -> None:
        async with get_async_db_context(db) as db:
            result = await db.execute(select(BenchmarkRunLoadInfo).filter_by(run_id=run_id))
            row = result.scalars().first()
            if row:
                for key, value in info.items():
                    setattr(row, key, value)
            else:
                db.add(BenchmarkRunLoadInfo(run_id=run_id, **info))
            await db.commit()

    async def get_by_run_id(self, run_id: int, db: Optional[AsyncSession] = None) -> Optional[BenchmarkRunLoadInfoModel]:
        async with get_async_db_context(db) as db:
            result = await db.execute(select(BenchmarkRunLoadInfo).filter_by(run_id=run_id))
            row = result.scalars().first()
            return BenchmarkRunLoadInfoModel.model_validate(row) if row else None


BenchmarkConfigs = BenchmarkConfigTable()
BenchmarkRuns = BenchmarkRunTable()
BenchmarkRunLoadInfos = BenchmarkRunLoadInfoTable()
