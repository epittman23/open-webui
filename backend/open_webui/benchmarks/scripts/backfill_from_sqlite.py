#!/usr/bin/env python3
"""backfill_from_sqlite.py - one-time copy of the pre-migration sqlite store into Postgres.

Part of the migration that retires local-llm's standalone `lllm-test`/
`lllm-compare`/`lllm-report`/`lllm-tune` CLI suite in favour of this fork's
own Benchmarks feature. That CLI kept its own history in a sqlite file
(`logs/llama.db` in the outer local-llm repo); this script copies every row
of it, once, into the `benchmark_*` tables this app now owns, preserving
primary keys exactly so foreign-key relationships stay intact -- it is a
schema-preserving copy, not a transform (the Alembic migration that created
these tables was deliberately kept isomorphic to the sqlite original for
exactly this reason).

Run once, then verify, then the sqlite file can be retired. Safe to run
before that: it does not modify or delete `logs/llama.db`. Guarded against
an accidental second run by a sentinel `benchmark_schema_note` row -- pass
--force to re-run anyway (it will re-insert every row `ON CONFLICT DO
NOTHING`, so a second run is a no-op on rows that made it in the first
time, not a duplication).

Usage:
    cd backend && .venv/bin/python -m open_webui.benchmarks.scripts.backfill_from_sqlite \\
        --sqlite-path /path/to/logs/llama.db [--database-url postgresql://...] [--force]

Deliberately standalone (plain psycopg, not the async SQLAlchemy models):
this runs once, outside the running backend process, the same reasoning
`telemetry_recorder.py` follows for staying a simple script rather than
requiring an event loop.
"""

from __future__ import annotations

import argparse
import calendar
import json
import sqlite3
import time
from pathlib import Path

import psycopg
from psycopg.types.json import Jsonb

from open_webui.benchmarks.telemetry_recorder import connect

SENTINEL_PREFIX = 'benchmarks: backfilled from sqlite llama.db on '


def _epoch(iso: str | None) -> int | None:
    """"2026-08-30T12:00:00Z" -> epoch seconds. The sqlite store's timestamps
    are always this exact UTC format (time.strftime with time.gmtime()), so
    calendar.timegm (UTC-aware) is used rather than time.mktime (local)."""
    if not iso:
        return None
    return calendar.timegm(time.strptime(iso, '%Y-%m-%dT%H:%M:%SZ'))


def _json(text: str | None) -> Jsonb | None:
    if not text:
        return None
    try:
        return Jsonb(json.loads(text))
    except (json.JSONDecodeError, TypeError):
        return None


def _bool(value) -> bool | None:
    return None if value is None else bool(value)


def _rows(sconn: sqlite3.Connection, table: str) -> list[sqlite3.Row]:
    return sconn.execute(f'SELECT * FROM {table}').fetchall()  # noqa: S608 - fixed table names, not user input


def _already_backfilled(pconn: psycopg.Connection) -> str | None:
    row = pconn.execute(
        "SELECT note FROM benchmark_schema_note WHERE note LIKE %s ORDER BY note_id DESC LIMIT 1",
        (SENTINEL_PREFIX + '%',),
    ).fetchone()
    return row['note'] if row else None


def _reset_sequence(pconn: psycopg.Connection, table: str, column: str) -> None:
    """After inserting explicit PKs into a serial column, point the sequence
    past the highest one backfilled -- otherwise the running app's next
    autoincrement INSERT collides with a historical row."""
    pconn.execute(
        f'SELECT setval(pg_get_serial_sequence(%s, %s), COALESCE((SELECT MAX({column}) FROM {table}), 0) + 1, false)',  # noqa: S608 - table/column are fixed call-site constants, not user input
        (table, column),
    )


def backfill_config(sconn, pconn) -> int:
    rows = _rows(sconn, 'config')
    for r in rows:
        pconn.execute(
            """
            INSERT INTO benchmark_config
                (config_id, alias, config_text, arch, ngl, ctx, parallel, threads, moe,
                 override_tensors, speculative, spec_draft_n_max, cache_k, cache_v,
                 flash_attn, batch, ubatch, reasoning_effort, samplers, first_seen)
            VALUES (%(config_id)s, %(alias)s, %(config_text)s, %(arch)s, %(ngl)s, %(ctx)s,
                    %(parallel)s, %(threads)s, %(moe)s, %(override_tensors)s, %(speculative)s,
                    %(spec_draft_n_max)s, %(cache_k)s, %(cache_v)s, %(flash_attn)s, %(batch)s,
                    %(ubatch)s, %(reasoning_effort)s, %(samplers)s, %(first_seen)s)
            ON CONFLICT (config_id) DO NOTHING
            """,
            {**dict(r), 'first_seen': _epoch(r['first_seen'])},
        )
    return len(rows)


def backfill_run(sconn, pconn) -> int:
    rows = _rows(sconn, 'run')
    for r in rows:
        pconn.execute(
            """
            INSERT INTO benchmark_run
                (run_id, config_id, model, quant, build, port, pid, started_at, ended_at, ended_reason)
            VALUES (%(run_id)s, %(config_id)s, %(model)s, %(quant)s, %(build)s, %(port)s, %(pid)s,
                    %(started_at)s, %(ended_at)s, %(ended_reason)s)
            ON CONFLICT (run_id) DO NOTHING
            """,
            {**dict(r), 'started_at': _epoch(r['started_at']), 'ended_at': _epoch(r['ended_at'])},
        )
    _reset_sequence(pconn, 'benchmark_run', 'run_id')
    return len(rows)


def backfill_gpu_sample(sconn, pconn) -> int:
    rows = _rows(sconn, 'gpu_sample')
    for r in rows:
        pconn.execute(
            """
            INSERT INTO benchmark_gpu_sample
                (sample_id, run_id, at, temp_c, util_pct, mem_used_mib, mem_total_mib, power_w, sm_mhz, throttle)
            VALUES (%(sample_id)s, %(run_id)s, %(at)s, %(temp_c)s, %(util_pct)s, %(mem_used_mib)s,
                    %(mem_total_mib)s, %(power_w)s, %(sm_mhz)s, %(throttle)s)
            ON CONFLICT (sample_id) DO NOTHING
            """,
            {**dict(r), 'at': _epoch(r['at'])},
        )
    _reset_sequence(pconn, 'benchmark_gpu_sample', 'sample_id')
    return len(rows)


def backfill_metrics_scrape(sconn, pconn) -> int:
    rows = _rows(sconn, 'metrics_scrape')
    for r in rows:
        pconn.execute(
            """
            INSERT INTO benchmark_metrics_scrape (run_id, at, counter, value)
            VALUES (%(run_id)s, %(at)s, %(counter)s, %(value)s)
            ON CONFLICT (run_id, at, counter) DO NOTHING
            """,
            {**dict(r), 'at': _epoch(r['at'])},
        )
    return len(rows)


def backfill_run_load_info(sconn, pconn) -> int:
    rows = _rows(sconn, 'run_load_info')
    for r in rows:
        d = dict(r)
        pconn.execute(
            """
            INSERT INTO benchmark_run_load_info
                (run_id, n_layer, n_layer_all, layers_gpu, layers_total, layers_derived,
                 n_slots, n_ctx_slot, kv_unified, fused_gdn, mtp_head, buffers,
                 cpu_buffer_mib, gpu_buffer_mib, unused_tensors, unused_prefixes, warnings, deprecated)
            VALUES (%(run_id)s, %(n_layer)s, %(n_layer_all)s, %(layers_gpu)s, %(layers_total)s,
                    %(layers_derived)s, %(n_slots)s, %(n_ctx_slot)s, %(kv_unified)s, %(fused_gdn)s,
                    %(mtp_head)s, %(buffers)s, %(cpu_buffer_mib)s, %(gpu_buffer_mib)s,
                    %(unused_tensors)s, %(unused_prefixes)s, %(warnings)s, %(deprecated)s)
            ON CONFLICT (run_id) DO NOTHING
            """,
            {
                **d,
                'buffers': _json(d['buffers']),
                'unused_prefixes': _json(d['unused_prefixes']),
                'warnings': _json(d['warnings']),
                'deprecated': _json(d['deprecated']),
            },
        )
    return len(rows)


REQUEST_TIMING_COLS = [
    'cache_n', 'prompt_n', 'prompt_ms', 'prompt_per_token_ms', 'prompt_per_second',
    'predicted_n', 'predicted_ms', 'predicted_per_token_ms', 'predicted_per_second',
    'draft_n', 'draft_n_accepted',
]


def backfill_request(sconn, pconn) -> int:
    rows = _rows(sconn, 'request')
    cols = ', '.join(REQUEST_TIMING_COLS)
    placeholders = ', '.join(f'%({c})s' for c in REQUEST_TIMING_COLS)
    for r in rows:
        d = dict(r)
        pconn.execute(
            f"""
            INSERT INTO benchmark_request
                (request_id, run_id, at, model, label, wall_ms, params, {cols}, timings)
            VALUES (%(request_id)s, %(run_id)s, %(at)s, %(model)s, %(label)s, %(wall_ms)s,
                    %(params)s, {placeholders}, %(timings)s)
            ON CONFLICT (request_id) DO NOTHING
            """,  # noqa: S608 - REQUEST_TIMING_COLS is a fixed constant, not user input
            {**d, 'at': _epoch(d['at']), 'params': _json(d['params']), 'timings': _json(d['timings'])},
        )
    _reset_sequence(pconn, 'benchmark_request', 'request_id')
    return len(rows)


def backfill_result(sconn, pconn) -> int:
    rows = _rows(sconn, 'result')
    for r in rows:
        d = dict(r)
        pconn.execute(
            """
            INSERT INTO benchmark_result
                (result_id, suite_run_id, run_id, request_id, config_id, at, model, profile,
                 benchmark, item_id, dataset_revision, tier, seed, outcome, reason,
                 reasoning_chars, wall_ms, params, timings, system_name, system_sha, adapter_sha)
            VALUES (%(result_id)s, %(suite_run_id)s, %(run_id)s, %(request_id)s, %(config_id)s,
                    %(at)s, %(model)s, %(profile)s, %(benchmark)s, %(item_id)s, %(dataset_revision)s,
                    %(tier)s, %(seed)s, %(outcome)s, %(reason)s, %(reasoning_chars)s, %(wall_ms)s,
                    %(params)s, %(timings)s, %(system_name)s, %(system_sha)s, %(adapter_sha)s)
            ON CONFLICT (result_id) DO NOTHING
            """,
            {**d, 'at': _epoch(d['at']), 'params': _json(d['params']), 'timings': _json(d['timings'])},
        )
    _reset_sequence(pconn, 'benchmark_result', 'result_id')
    return len(rows)


def backfill_answer(sconn, pconn) -> int:
    rows = _rows(sconn, 'answer')
    for r in rows:
        pconn.execute(
            """
            INSERT INTO benchmark_answer (result_id, prompt, content, reasoning)
            VALUES (%(result_id)s, %(prompt)s, %(content)s, %(reasoning)s)
            ON CONFLICT (result_id) DO NOTHING
            """,
            dict(r),
        )
    return len(rows)


def backfill_suite_exclusion(sconn, pconn) -> int:
    rows = _rows(sconn, 'suite_exclusion')
    for r in rows:
        d = dict(r)
        pconn.execute(
            """
            INSERT INTO benchmark_suite_exclusion (benchmark, item_id, dataset_revision, kind, reason, recorded_at)
            VALUES (%(benchmark)s, %(item_id)s, %(dataset_revision)s, %(kind)s, %(reason)s, %(recorded_at)s)
            ON CONFLICT (benchmark, item_id, dataset_revision) DO NOTHING
            """,
            {**d, 'recorded_at': _epoch(d['recorded_at'])},
        )
    return len(rows)


def backfill_schema_note(sconn, pconn) -> int:
    rows = _rows(sconn, 'schema_note')
    for r in rows:
        pconn.execute(
            "INSERT INTO benchmark_schema_note (noted_on, note) VALUES (%(noted_on)s, %(note)s) "
            'ON CONFLICT (note) DO NOTHING',
            dict(r),
        )
    return len(rows)


def backfill_tune_sweep(sconn, pconn) -> int:
    rows = _rows(sconn, 'tune_sweep')
    for r in rows:
        d = dict(r)
        pconn.execute(
            """
            INSERT INTO benchmark_tune_sweep
                (sweep_id, started_at, ended_at, ended_reason, pid, profile, tier, benchmark,
                 system_name, system_sha, grid_path, grid_sha, item_order_sha, item_count,
                 budget_mode, budget_seconds, budget_visits, eta, round_items, candidates,
                 stages, objective, alpha, on_drift, seed, baseline_config_id, winner_candidate,
                 verdict, verdict_reason)
            VALUES (%(sweep_id)s, %(started_at)s, %(ended_at)s, %(ended_reason)s, %(pid)s,
                    %(profile)s, %(tier)s, %(benchmark)s, %(system_name)s, %(system_sha)s,
                    %(grid_path)s, %(grid_sha)s, %(item_order_sha)s, %(item_count)s, %(budget_mode)s,
                    %(budget_seconds)s, %(budget_visits)s, %(eta)s, %(round_items)s, %(candidates)s,
                    %(stages)s, %(objective)s, %(alpha)s, %(on_drift)s, %(seed)s,
                    %(baseline_config_id)s, %(winner_candidate)s, %(verdict)s, %(verdict_reason)s)
            ON CONFLICT (sweep_id) DO NOTHING
            """,
            {**d, 'started_at': _epoch(d['started_at']), 'ended_at': _epoch(d['ended_at'])},
        )
    return len(rows)


def backfill_tune_candidate(sconn, pconn) -> int:
    rows = _rows(sconn, 'tune_candidate')
    for r in rows:
        d = dict(r)
        pconn.execute(
            """
            INSERT INTO benchmark_tune_candidate
                (sweep_id, candidate_sha, stage, overrides, is_baseline, config_id, suite_run_id,
                 status, status_reason, score, eliminated_round, created_at)
            VALUES (%(sweep_id)s, %(candidate_sha)s, %(stage)s, %(overrides)s, %(is_baseline)s,
                    %(config_id)s, %(suite_run_id)s, %(status)s, %(status_reason)s, %(score)s,
                    %(eliminated_round)s, %(created_at)s)
            ON CONFLICT (sweep_id, candidate_sha) DO NOTHING
            """,
            {
                **d,
                'overrides': _json(d['overrides']) or Jsonb({}),
                'is_baseline': _bool(d['is_baseline']),
                'created_at': _epoch(d['created_at']),
            },
        )
    return len(rows)


def backfill_tune_round(sconn, pconn) -> int:
    rows = _rows(sconn, 'tune_round')
    for r in rows:
        d = dict(r)
        pconn.execute(
            """
            INSERT INTO benchmark_tune_round
                (sweep_id, round, stage, started_at, ended_at, item_from, item_to, survivors,
                 baseline_gen_tps, baseline_regime, drift_ratio, decision, notes)
            VALUES (%(sweep_id)s, %(round)s, %(stage)s, %(started_at)s, %(ended_at)s, %(item_from)s,
                    %(item_to)s, %(survivors)s, %(baseline_gen_tps)s, %(baseline_regime)s,
                    %(drift_ratio)s, %(decision)s, %(notes)s)
            ON CONFLICT (sweep_id, round) DO NOTHING
            """,
            {**d, 'started_at': _epoch(d['started_at']), 'ended_at': _epoch(d['ended_at'])},
        )
    return len(rows)


def backfill_tune_visit(sconn, pconn) -> int:
    rows = _rows(sconn, 'tune_visit')
    for r in rows:
        d = dict(r)
        pconn.execute(
            """
            INSERT INTO benchmark_tune_visit
                (visit_id, sweep_id, candidate_sha, round, attempt, run_id, config_id, started_at,
                 ended_at, server_pgid, load_ms, item_from, item_to, items_done,
                 since_pause_seconds, counts_toward_round, status, reason)
            VALUES (%(visit_id)s, %(sweep_id)s, %(candidate_sha)s, %(round)s, %(attempt)s,
                    %(run_id)s, %(config_id)s, %(started_at)s, %(ended_at)s, %(server_pgid)s,
                    %(load_ms)s, %(item_from)s, %(item_to)s, %(items_done)s,
                    %(since_pause_seconds)s, %(counts_toward_round)s, %(status)s, %(reason)s)
            ON CONFLICT (visit_id) DO NOTHING
            """,
            {
                **d,
                'started_at': _epoch(d['started_at']),
                'ended_at': _epoch(d['ended_at']),
                'counts_toward_round': _bool(d['counts_toward_round']),
            },
        )
    _reset_sequence(pconn, 'benchmark_tune_visit', 'visit_id')
    return len(rows)


def backfill_tune_pause(sconn, pconn) -> int:
    rows = _rows(sconn, 'tune_pause')
    for r in rows:
        d = dict(r)
        pconn.execute(
            """
            INSERT INTO benchmark_tune_pause
                (pause_id, sweep_id, round, visit_id, started_at, ended_at, trigger_kind,
                 drift_ratio, attempt, throttle_before, throttle_after, temp_before, temp_after,
                 power_before, power_after, probe_tps, resolution)
            VALUES (%(pause_id)s, %(sweep_id)s, %(round)s, %(visit_id)s, %(started_at)s,
                    %(ended_at)s, %(trigger_kind)s, %(drift_ratio)s, %(attempt)s,
                    %(throttle_before)s, %(throttle_after)s, %(temp_before)s, %(temp_after)s,
                    %(power_before)s, %(power_after)s, %(probe_tps)s, %(resolution)s)
            ON CONFLICT (pause_id) DO NOTHING
            """,
            {**d, 'started_at': _epoch(d['started_at']), 'ended_at': _epoch(d['ended_at'])},
        )
    _reset_sequence(pconn, 'benchmark_tune_pause', 'pause_id')
    return len(rows)


# Dependency order: FK parents before children.
STEPS = [
    ('config', backfill_config),
    ('run', backfill_run),
    ('gpu_sample', backfill_gpu_sample),
    ('metrics_scrape', backfill_metrics_scrape),
    ('run_load_info', backfill_run_load_info),
    ('request', backfill_request),
    ('result', backfill_result),
    ('answer', backfill_answer),
    ('suite_exclusion', backfill_suite_exclusion),
    ('schema_note', backfill_schema_note),
    ('tune_sweep', backfill_tune_sweep),
    ('tune_candidate', backfill_tune_candidate),
    ('tune_round', backfill_tune_round),
    ('tune_visit', backfill_tune_visit),
    ('tune_pause', backfill_tune_pause),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sqlite-path', required=True, type=Path, help='path to the source logs/llama.db')
    parser.add_argument('--database-url', default=None, help='defaults to the DATABASE_URL env var')
    parser.add_argument('--force', action='store_true', help='run again even if a sentinel note is already present')
    args = parser.parse_args()

    if not args.sqlite_path.is_file():
        raise SystemExit(f'no sqlite file at {args.sqlite_path}')

    sconn = sqlite3.connect(str(args.sqlite_path))
    sconn.row_factory = sqlite3.Row
    pconn = connect(args.database_url)

    existing = _already_backfilled(pconn)
    if existing and not args.force:
        raise SystemExit(f'already backfilled: "{existing}" - pass --force to run again')

    counts: dict[str, int] = {}
    for name, fn in STEPS:
        counts[name] = fn(sconn, pconn)
        print(f'{name}: {counts[name]} row(s)')

    note = (
        f'{SENTINEL_PREFIX}{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} '
        f'(source: {args.sqlite_path}, counts: {json.dumps(counts)})'
    )
    pconn.execute(
        "INSERT INTO benchmark_schema_note (noted_on, note) VALUES (%s, %s) ON CONFLICT (note) DO NOTHING",
        (time.strftime('%Y-%m-%d', time.gmtime()), note),
    )
    print('done:', note)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
