"""benchmarks/telemetry_recorder.py - record one serving run's GPU telemetry.

Ported from local-llm's scripts/llama_record.py (outer repo). Still started as
a *detached subprocess* for the entire life of one llama-server run -- a
future Serve router spawns it the way scripts/shell/vram-log.sh spawned the
original, deliberately independent of the backend process so a backend
crash or restart does not truncate an in-progress recording. It:

    wait for the port      a model this size takes minutes to load, and samples
                           taken before it is up describe nothing
    open a run row         ended_at IS NULL is the active-run marker a future
                           Serve router looks for; the pid on the row is what
                           makes a dead recorder detectable
    sample nvidia-smi      every LLAMA_VRAM_INTERVAL seconds, committed as taken
    scrape /metrics        every pass, all of them kept, not just first and last
    parse the server log   what llama.cpp said about the model it loaded
    close the run          on SIGTERM from the router, or when the port stops
                           answering, or from the finally block

WHAT CHANGED, AND WHY IT MATTERS. The original's "stdlib only" constraint --
so it could run under bare python3, independent of <repo>/.venv -- is
retired here on purpose: this fork already ships psycopg (v3) for its own
Postgres backend, so the recorder now writes there directly with plain
parameterized SQL against benchmark_run / benchmark_gpu_sample /
benchmark_metrics_scrape / benchmark_run_load_info (and, matching the
original's own division of responsibility -- record() upserts the config and
opens the run itself, nothing upstream does it for it -- benchmark_config
too). It still runs under this app's own interpreter rather than a system
one, and it still has no event loop: a ~5-second poll loop has no need of
one, and SQLAlchemy's async engine would only add a dependency this process
does not need to reason about itself. One psycopg connection, opened for the
life of the process, matches the original's one-sqlite-connection pattern.

The async SQLAlchemy models in open_webui/models/benchmark_configs.py and
benchmark_telemetry.py are the *read* side other backend code (a future
Serve router, a Report page) uses inside the event loop; this file is the
one writer that runs outside it, so both sides read and write the same
tables without disagreeing on their shape.

Kept genuinely dependency-light -- psycopg plus stdlib, no SQLAlchemy, no
FastAPI, no async -- so it stays simple to reason about as a detached child
process, per the reason it is a separate subprocess and not an asyncio task
inside the main backend in the first place.
"""

from __future__ import annotations

import argparse
import errno
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from open_webui.benchmarks import stats

# Consecutive failed port probes before concluding the server is gone. This is
# the fallback stop condition; the normal one is a SIGTERM from the router.
MISS_LIMIT = 3

SMI_FIELDS = ('temperature.gpu,utilization.gpu,memory.used,memory.total,'
              'power.draw,clocks.sm,clocks_throttle_reasons.active')

# The /metrics counters worth a row. llama.cpp exposes more, but these are the
# ones a serving configuration is judged on; anything else would be noise at one
# row per counter per pass.
COUNTERS = [
    'llamacpp:prompt_tokens_total',
    'llamacpp:prompt_tokens_cached_total',
    'llamacpp:prompt_seconds_total',
    'llamacpp:tokens_predicted_total',
    'llamacpp:tokens_predicted_seconds_total',
    'llamacpp:spec_decode_num_draft_tokens_total',
    'llamacpp:spec_decode_num_accepted_tokens_total',
    'llamacpp:spec_decode_num_drafts_total',
]

_stop = False


def _signal(_signum, _frame) -> None:
    global _stop
    _stop = True


def note(message: str) -> None:
    print(f'llama-record: {message}', file=sys.stderr, flush=True)


def now() -> int:
    """Epoch seconds -- the unit every `bigint` timestamp column here is in."""
    return int(time.time())


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------
def port_open(port: int) -> bool:
    """Connect rather than shell out to `ss`.

    Same question as before -- is anything serving on this port -- with one
    fewer external command in a loop that runs every five seconds.
    """
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=1):
            return True
    except OSError:
        return False


def sample_gpu() -> dict | None:
    """One nvidia-smi row. None when the query fails, so a transient failure
    skips a sample rather than writing a row of nulls that would drag every
    average toward zero."""
    try:
        out = subprocess.run(
            ['nvidia-smi', f'--query-gpu={SMI_FIELDS}',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    fields = [f.strip() for f in out.stdout.strip().split('\n')[0].split(',')]
    if len(fields) < 7:
        return None

    def as_int(value: str):
        try:
            return int(float(value))
        except ValueError:
            return None                    # "[N/A]" on a laptop without a sensor

    def as_float(value: str):
        try:
            return float(value)
        except ValueError:
            return None

    throttle = None
    if re.fullmatch(r'0x[0-9a-fA-F]+', fields[6]):
        throttle = int(fields[6], 16)
    return {'at': now(), 'temp_c': as_int(fields[0]),
            'util_pct': as_int(fields[1]), 'mem_used_mib': as_int(fields[2]),
            'mem_total_mib': as_int(fields[3]), 'power_w': as_float(fields[4]),
            'sm_mhz': as_int(fields[5]), 'throttle': throttle}


def scrape_metrics(port: int) -> dict | None:
    """The server's cumulative counters, or None when /metrics is not answering.

    It only exists if the server was started with --metrics, and it answers
    later than the port opens because the model has to finish loading first,
    so a failure early in a run is expected and is simply retried on the next
    pass.
    """
    try:
        with urllib.request.urlopen(
                f'http://127.0.0.1:{port}/metrics', timeout=5) as response:
            body = response.read().decode('utf-8', 'replace')
    except (urllib.error.URLError, OSError, ValueError):
        return None
    if not body.startswith('#'):
        return None                        # a JSON error means --metrics is off

    out: dict[str, float] = {}
    for line in body.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 2 or parts[0] not in COUNTERS:
            continue
        try:
            out[parts[0].split(':', 1)[1]] = float(parts[1])
        except ValueError:
            continue
    return out or None


# ---------------------------------------------------------------------------
# database -- plain parameterized SQL over benchmark_config / benchmark_run /
# benchmark_gpu_sample / benchmark_metrics_scrape / benchmark_run_load_info,
# the same tables open_webui/models/benchmark_configs.py and
# benchmark_telemetry.py read through the async engine. Column lists here are
# the migration's (b3f8a1d94e70_add_benchmark_tables.py), not guessed at.
# ---------------------------------------------------------------------------
def _database_url(explicit: str | None) -> str:
    url = explicit or os.environ.get('DATABASE_URL')
    if not url:
        raise SystemExit('--database-url or the DATABASE_URL env var must be set')
    # open_webui/env.py's own DATABASE_URL is already a plain libpq URI by the
    # time it leaves that module ('postgres://' normalised to 'postgresql://');
    # only internal/db.py's create_async_engine() appends '+psycopg' for
    # SQLAlchemy's dialect lookup. psycopg.connect() wants the undecorated
    # scheme, so any '+driver' suffix is stripped here rather than assumed away.
    if url.startswith('postgres://'):
        url = 'postgresql://' + url[len('postgres://'):]
    elif url.startswith('postgresql+'):
        url = 'postgresql://' + url.split('://', 1)[1]
    return url


def connect(database_url: str | None) -> psycopg.Connection:
    """One connection for the life of the process, autocommitting every
    statement -- the psycopg equivalent of the original's
    isolation_level=None, so a sample is durable the moment it is taken."""
    return psycopg.connect(_database_url(database_url), autocommit=True, row_factory=dict_row)


def upsert_config(con: psycopg.Connection, config_id: str, alias: str,
                  config_lines: 'list[str] | str') -> None:
    """Record a serving configuration, keyed by its fingerprint.

    config_text is stored verbatim because it is what the fingerprint covers.
    The typed columns are parsed out of that same text rather than passed in
    separately, so no column can disagree with the hash that identifies the
    row. Left alone on conflict: an existing row under the same id is by
    definition the same config.
    """
    if isinstance(config_lines, str):
        config_lines = config_lines.split('\n')
    text = '\n'.join(line.strip() for line in config_lines)
    fields = stats.parse_config_text(text)
    keys = ['arch', 'ngl', 'ctx', 'parallel', 'threads', 'moe', 'override_tensors',
            'speculative', 'spec_draft_n_max', 'cache_k', 'cache_v', 'flash_attn',
            'batch', 'ubatch', 'reasoning_effort', 'samplers']
    columns = ['config_id', 'alias', 'config_text', *keys, 'first_seen']
    values = {'config_id': config_id, 'alias': alias, 'config_text': text,
              'first_seen': now(), **{k: fields.get(k) for k in keys}}
    placeholders = ', '.join(f'%({c})s' for c in columns)
    with con.cursor() as cur:
        cur.execute(
            f'INSERT INTO benchmark_config ({", ".join(columns)}) '
            f'VALUES ({placeholders}) ON CONFLICT (config_id) DO NOTHING',
            values)


def open_run(con: psycopg.Connection, config_id: str, *, model: str, quant: str,
            build: str, port: int, pid: int) -> int:
    with con.cursor() as cur:
        cur.execute(
            'INSERT INTO benchmark_run (config_id, model, quant, build, port, '
            ' pid, started_at) VALUES (%s, %s, %s, %s, %s, %s, %s) '
            'RETURNING run_id',
            (config_id, model, quant, build, int(port), int(pid), now()))
        return int(cur.fetchone()['run_id'])


def close_run(con: psycopg.Connection, run_id: int, reason: str = 'clean') -> None:
    with con.cursor() as cur:
        cur.execute(
            'UPDATE benchmark_run SET ended_at = %s, ended_reason = %s '
            'WHERE run_id = %s AND ended_at IS NULL',
            (now(), reason, run_id))


def add_sample(con: psycopg.Connection, run_id: int, sample: dict) -> None:
    with con.cursor() as cur:
        cur.execute(
            'INSERT INTO benchmark_gpu_sample (run_id, at, temp_c, util_pct, '
            ' mem_used_mib, mem_total_mib, power_w, sm_mhz, throttle) '
            'VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)',
            (run_id, sample['at'], sample.get('temp_c'), sample.get('util_pct'),
             sample.get('mem_used_mib'), sample.get('mem_total_mib'),
             sample.get('power_w'), sample.get('sm_mhz'), sample.get('throttle')))


def add_metrics(con: psycopg.Connection, run_id: int, at: int, values: dict) -> None:
    """One /metrics scrape, stored whole rather than as a start/end pair.

    The previous store kept only the first and last scrape and reported one
    delta, which had a documented defect: llama.cpp updates its prompt
    counters when a prompt is processed but its generation counters when the
    task completes, so a scrape taken as the server stops holds the prompt
    half and not the generation half. Keeping the series means the delta can
    be taken to the last scrape after the final completion instead.
    """
    if not values:
        return
    with con.cursor() as cur:
        cur.executemany(
            'INSERT INTO benchmark_metrics_scrape (run_id, at, counter, value) '
            'VALUES (%s, %s, %s, %s) '
            'ON CONFLICT (run_id, at, counter) DO UPDATE SET value = EXCLUDED.value',
            [(run_id, at, name, float(value)) for name, value in values.items()])


def set_load_info(con: psycopg.Connection, run_id: int, info: dict) -> None:
    """What the server said about the model it loaded, per run.

    Never fingerprinted: llama.cpp resolves the fused Gated Delta Net kernels
    per context at load time by checking the fused node landed on the same
    device as its layer, so two runs with an identical fingerprint can
    execute different operations. Observed, therefore recorded against the
    run -- upserted rather than inserted-once, matching the original's
    INSERT OR REPLACE, because it may be written once mid-run (on the first
    successful /metrics scrape) and again on close.
    """
    keys = ['n_layer', 'n_layer_all', 'layers_gpu', 'layers_total',
            'layers_derived', 'n_slots', 'n_ctx_slot', 'kv_unified',
            'fused_gdn', 'mtp_head', 'buffers', 'cpu_buffer_mib',
            'gpu_buffer_mib', 'unused_tensors', 'unused_prefixes',
            'warnings', 'deprecated']
    values = {k: info.get(k) for k in keys}
    for name in ('buffers', 'warnings', 'deprecated', 'unused_prefixes'):
        if isinstance(values[name], (list, dict)):
            values[name] = Jsonb(values[name])
    columns = ', '.join(keys)
    placeholders = ', '.join(f'%({k})s' for k in keys)
    updates = ', '.join(f'{k} = EXCLUDED.{k}' for k in keys)
    with con.cursor() as cur:
        cur.execute(
            f'INSERT INTO benchmark_run_load_info (run_id, {columns}) '
            f'VALUES (%(run_id)s, {placeholders}) '
            f'ON CONFLICT (run_id) DO UPDATE SET {updates}',
            {'run_id': run_id, **values})


def samples(con: psycopg.Connection, run_id: int) -> list[dict]:
    with con.cursor() as cur:
        cur.execute(
            'SELECT * FROM benchmark_gpu_sample WHERE run_id = %s '
            'ORDER BY at, sample_id', (run_id,))
        return cur.fetchall()


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------
def record(args: argparse.Namespace) -> int:
    con = connect(args.database_url)
    upsert_config(con, args.config_id, args.alias, args.config_line)

    # Wait for the server to bind its port before opening the run, so a run that
    # never came up leaves no row at all rather than an empty one.
    waited = 0
    while not port_open(args.port):
        if _stop:
            note('stopped before the server came up')
            return 0
        if waited >= args.wait:
            note(f'no server on port {args.port} after {args.wait}s; giving up')
            return 0
        time.sleep(1)
        waited += 1

    run_id = open_run(con, args.config_id, model=args.model, quant=args.quant,
                      build=args.build, port=args.port, pid=os.getpid())
    note(f'recording run {run_id} (config {args.config_id}) into Postgres')

    server_log = Path(args.server_log) if args.server_log else None
    load_seen = False
    misses = 0
    try:
        while not _stop:
            # Probe before sampling, so telemetry from after the server exited is
            # never attributed to the run.
            if port_open(args.port):
                misses = 0
            else:
                misses += 1
                if misses >= MISS_LIMIT:
                    break
                time.sleep(1)
                continue

            sample = sample_gpu()
            if sample:
                add_sample(con, run_id, sample)

            counters = scrape_metrics(args.port)
            if counters:
                add_metrics(con, run_id, now(), counters)
                if not load_seen:
                    # /metrics answering means the model finished loading, so the
                    # log now has everything worth reading. Recorded here as well
                    # as at close so a kill -9 does not lose it.
                    load_seen = store_load_info(con, run_id, server_log, args.ngl)

            slept = 0.0
            while slept < args.interval and not _stop:
                time.sleep(min(0.5, args.interval - slept))
                slept += 0.5
    finally:
        finish(con, run_id, server_log, args.ngl)
        con.close()
    return 0


def store_load_info(con: psycopg.Connection, run_id: int, server_log: Path | None,
                    ngl: str | None) -> bool:
    info = stats.parse_server_log(server_log, ngl)
    if not info:
        return False
    set_load_info(con, run_id, info)
    return True


def finish(con: psycopg.Connection, run_id: int, server_log: Path | None,
          ngl: str | None) -> None:
    store_load_info(con, run_id, server_log, ngl)
    close_run(con, run_id, 'clean')

    rows = samples(con, run_id)
    if not rows:
        note(f'run {run_id} recorded no samples')
        return
    figures = stats.gpu_stats(rows)
    headroom = figures.get('vram_headroom_mib')
    note(f"run {run_id} closed: {figures['samples']} samples, "
         f"util active avg {figures['util_active_avg']:.0f}%, "
         f"peak {figures['mem_max']} MiB used"
         + (f', {headroom} MiB free' if headroom is not None else ''))
    warning = stats.headroom_warning(str(run_id), headroom)
    if warning:
        note('warning: ' + warning)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config-id', required=True)
    ap.add_argument('--alias', required=True)
    ap.add_argument('--config-line', action='append', default=[],
                    help='one fingerprinted configuration line; repeat in order')
    ap.add_argument('--model', required=True)
    ap.add_argument('--quant', required=True)
    ap.add_argument('--build', default='unknown')
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--ngl', default=None,
                    help='the requested layer count, used only to derive the '
                         'split when the load log does not report one')
    ap.add_argument('--server-log', default=os.environ.get('LLAMA_SERVER_LOG', ''))
    ap.add_argument('--interval', type=float,
                    default=float(os.environ.get('LLAMA_VRAM_INTERVAL', '5')))
    ap.add_argument('--wait', type=int,
                    default=int(os.environ.get('LLAMA_VRAM_WAIT', '600')))
    ap.add_argument('--database-url', default=None,
                    help='Postgres DSN for this app\'s own database; falls '
                         'back to the DATABASE_URL env var')
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _signal)
    signal.signal(signal.SIGINT, _signal)
    try:
        return record(args)
    except OSError as exc:                              # pragma: no cover
        if exc.errno == errno.EINTR:
            return 0
        raise


if __name__ == '__main__':
    sys.exit(main())
