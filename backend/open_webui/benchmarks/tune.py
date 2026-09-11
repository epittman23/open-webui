"""benchmarks/tune.py - search the serving configuration space, and judge the winner.

Ported from local-llm's scripts/llama_tune.py (outer repo, 2265 lines -- the
`lllm-tune` configuration-search / A-B testing engine, and the highest-risk
port in this migration). `benchmarks/compare.py` and `benchmarks/report.py`
analyse configurations somebody already thought to try. This searches: it
serves a batch of configurations against the same benchmark items,
eliminates the slow ones round by round, narrows onto the best values of
each knob, and then asks whether the winner answers any worse than the
profile's own defaults.

What it will and will not claim, because this is the whole design (preserved
verbatim from the original):

  * It ranks on **generation throughput**, paired per item against the
    baseline. It never ranks on pass rate: at the discordance this store
    measures, a paired correctness test needs several hundred items to see a
    5-point difference, so a per-round correctness ranking at 6 or 8 items
    would be ranking noise.
  * The winner then has to survive **one** pre-registered correctness check
    against the incumbent, at the end, over every item both ran (`Sweep.guard()`).
    The output of that check is "no regression was detectable, and here is
    the smallest one this comparison could have found". The word "better" is
    never used, because the design cannot support it.
  * When the design cannot support the throughput claim either -- levels
    that never overlap in time, a machine state that changed underneath the
    sweep -- the verdict is `indeterminate` and the ranking is printed
    labelled as refused. That refusal is `benchmarks.report`'s
    `audit_design()`, reused rather than re-derived (`Sweep.audit()`).

Three structural choices, each of them a lesson from this project's own
history, preserved exactly:

  * **Candidates are interleaved, not run one at a time to completion.**
    Generation throughput has been observed to collapse mid-run and never
    recover; a sweep that measured candidate A for an hour and then
    candidate B would report a power cap as a configuration effect. Round-
    robin makes every pair of candidates share a machine state, which is
    what makes the paired ratio mean anything (`Sweep.run_round()`).
  * **The statistic is the median paired log-ratio against the baseline**
    (`tune_schedule.log_ratios()` + `.median()`), which is invariant to a
    slowdown that hits both arms, rather than a mean of absolute t/s, which
    is not.
  * **Nothing here reimplements serving.** A candidate is `llama-serve`
    under `LLAMA_*` overrides launched through the same
    `benchmarks.proc.Command` the rest of this app uses, so it is
    fingerprinted, telemetered and recorded as an ordinary run. This module
    is a caller of scripts/llama-env.sh, never a second copy of it.

Adoption is deliberately manual: `sweep_report()`'s `adoption` field is the
override diff for a human to review, and nothing in this module writes
llama-env.sh. That file is the source of truth for serving configuration in
the outer repo, and a tool that rewrote it would be application code in a
repo that holds none.

WHAT CHANGED IN THIS PORT, AND WHY.

Sequential-per-round candidate execution is preserved on purpose, not
simplified into concurrent candidates despite asyncio making that
mechanically easy. llama-server is a GPU-resident process, and the original
serves exactly one candidate at a time within a round (`Server` owns one
port and one process group; `Sweep.run_round()`'s `for cand in order:` loop
awaits each visit to completion before starting the next) precisely because
this hardware cannot hold two candidates' weights in VRAM at once -- see
`GpuProbe`/`Cooldown` and the whole cooldown design, which exists only
because there is one card to share. Running candidates concurrently would
not just change timing, it would OOM. Nothing about "a round's candidates
are independent trials" in the original implies they may run at the same
time; it means they are scored against each other, which round-robin
*sequencing* (not concurrency) is what makes valid.

Console/CLI machinery is dropped, exactly as benchmarks.runner dropped
llama_test.py's live token rendering: there is no terminal on the other end
of an ASGI request. `con.rule/note/warn/say/table` calls become the module
logger; `report_sweep()`'s printed tables become `sweep_report()`'s
structured dict, matching benchmarks.compare/report's own "no argparse, no
printing, structured data for a router to shape" contract. `cmd_run`/
`cmd_resume`/`cmd_status`/`cmd_list`/`parse_args`/`main` become the
module-level async functions at the bottom of this file
(`start_sweep()`/`resume_sweep()`/`sweep_status()`/`list_recent_sweeps()`/
`plan_sweep()`); a future router calls these, not argparse.

KeyboardInterrupt handling becomes cooperative cancellation, per
benchmarks.runner's own precedent ("there is no Ctrl-C in an ASGI worker;
cooperative cancellation is should_stop()") -- but `finish()` is still
reached afterward, exactly as the original reached it after catching
SIGINT, rather than left uncalled. Two paths lead there now:

  * `Sweep.request_cancel()` sets a flag `Sweep.spent()` observes at exactly
    the points the original checked its time/visit budget, so the round
    loop winds down on its own between visits and falls through to
    `finish()` with no exception involved at all.
  * A real `asyncio.CancelledError` (a task cancellation forced on the
    coroutine from outside, e.g. a backend shutdown) is caught in two
    places: `Sweep.visit()` catches it around one visit to close that visit
    as aborted before re-raising (so the in-flight visit is never left
    'running' in the store), and `execute()`/`resume_sweep()` catch it again
    at the top level -- mirroring the original's own
    `except KeyboardInterrupt: self.ended_reason = "interrupted"` -- to mark
    the sweep interrupted and still call `finish()`, so a forced
    cancellation still produces a resumable `incomplete` verdict rather than
    depending solely on `reclaim_stale_sweeps()` noticing a dead pid later.
    The `CancelledError` is deliberately not re-raised past that point, the
    same way the original's caught `KeyboardInterrupt` was not re-raised: a
    caller that force-cancelled the task already got what it asked for (no
    further visit will start); it is only allowed to run the handful of
    already-fast DB writes `finish()` needs to leave the sweep in a
    resumable state.

Crash recovery (`reclaim_stale_runs()`/`reclaim_stale_sweeps()`) is ported
from scripts/llama_db.py's `sweep_stale_runs()`/`sweep_stale_sweeps()`,
which the original ran as a side effect of every `db.connect(sweep=True)`.
Nothing in the already-ported model layer (open_webui.models.benchmark_tune/
benchmark_configs) does this, so it is ported here instead, called
explicitly by `start_sweep()`/`resume_sweep()` before touching a live sweep
-- deliberately NOT inside `Sweep.create()`/`plan_sweep()`, so a dry-run plan
still writes nothing, unlike the original where dry-run's own construction
silently reclaimed stale rows as a side effect of connecting. `pid`-liveness
(`_alive()`) and the `verdict` state machine (`BenchmarkTuneSweeps.close_sweep()`'s
own "guarded on verdict, not ended_at" semantics) are unchanged: a killed
sweep is detected because its recorded pid is dead, not because a state
file went missing.

Everything DB-touching is `async def` calling the async Table-wrapper
methods on open_webui.models.benchmark_tune / benchmark_configs /
benchmark_tests / benchmark_telemetry. `tune_matrix()` below is the one
piece with no ready-made model-layer method (the original had a dedicated
SQL view, `v_tune_request`); it is built here the same way
benchmarks.compare.serving_summary() and
BenchmarkRuns.get_latest_run_per_config() already handle their own derived
joins -- fetched through existing Table-wrapper query methods and joined in
Python -- per this migration's decision to keep derived views out of the
schema.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import math
import os
import time

import aiohttp

from open_webui.benchmarks import env_profile, report, runner
from open_webui.benchmarks.adapters import load_adapters, render_prompt
from open_webui.benchmarks.runner import SuiteLoadError
from open_webui.benchmarks.suites import build_suite, interleave, load_suite, order_sha as order_sha_of
from open_webui.benchmarks.tune_probe import (
    CollapseWatch,
    GpuProbe,
    Infeasible,
    Server,
    kill_pgid,
    port_busy,
    throttled,
    tps,
)
from open_webui.benchmarks.tune_schedule import (
    Budget,
    Candidate,
    Grid,
    MODES,
    Schedule,
    TuneRefused,
    load_grid,
    log_ratios,
    median,
    plan_schedule,
    refine_candidates,
    sample_candidates,
    sign_test,
)
from open_webui.models.benchmark_configs import BenchmarkRuns
from open_webui.models.benchmark_telemetry import BenchmarkGpuSamples
from open_webui.models.benchmark_tests import BenchmarkResults
from open_webui.models.benchmark_tune import (
    BenchmarkTuneCandidateModel,
    BenchmarkTuneCandidates,
    BenchmarkTunePauses,
    BenchmarkTuneRounds,
    BenchmarkTuneSweeps,
    BenchmarkTuneVisits,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# outcome vocabulary -- reimplemented rather than imported, the same
# modularity choice benchmarks/report.py's own module docstring explains for
# its copy of these six lines: this module must not depend on
# benchmarks.compare or .report for something this small, and duplicating it
# is cheaper than the coupling.
# ---------------------------------------------------------------------------
PASS = 'pass'
FAILURES = {'fail_assert', 'fail_error', 'fail_timeout', 'no_code'}
GRADED = FAILURES | {PASS}


# ---------------------------------------------------------------------------
# crash recovery: pid liveness, not a state file
# ---------------------------------------------------------------------------
def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # someone else's process
        return True
    except OSError:  # pragma: no cover
        return True
    return True


async def reclaim_stale_runs(limit: int = 10_000) -> int:
    """Close serving runs whose recorder is gone.

    Ported from scripts/llama_db.py's sweep_stale_runs(). An open run
    (ended_at IS NULL) is the active-run marker a suite run attaches its
    requests to; the run carries the recorder's pid, so a dead recorder is
    detectable rather than merely likely.

    One known gap versus the original: the original closes the run at its
    *last observation* (MAX(at) over its gpu samples), not at the moment of
    the sweep, so nothing is claimed to have been measured between the
    recorder dying and someone noticing. BenchmarkRuns.close_run() always
    stamps `ended_at` as "now" and takes no override, and that method is
    part of the already-ported model layer this port must call rather than
    modify -- so a stale run closed here is timestamped less precisely than
    the original. `ended_reason='stale'` still marks it as distinct from a
    clean close.

    `limit` works around BenchmarkRuns.list_runs()'s default page size of 20;
    there is no unlimited variant, so a large limit is passed instead of
    adding one to the model layer for this one caller.
    """
    closed = 0
    for row in await BenchmarkRuns.list_runs(limit=limit, active_only=True):
        if row.pid and _alive(row.pid):
            continue
        await BenchmarkRuns.close_run(row.run_id, reason='stale')
        closed += 1
    return closed


async def reclaim_stale_sweeps(limit: int = 10_000) -> int:
    """Close sweeps whose tuner is gone, and the visits/pauses it left open.

    Ported from scripts/llama_db.py's sweep_stale_sweeps(). The sibling of
    reclaim_stale_runs(), by the same test: a sweep carries the tuner's pid,
    so a dead tuner is detectable rather than merely likely. This is the
    whole of the engine's crash recovery -- there is deliberately no state
    file, because a state file can be skipped by a `kill -9` and leave a
    stale marker that later results are filed under, which is exactly the
    failure this project moved away from.

    An open pause closes as 'abandoned' rather than 'recovered': nothing was
    watching the card while the tuner was dead, so whether it recovered is
    unknown, and a resume has to re-probe rather than assume (see
    resume_sweep()'s stale-pause re-probe).
    """
    closed = 0
    for row in await BenchmarkTuneSweeps.list_sweeps(limit=limit):
        if row.ended_at is not None:
            continue
        if row.pid and _alive(row.pid):
            continue
        for visit in await BenchmarkTuneVisits.sweep_visits(row.sweep_id):
            if visit.status == 'running':
                await BenchmarkTuneVisits.close_visit(visit.visit_id, status='aborted', reason='tuner gone')
        for pause in await BenchmarkTunePauses.sweep_pauses(row.sweep_id):
            if pause['resolution'] is None:
                await BenchmarkTunePauses.close_pause(pause['pause_id'], resolution='abandoned')
        await BenchmarkTuneSweeps.close_sweep(row.sweep_id, reason='stale')
        closed += 1
    return closed


# ---------------------------------------------------------------------------
# the paired measurement matrix
# ---------------------------------------------------------------------------
async def tune_matrix(sweep_id: str, *, rnd: int | None = None, counted_only: bool = True) -> dict:
    """(benchmark, item_id, candidate_sha) -> the cell measured for it.

    Ported from scripts/llama_db.py's tune_matrix(), which read a dedicated
    SQL view (`v_tune_request`: tune_visit JOIN tune_candidate JOIN result
    JOIN request, filtered to `result.at` falling inside the visit's own
    [started_at, ended_at] window). There is no equivalent view in this
    app's schema and none is added here, per this migration's "derived views
    stay out of the schema" choice -- so the join happens in Python, over
    rows already fetched through the async Table-wrapper layer.

    gen_tps is read off `BenchmarkResultModel.timings` directly (the same
    dict `benchmarks.runner.run_item()` denormalises onto the result at
    write time) rather than a joined request row, since this port's
    `benchmark_result` already carries what the original had to join
    `request` to get.

    Keyed on the item and the candidate because the whole design is paired:
    every candidate draws the same items in the same order, so a cell
    missing on one side is a pair that cannot be used rather than a zero.

    `counted_only` drops visits a mid-visit pause contaminated. Those rows
    stay in `benchmark_result` -- they are real graded outcomes, and
    Sweep.guard() reads them with counted_only=False for exactly that reason
    -- but their throughput sits either side of a machine-state change, so
    they are excluded from a round's elimination statistic.

    Not filtered at the SQL level by sweep_id: BenchmarkResults.list_results()
    has no suite_run_id filter, so (as in the original's own
    Sweep.sweep_records()) every result is fetched and narrowed here to the
    suite_run_ids this sweep's candidates own. Functionally correct, less
    efficient than a joined query would be as the store grows -- an
    accepted tradeoff matching benchmarks.compare.serving_summary()'s own
    precedent rather than a new gap this port introduces.
    """
    visits = await BenchmarkTuneVisits.sweep_visits(sweep_id, rnd=rnd)
    if not visits:
        return {}
    cand_rows = await BenchmarkTuneCandidates.sweep_candidates(sweep_id)
    by_suite_run = {c.suite_run_id: c.candidate_sha for c in cand_rows}
    runs = set(by_suite_run)

    by_candidate: dict[str, list] = {}
    for v in visits:
        if counted_only and not v.counts_toward_round:
            continue
        by_candidate.setdefault(v.candidate_sha, []).append(v)

    results = await BenchmarkResults.list_results()
    out: dict[tuple[str, str, str], dict] = {}
    for r in sorted((r for r in results if r.suite_run_id in runs), key=lambda r: r.at):
        sha = by_suite_run.get(r.suite_run_id)
        if sha is None:
            continue
        windows = by_candidate.get(sha)
        if not windows:
            continue
        hit = next((v for v in windows if v.started_at <= r.at and (v.ended_at is None or r.at <= v.ended_at)), None)
        if hit is None:
            continue
        # Last write wins (results are visited oldest-first): a re-run after
        # a pause supersedes the attempt the pause contaminated, the same
        # rule the original view's ORDER BY at + dict-overwrite implements.
        out[(r.benchmark, r.item_id, sha)] = {
            'benchmark': r.benchmark,
            'item_id': r.item_id,
            'candidate_sha': sha,
            'gen_tps': tps(r.timings),
            'outcome': r.outcome,
            'round': hit.round,
            'attempt': hit.attempt,
            'at': r.at,
        }
    return out


# ---------------------------------------------------------------------------
# sweep options -- the async port's replacement for the original's argparse
# Namespace / Args shim
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class SweepOptions:
    """Every parameter a sweep needs, mirroring the original's `run` subcommand
    argument defaults 1:1 (see the original's parse_args()). No `--format`,
    `--dry-run` or `--db`: format/printing is gone (sweep_report() always
    returns structured data), dry-run is its own function (plan_sweep()),
    and there is one Postgres database for the whole app, not a per-invocation
    sqlite path.
    """

    profile: str | None = None
    tier: str = 'smoke'
    benchmark: str | None = None
    system: str | None = None
    grid: str | None = None
    budget: str = 'interactive'
    max_visits: int | None = None
    max_hours: float | None = None
    stages: list[str] = dataclasses.field(default_factory=lambda: ['explore', 'refine'])
    candidates: int | None = None
    round_items: int | None = None
    eta: int = 2
    alpha: float = 0.05
    drift_tolerance: float = 0.15
    cliff_ratio: float = 0.5
    on_drift: str = 'cooldown'
    cooldown_poll: float = 20.0
    cooldown_hold: float = 90.0
    cooldown_temp: float | None = None
    cooldown_max: float = 900.0
    cooldown_attempts: int = 4
    cooldown_probe: int = 4
    max_cooldown_share: float = 0.35
    # Reserved, unused: 'marginal' is part of CANDIDATE_STATUSES and resume's
    # own alive-status filter, but no code path -- in the original or here --
    # ever assigns a candidate that status. Kept for parity with the
    # original's own --reject-marginal flag, which had the same property.
    reject_marginal: bool = False
    load_timeout: float = 420.0
    seed: int = 0
    assume_item_seconds: float = 42.0
    assume_load_seconds: float = 120.0
    sweep_id: str | None = None

    def __post_init__(self) -> None:
        if 'explore' not in self.stages:
            raise TuneRefused(
                "llama-tune: stages must include 'explore'; refine descends from the explore "
                'winner and has nothing to start from without it'
            )


# ---------------------------------------------------------------------------
# cooldown: stop, let the card recover, prove it recovered, resume
# ---------------------------------------------------------------------------
class Cooldown:
    """Stop, let the card recover, prove it recovered, resume.

    The design decision worth restating (ported verbatim from the original):
    an idle GPU reports `GpuIdle` and clears its thermal and power bits **by
    construction**, so idle telemetry can never show that recovery will hold
    under load -- it shows only that nothing is currently loading it.
    Waiting for clear bits and then resuming would therefore resume onto a
    still-capped card and file its throughput under whichever candidate came
    next. So recovery is confirmed by re-serving the baseline and
    re-measuring items already measured, against a number already known.

    The loop is bounded, and that bound is not defensive coding: an
    unbounded cooldown would spend an entire overnight budget waiting on a
    card that is capped rather than hot.

    The probe writes no `benchmark_result` rows. It re-asks items whose
    verdicts are already stored, so recording them again would both spend
    budget and collide with the UNIQUE (suite_run_id, benchmark, item_id)
    constraint. It calls benchmarks.runner.ask() directly and keeps only the
    timings.
    """

    def __init__(self, sweep: 'Sweep', opts: SweepOptions):
        self.sweep = sweep
        self.o = opts
        self.gpu = GpuProbe()
        self.total_seconds = 0.0
        self.count = 0

    def budget_exhausted(self) -> bool:
        limit = self.sweep.budget.seconds
        if limit is None:
            return False
        return self.total_seconds > limit * self.o.max_cooldown_share

    async def run(self, *, trigger: str, rnd: int | None, visit_id: int | None, drift_ratio: float | None) -> str:
        """Cool until the baseline probes clean. Returns the resolution word."""
        before = await self.gpu.read()
        pause_id = await BenchmarkTunePauses.open_pause(
            self.sweep.sweep_id,
            trigger=trigger,
            rnd=rnd,
            visit_id=visit_id,
            attempt=self.count + 1,
            drift_ratio=drift_ratio,
            throttle_before=(before or {}).get('reasons'),
            temp_before=(before or {}).get('temp'),
            power_before=(before or {}).get('power'),
        )
        self.count += 1

        # Auto: 8 C under the temperature observed while throttling, floored
        # so a card that idles warm is not waited on forever.
        target = self.o.cooldown_temp
        if target is None:
            target = max(55.0, ((before or {}).get('temp') or 75.0) - 8.0)

        started = time.time()
        window = self.o.cooldown_max
        resolution = 'timeout'
        probe_tps = None
        for attempt in range(1, self.o.cooldown_attempts + 1):
            log.warning(
                'cooling (%s, attempt %s/%s): waiting for the card to settle below %.0fC with the '
                'thermal and power bits clear',
                trigger,
                attempt,
                self.o.cooldown_attempts,
                target,
            )
            await self._wait(target, window)
            probe_tps = await self._probe()
            if probe_tps is not None and self.sweep.baseline_tps:
                ratio = probe_tps / self.sweep.baseline_tps
                log.info(
                    '  probe %.2f t/s against a round-1 baseline of %.2f (%.2fx)',
                    probe_tps,
                    self.sweep.baseline_tps,
                    ratio,
                )
                # One-sided: this probe exists to prove the card will hold up
                # under load, and round-1's own baseline is not a ceiling --
                # it ran after other candidates had already been warming the
                # card, so a fully-cooled probe legitimately beats it. Only a
                # probe that is still slow is a sign the cap has not lifted.
                if ratio >= 1 - self.o.drift_tolerance:
                    resolution = 'recovered'
                    break
            elif probe_tps is not None:
                resolution = 'recovered'
                break
            # Doubling rather than retrying the same window: if the first
            # wait was not enough, waiting the same amount again is a guess
            # that has already been wrong once.
            window *= 2
            self.total_seconds = time.time() - started
            if self.budget_exhausted():
                resolution = 'timeout'
                break

        self.total_seconds += time.time() - started
        after = await self.gpu.read()
        await BenchmarkTunePauses.close_pause(
            pause_id,
            resolution=resolution,
            throttle_after=(after or {}).get('reasons'),
            temp_after=(after or {}).get('temp'),
            power_after=(after or {}).get('power'),
            probe_tps=probe_tps,
        )
        self.sweep.last_pause_at = time.time()
        if resolution == 'recovered':
            log.info('resumed after %.0fs of cooling', time.time() - started)
        else:
            log.warning('gave up cooling after %.0fs; the card did not return to its round-1 throughput', time.time() - started)
        return resolution

    async def _wait(self, target: float, window: float) -> None:
        """Poll until clear-and-cool has held, or the window elapses."""
        deadline = time.time() + window
        held_since = None
        while time.time() < deadline:
            sample = await self.gpu.read()
            if sample is None:
                # No nvidia-smi is not a reason to resume immediately: the
                # probe still gates recovery, so fall back to serving out the
                # window rather than pretending the card is cool.
                await asyncio.sleep(self.o.cooldown_poll)
                continue
            calm = not throttled(sample) and sample['temp'] <= target
            if calm:
                held_since = held_since or time.time()
                if time.time() - held_since >= self.o.cooldown_hold:
                    return
            else:
                held_since = None
            await asyncio.sleep(self.o.cooldown_poll)

    async def _probe(self) -> float | None:
        """Serve the baseline and re-ask a few already-measured items."""
        rows = self.sweep.probe_rows()
        if not rows:
            return None
        server = Server(self.sweep.baseline, self.sweep.port, load_timeout=self.o.load_timeout, session=self.sweep.session)
        try:
            await server.start()
            await server.wait()
            model = await env_profile.served_model(self.sweep.session, self.sweep.port, self.sweep.profile_dict)
            rates = []
            for row in rows:
                prompt = render_prompt(row['adapter'], row['item'])
                ans = await runner.ask(
                    self.sweep.session,
                    prompt,
                    model,
                    self.sweep.profile_dict,
                    self.sweep.port,
                    stream=False,
                    system=self.sweep.system,
                )
                if ans.gone or ans.error:
                    continue
                rate = tps(ans.timings)
                if rate:
                    rates.append(rate)
            return median(rates)
        except Infeasible as exc:
            log.warning('  probe could not serve the baseline: %s', exc.reason)
            return None
        finally:
            await server.stop()


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------

# Keyed by sweep_id rather than held on the Sweep instance alone, so a
# status/log poll (routers/benchmarks/tune.py, which only ever has a
# sweep_id, never the live Sweep object a background task is running) can
# look up "what is this sweep doing right now" without threading the
# instance itself through the module boundary. Sweep.visit() is the only
# writer: it registers an entry for the duration of one candidate's visit
# and removes it in the same `finally` that already stops that visit's
# server, so a sweep that is not between visits (loading a fresh candidate
# is still "in a visit"; between rounds is not) simply has no entry.
_current_visits: dict[str, dict] = {}


def current_visit_progress(sweep_id: str) -> dict | None:
    """Live progress for whichever candidate `sweep_id` is visiting right now.

    A round can sit on an unchanging DB row for minutes at a time -- a
    partially-offloaded MoE model reloading, then generating at single-digit
    tokens/sec -- with nothing to show until the visit closes. This reads
    llama-server's own periodic progress line (Server._pump() in
    tune_probe.py already parses it) rather than waiting for that.
    """
    visit = _current_visits.get(sweep_id)
    if visit is None:
        return None
    server = visit['server']
    return {
        'candidate_sha': visit['candidate_sha'],
        'label': visit['label'],
        'elapsed_seconds': time.time() - visit['started_at'],
        'n_gen': server.n_gen,
        'tokens_per_second': server.tokens_per_second,
    }


class Sweep:
    """One search: candidates, rounds, drift control, guard, verdict.

    Construction is split into `__init__` (no I/O) and the async classmethod
    `create()` (profile resolution, grid/suite loading, schedule planning --
    all of it I/O), the same shape benchmarks.runner uses for
    prepare_suite() versus context(). Use `await Sweep.create(opts, session)`,
    never `Sweep(opts, session)` directly.
    """

    def __init__(self, opts: SweepOptions, session: aiohttp.ClientSession):
        self.o = opts
        self.session = session
        self.sweep_id = opts.sweep_id or time.strftime('tune-%Y%m%dT%H%M%SZ', time.gmtime())

        self.profile_dict: dict = {}
        self.profile_name: str = opts.profile or ''
        self.port: int = 0
        self.profile_json: dict = {}
        self.grid: Grid | None = None
        self.system: dict = {}
        self.adapters: dict = {}
        self.suite: dict = {}
        self.skipped: list = []
        self.items: list[dict] = []
        self.order_sha: str = ''
        self.budget: Budget | None = None
        self.schedule: Schedule | None = None

        self.started = time.time()
        self.visits = 0
        self.baseline: Candidate | None = None
        self.candidates: dict[str, Candidate] = {}
        self.baseline_tps: float | None = None
        self.last_pause_at: float | None = None
        self.cool = Cooldown(self, opts)
        self.load_ms: list[float] = []
        self.serve_seconds = 0.0
        self.ended_reason = 'complete'
        self.cancel_requested = False

    @classmethod
    async def create(cls, opts: SweepOptions, session: aiohttp.ClientSession) -> 'Sweep':
        sweep = cls(opts, session)
        sweep.profile_dict = await env_profile.profile(opts.profile)
        sweep.profile_name = sweep.profile_dict.get('name') or opts.profile or ''
        sweep.port = env_profile.port(sweep.profile_dict)
        # A second, independent call, by the resolved name -- not a reuse of
        # profile_dict above. The grid's constraint evaluator resolves an
        # unset knob to the profile's default, and those defaults must come
        # from exactly the profile this sweep will serve, not from
        # whichever profile answered when opts.profile was possibly None.
        sweep.profile_json = await env_profile.profile(sweep.profile_name)
        sweep.grid = load_grid(opts.grid, sweep.profile_name)
        sweep.system = runner.load_system(opts.system)

        sweep.adapters = load_adapters()
        sweep.suite = _guard(load_suite, opts.tier)
        selected, sweep.skipped = _guard(build_suite, sweep.suite, sweep.adapters, only=opts.benchmark)
        if not selected:
            raise TuneRefused(f"llama-tune: tier '{opts.tier}' selected no items")
        sweep.items = interleave(selected)
        sweep.order_sha = order_sha_of(sweep.items)

        mode = MODES[opts.budget]
        sweep.budget = Budget(
            opts.budget,
            seconds=(round(opts.max_hours * 3600) if opts.max_hours else mode['seconds']),
            visits=(opts.max_visits or mode['visits']),
            candidates=(opts.candidates or mode['candidates']),
            round_items=(opts.round_items or mode['round_items']),
        )
        sweep.schedule = plan_schedule(len(sweep.items), sweep.budget.candidates, sweep.budget.round_items, opts.eta)
        return sweep

    def request_cancel(self) -> None:
        """Ask a running sweep to wind down after its current visit.

        The cooperative replacement for SIGINT: `spent()` observes this flag
        at exactly the points the original checked its time/visit budget, so
        the round loop exits cleanly and still reaches finish() -- no
        exception-based unwind, and no special-casing needed anywhere the
        original checked `self.spent()`.
        """
        self.cancel_requested = True

    # -- setup -------------------------------------------------------------
    def suite_run_id(self, cand: Candidate) -> str:
        """One suite run per (sweep, candidate).

        The highest-value reuse in the whole design: the existing
        UNIQUE (suite_run_id, benchmark, item_id) constraint on
        benchmark_result then makes double-counting across rounds impossible
        *at the schema level* rather than by the round loop being careful,
        and BenchmarkResults.completed() gives crash-safe per-candidate
        resume for free -- the same mechanism a suite-run resume already
        uses.
        """
        return f'{self.sweep_id}-{cand.sha[:6]}'

    async def register(self, cands: list[Candidate]) -> None:
        for c in cands:
            self.candidates[c.sha] = c
            if c.baseline:
                self.baseline = c
        # config_id is deliberately NOT written here, even though
        # config_id_of() has already predicted it (that prediction is what
        # plan_sweep()'s table shows). BenchmarkTuneCandidate.config_id is a
        # foreign key into benchmark_config, and no config row exists until a
        # server under these flags has actually bound and the recorder has
        # upserted one -- a candidate that OOMs on load never gets one at
        # all. The real id is attached in visit() once the recorder confirms
        # it.
        await BenchmarkTuneCandidates.add_candidates(
            self.sweep_id,
            [
                {
                    'candidate_sha': c.sha,
                    'stage': c.stage,
                    'overrides': c.overrides,
                    'is_baseline': c.baseline,
                    'suite_run_id': self.suite_run_id(c),
                }
                for c in cands
            ],
        )

    async def open(self) -> None:
        await BenchmarkTuneSweeps.open_sweep(
            self.sweep_id,
            pid=os.getpid(),
            profile=self.profile_name,
            tier=self.o.tier,
            benchmark=self.o.benchmark,
            system_name=self.system.get('name'),
            system_sha=self.system.get('sha'),
            grid_path=str(self.grid.path),
            grid_sha=self.grid.sha,
            item_order_sha=self.order_sha,
            item_count=len(self.items),
            budget_mode=self.budget.mode,
            budget_seconds=self.budget.seconds,
            budget_visits=self.budget.visits,
            eta=self.o.eta,
            round_items=self.budget.round_items,
            candidates=self.budget.candidates,
            stages=','.join(self.o.stages),
            objective='gen_tps',
            alpha=self.o.alpha,
            on_drift=self.o.on_drift,
            seed=self.o.seed,
        )

    async def guard_port(self) -> None:
        """Refuse to measure a server this sweep did not start.

        A port that already answers is somebody else's model, and running
        items against it would file their throughput under a candidate
        whose flags were never in effect. The one exception is an orphan
        from this sweep's own crashed visit, identified by the pgid the
        visit recorded and killed rather than measured.

        Callers must run reclaim_stale_sweeps() first (start_sweep() and
        resume_sweep() both do): that reclassifies any visit whose tuner pid
        is dead to 'aborted'/'tuner gone' before this method ever sees it.
        Looking only for 'running' would miss every real crash and leave the
        orphaned server listening forever, so both states are treated as the
        same orphan here.
        """
        if not await port_busy(self.port, self.session):
            return
        visits = await BenchmarkTuneVisits.sweep_visits(self.sweep_id)
        orphans = [v for v in visits if v.status == 'running' or (v.status == 'aborted' and v.reason == 'tuner gone')]
        killed_one = False
        for orphan in orphans:
            if orphan.server_pgid:
                log.warning('killing an orphaned server from visit %s (pgid %s)', orphan.visit_id, orphan.server_pgid)
                await kill_pgid(orphan.server_pgid)
                killed_one = True
            if orphan.status == 'running':
                await BenchmarkTuneVisits.close_visit(orphan.visit_id, status='aborted', reason='orphaned by a crash')
        if not killed_one or await port_busy(self.port, self.session):
            raise TuneRefused(
                f'llama-tune: something is already serving on port {self.port} and this sweep did not '
                f'start it. Measuring it would file its throughput under a candidate whose flags were '
                f'never in effect. Stop it, then re-run.'
            )

    # -- one visit -----------------------------------------------------------
    async def rows_for(self, lo: int, hi: int, cand: Candidate) -> list[dict]:
        done = await BenchmarkResults.completed(self.suite_run_id(cand))
        return [r for r in self.items[lo:hi] if (r['benchmark'], r['item_id']) not in done]

    def probe_rows(self) -> list[dict]:
        n = max(1, self.o.cooldown_probe)
        return self.items[:n]

    async def visit(self, cand: Candidate, rnd: int, lo: int, hi: int, attempt: int = 1) -> dict:
        """Serve one candidate and run its slice. Never raises for a bad config.

        Returns a small summary; everything durable is already committed by
        the async Table-wrapper calls below.
        """
        todo = await self.rows_for(lo, hi, cand)
        if not todo:
            return {'status': 'done', 'items': 0, 'drift': None}

        since = (time.time() - self.last_pause_at) if self.last_pause_at else None
        visit_id = await BenchmarkTuneVisits.open_visit(
            self.sweep_id, cand.sha, rnd, attempt=attempt, item_from=lo, item_to=hi, since_pause_seconds=since
        )
        log.info('round %s - %s - items %s:%s%s', rnd, cand.label, lo, hi, f' (attempt {attempt})' if attempt > 1 else '')
        log.info(cand.flags())

        server = Server(cand, self.port, load_timeout=self.o.load_timeout, session=self.session)
        drift: str | None = None
        records: list[dict] = []
        t0 = time.time()
        _current_visits[self.sweep_id] = {
            'candidate_sha': cand.sha,
            'label': cand.label,
            'started_at': t0,
            'server': server,
        }
        try:
            await server.start()
            load_ms = await server.wait()
            self.load_ms.append(load_ms)
            await BenchmarkTuneVisits.set_visit(visit_id, server_pgid=server.pgid(), load_ms=load_ms)

            ctx = await runner.context(
                self.session,
                profile_name=self.profile_name,
                tier=self.suite['id'],
                seed=int(self.suite.get('seed', 0)),
                benches=sorted({r['benchmark'] for r in todo}),
                system_name=self.o.system,
                suite_run_id=self.suite_run_id(cand),
            )
            if cand.predicted_config_id and ctx['config_id'] and ctx['config_id'] != cand.predicted_config_id:
                # Not fatal, but it means the fingerprint the sweep planned
                # with and the one the recorder observed disagree, and every
                # comparison downstream keys on the recorder's.
                log.warning(
                    '  the recorder filed this visit under %s, not the %s llama-config-id predicted',
                    ctx['config_id'],
                    cand.predicted_config_id,
                )
            if ctx['config_id']:
                # Only the confirmed id is ever written: config_id is a
                # foreign key into benchmark_config, and the prediction has
                # no row there until this moment.
                cand.config_id = ctx['config_id']
                await BenchmarkTuneCandidates.set_candidate(self.sweep_id, cand.sha, config_id=cand.config_id)
                await BenchmarkTuneVisits.set_visit(visit_id, run_id=ctx['run_id'], config_id=cand.config_id)
            else:
                await BenchmarkTuneVisits.set_visit(visit_id, run_id=ctx['run_id'])

            watch = CollapseWatch(self.o.cliff_ratio)
            await runner.run_items(
                self.session,
                todo,
                ctx,
                results=records,
                on_record=watch,
                should_stop=lambda: watch.collapsed or self.spent() is not None,
            )
            drift = 'cliff' if watch.collapsed else None
        except Infeasible as exc:
            await BenchmarkTuneVisits.close_visit(
                visit_id, status='infeasible', reason=f'{exc.kind}: {exc.reason}', items_done=0, counts_toward_round=False
            )
            await BenchmarkTuneCandidates.set_candidate(
                self.sweep_id, cand.sha, status='infeasible', status_reason=f'{exc.kind}: {exc.reason}'
            )
            cand.status, cand.reason = 'infeasible', exc.reason
            log.warning('  %s: %s', exc.kind, exc.reason)
            return {'status': 'infeasible', 'items': 0, 'drift': None}
        except runner.ServerGone as exc:
            # The server died under us. The items that landed are committed
            # and the rest were never written, so the candidate is
            # re-visited on the next round rather than eliminated on a
            # truncated slice.
            await BenchmarkTuneVisits.close_visit(
                visit_id, status='aborted', reason=str(exc)[:400], items_done=len(records), counts_toward_round=False
            )
            log.warning('  server gone after %s items', len(records))
            return {'status': 'aborted', 'items': len(records), 'drift': None}
        except asyncio.CancelledError:
            # A real task cancellation (e.g. backend shutdown) mid-visit.
            # Close what ran as aborted, then re-raise so the cancellation
            # keeps propagating -- execute()/resume_sweep() catch it again
            # one level up to mark the sweep interrupted and still reach
            # finish() (see the module docstring's cancellation section).
            await BenchmarkTuneVisits.close_visit(
                visit_id, status='aborted', reason='interrupted', items_done=len(records), counts_toward_round=False
            )
            raise
        finally:
            await server.stop()
            self.serve_seconds += time.time() - t0
            _current_visits.pop(self.sweep_id, None)

        counts = not drift
        await BenchmarkTuneVisits.close_visit(
            visit_id, status='done', items_done=len(records), counts_toward_round=counts, reason='drift' if drift else ''
        )
        if drift:
            log.warning("  throughput collapsed mid-visit; this visit's %s items are excluded from the round", len(records))
        return {'status': 'done', 'items': len(records), 'drift': drift, 'visit_id': visit_id}

    def spent(self) -> str | None:
        if self.cancel_requested:
            return 'cancelled'
        return self.budget.spent(self.started, self.visits)

    def stopped(self) -> bool:
        """Whether the round loop must not start another round.

        A spent budget is one reason. The other is a fall_back() that
        already decided to end the sweep on a drift the card never
        recovered from (on_drift='halt'): it sets ended_reason but, unlike
        the budget, has no counter the loop checks on its own, so a caller
        must ask this after every run_round() rather than only before it.
        """
        return self.spent() is not None or self.ended_reason != 'complete'

    # -- rounds --------------------------------------------------------------
    async def record_rejected(self, rejected: list[tuple[Candidate, str]]) -> None:
        """Candidates the grid ruled out, stored rather than silently dropped.

        A reader asking why a knob value never appears in the results
        deserves an answer from the database, not from re-running the
        sampler.
        """
        if not rejected:
            return
        await BenchmarkTuneCandidates.add_candidates(
            self.sweep_id,
            [
                {
                    'candidate_sha': c.sha,
                    'stage': c.stage,
                    'overrides': c.overrides,
                    'is_baseline': False,
                    'suite_run_id': self.suite_run_id(c),
                    'status': 'infeasible',
                }
                for c, _ in rejected
            ],
        )
        for cand, why in rejected:
            await BenchmarkTuneCandidates.set_candidate(self.sweep_id, cand.sha, status='infeasible', status_reason=why[:400])

    async def scores(self, rnd: int | None) -> dict[str, tuple[float | None, int]]:
        """Median paired log-ratio against the baseline, per candidate."""
        matrix = await tune_matrix(self.sweep_id, rnd=rnd)
        items = sorted({(k[0], k[1]) for k in matrix})
        base = self.baseline.sha
        out: dict[str, tuple[float | None, int]] = {}
        for sha in self.candidates:
            ratios = log_ratios(matrix, sha, base, items)
            out[sha] = (median(ratios), len(ratios))
        return out

    async def absolute_baseline(self, rnd: int) -> float | None:
        matrix = await tune_matrix(self.sweep_id, rnd=rnd)
        rates = [c['gen_tps'] for k, c in matrix.items() if k[2] == self.baseline.sha and c['gen_tps']]
        return median(rates)

    async def run_round(self, rnd: int, stage: str, alive: list[Candidate], lo: int, hi: int) -> list[Candidate]:
        """Visit every survivor once, then decide who survives.

        Idempotent by design, not just by intent: it is called a second
        time for the *same* round number whenever a resume finds this
        round's `benchmark_tune_round` row still open (no `ended_at`),
        which means budget ran out mid-round last time. Re-deriving
        `already` from `benchmark_tune_visit` skips whoever already has a
        done visit here rather than re-visiting them (open_visit() has no
        ON CONFLICT and the UNIQUE (sweep_id, candidate_sha, round, attempt)
        index would raise), and elimination is withheld -- not run on
        whatever partial data exists -- until every survivor has actually
        been visited in this round. Closing early on a thin sample was the
        original bug this design avoids: it let "budget ran out after 2 of
        11 candidates" read as a real "kept 11 of 11" decision, which then
        made the round look finished to every future resume.
        """
        await BenchmarkTuneRounds.open_round(self.sweep_id, rnd, stage=stage, item_from=lo, item_to=hi, survivors=len(alive))
        # A visit only counts as "handled" here if it either landed data
        # that counts toward the round or is permanently infeasible. A
        # drift- or crash-discarded visit (status 'done' with
        # counts_toward_round=False, or 'aborted') still owes this round a
        # real attempt, whether that retry happens later in this same call
        # (the cooldown branch below) or on a resume that re-enters this
        # round after a crash caught it mid-retry.
        visits_here = await BenchmarkTuneVisits.sweep_visits(self.sweep_id, rnd=rnd)
        already = {
            v.candidate_sha for v in visits_here if v.status == 'infeasible' or (v.status == 'done' and v.counts_toward_round)
        }
        # UNIQUE (sweep_id, candidate_sha, round, attempt) means a re-visit
        # after a crash or a drift-discarded attempt must claim the next
        # attempt number, not assume it is the first (a resume) or the
        # second (the cooldown retry below) -- either guess collides with a
        # leftover row from an attempt that never got the chance to count.
        next_attempt: dict[str, int] = {}
        for v in visits_here:
            next_attempt[v.candidate_sha] = max(next_attempt.get(v.candidate_sha, 0), v.attempt) + 1

        # rotate: the first candidate of a round pays whatever a cold page
        # cache costs, and always making the same one pay it would confound
        # candidate with position for the whole sweep.
        order = alive[rnd % len(alive) :] + alive[: rnd % len(alive)]
        order = [c for c in order if c.sha not in already]
        cut_short = False
        for cand in order:
            if self.spent():
                self.ended_reason = self.spent()
                cut_short = True
                break
            attempt = next_attempt.get(cand.sha, 1)
            outcome = await self.visit(cand, rnd, lo, hi, attempt=attempt)
            self.visits += 1
            next_attempt[cand.sha] = attempt + 1
            if outcome['status'] == 'infeasible':
                continue
            if outcome['drift']:
                if self.o.on_drift == 'cooldown':
                    resolution = await self.cool.run(
                        trigger='cliff', rnd=rnd, visit_id=outcome.get('visit_id'), drift_ratio=None
                    )
                    if resolution == 'recovered':
                        # Runs the *remainder* of the slice, not the whole of
                        # it: the contaminated attempt's rows are already in
                        # benchmark_result and UNIQUE (suite_run_id,
                        # benchmark, item_id) forbids rewriting them. They
                        # keep their verdicts, which the correctness guard
                        # still uses, and are excluded from the round's
                        # throughput statistic by counts_toward_round=False.
                        attempt = next_attempt[cand.sha]
                        await self.visit(cand, rnd, lo, hi, attempt=attempt)
                        self.visits += 1
                        next_attempt[cand.sha] = attempt + 1
                    elif not self.fall_back(rnd):
                        cut_short = True
                        break
                # on_drift in ('segment', 'halt') skips cooldown outright and
                # asks fall_back() directly what a cliff means under that
                # policy.
                elif not self.fall_back(rnd):
                    cut_short = True
                    break

        if cut_short:
            # Leave the round open (no close_round) and nobody eliminated: a
            # resume re-enters run_round for this same round number and
            # finishes visiting whoever `already` does not yet name.
            return alive

        alive = [c for c in alive if c.status != 'infeasible']
        base_tps = await self.absolute_baseline(rnd)
        if rnd == 1 and base_tps:
            self.baseline_tps = base_tps
        ratio = base_tps / self.baseline_tps if base_tps and self.baseline_tps else None
        if ratio is not None and abs(1 - ratio) > self.o.drift_tolerance:
            log.warning('round %s: the baseline moved to %.2fx its round-1 throughput', rnd, ratio)
            if self.o.on_drift == 'cooldown':
                resolution = await self.cool.run(trigger='drift', rnd=rnd, visit_id=None, drift_ratio=ratio)
                if resolution != 'recovered':
                    self.fall_back(rnd)
            else:
                # This round already ran to completion under the new regime,
                # so it is fair to close and score it; fall_back() decides
                # only whether a *later* round is allowed to start (halt) or
                # not (segment), via ended_reason.
                self.fall_back(rnd)

        survivors, decision = await self.eliminate(rnd, alive)
        await BenchmarkTuneRounds.close_round(
            self.sweep_id, rnd, baseline_gen_tps=base_tps, drift_ratio=ratio, decision=decision, survivors=len(survivors)
        )
        return survivors

    def fall_back(self, rnd: int) -> bool:
        """What an unrecoverable card means, per budget mode. False ends the sweep.

        `halt` for interactive, because somebody is waiting; `segment` for
        the unattended modes, which keeps measuring and lets audit_design()
        decide afterwards whether the segments can be compared at all.
        Neither pretends the numbers either side of the cap are one
        population.
        """
        mode = self.o.on_drift
        if mode == 'cooldown':
            mode = 'halt' if self.budget.mode == 'interactive' else 'segment'
        if mode == 'halt':
            self.ended_reason = 'drift'
            log.warning('stopping: the card did not recover and this is an interactive budget')
            return False
        log.warning('continuing under a changed machine state; the design audit decides whether these rounds can be compared')
        return True

    async def eliminate(self, rnd: int, alive: list[Candidate]) -> tuple[list[Candidate], str]:
        keep_n = max(2, math.ceil(len(alive) / self.o.eta))
        if len(alive) <= 2:
            return alive, 'kept all (at the floor)'
        scored = await self.scores(rnd)
        matrix = await tune_matrix(self.sweep_id, rnd=rnd)
        items = sorted({(k[0], k[1]) for k in matrix})

        # Too few paired items is not evidence of slowness. A candidate
        # whose visit was cut short by a dead server or a pause is
        # re-visited rather than eliminated on three items.
        thin = [c for c in alive if not c.baseline and scored.get(c.sha, (None, 0))[1] < 3]
        ranked = sorted(
            [c for c in alive if not c.baseline and c not in thin],
            key=lambda c: scored.get(c.sha, (None, 0))[0] or -math.inf,
            reverse=True,
        )
        for c in alive:
            score, n = scored.get(c.sha, (None, 0))
            c.score = score
            detail = f'{math.exp(score) - 1:+7.1%} over {n} paired items' if score is not None else 'no paired items'
            log.info('  %-10s %s%s', c.label, detail, '   [baseline]' if c.baseline else '')
            if score is not None:
                await BenchmarkTuneCandidates.set_candidate(self.sweep_id, c.sha, score=score)

        survivors = [self.baseline] + ranked[: max(1, keep_n - 1)] + thin
        cut = ranked[max(1, keep_n - 1) :]
        decision = f'kept {len(survivors)} of {len(alive)}'

        if cut and ranked[: max(1, keep_n - 1)]:
            lowest = ranked[max(1, keep_n - 1) - 1]
            highest = cut[0]
            paired = [
                a - b
                for a, b in zip(
                    log_ratios(matrix, lowest.sha, self.baseline.sha, items),
                    log_ratios(matrix, highest.sha, self.baseline.sha, items),
                )
            ]
            p, n = sign_test(paired)
            if n and p > 0.20:
                decision = 'uncertain'
                log.warning(
                    '  the cut between %s and %s is not supported (sign test p = %.2f on %s pairs)',
                    lowest.label,
                    highest.label,
                    p,
                    n,
                )
                for c in cut:
                    await BenchmarkTuneCandidates.set_candidate(
                        self.sweep_id,
                        c.sha,
                        status='eliminated',
                        eliminated_round=rnd,
                        status_reason='cut not statistically supported',
                    )
                return survivors, decision

        for c in cut:
            c.status = 'eliminated'
            await BenchmarkTuneCandidates.set_candidate(
                self.sweep_id, c.sha, status='eliminated', eliminated_round=rnd, status_reason=f'slower over round {rnd}'
            )
        return survivors, decision

    # -- the whole search ------------------------------------------------------
    async def execute(self) -> dict:
        """Run a freshly-created sweep end to end. Call reclaim_stale_sweeps()
        first (start_sweep() does)."""
        await self.guard_port()
        await self.open()
        cands, rejected = await sample_candidates(self.grid, self.profile_json, self.budget.candidates, self.o.seed)
        await self.register(cands)
        await self.record_rejected(rejected)

        alive = list(cands)
        rnd = 0
        try:
            for lo, hi in self.schedule.slices:
                if self.spent():
                    self.ended_reason = self.spent()
                    break
                rnd += 1
                alive = await self.run_round(rnd, 'explore', alive, lo, hi)
                if self.ended_reason != 'complete' or not alive:
                    # No survivors: nothing left for a further round to visit
                    # (run_round()'s rotation indexes by len(alive), which is
                    # zero here), and finish() already turns an empty `alive`
                    # into a clean 'rejected' verdict on its own.
                    break

            cursor = self.schedule.slices[-1][1] if self.schedule.slices else 0
            if 'refine' in self.o.stages and not self.stopped():
                rnd, alive, cursor = await self.refine(rnd, alive, cursor)
        except asyncio.CancelledError:
            # Mirrors the original's `except KeyboardInterrupt:` -- caught,
            # not re-raised, so a forced cancellation still reaches finish()
            # and leaves the sweep resumable (see the module docstring).
            self.ended_reason = 'interrupted'
            log.warning('interrupted. resume sweep %s', self.sweep_id)

        return await self.finish(alive)

    async def refine(self, rnd: int, alive: list[Candidate], cursor: int) -> tuple[int, list[Candidate], int]:
        """Coordinate descent around the leader, on items nobody has seen.

        It runs on the *remaining* prefix rather than re-using explore's
        items, because a refine candidate measured today against a baseline
        measured hours ago is the confound this whole module is built to
        avoid. When the tier has no items left, refine is skipped and says
        so -- a stage that silently compared across a gap would be worse
        than a stage that did not run.
        """
        best = await self.best(alive)
        if best is None or best.baseline:
            log.info("refine: nothing beat the profile's own defaults, so there is no neighbourhood to descend into")
            return rnd, alive, cursor
        remaining = len(self.items) - cursor
        if remaining < self.budget.round_items:
            log.warning(
                'refine skipped: %s items left after explore and a round needs %s. The refine stage would '
                'have to re-use explore items, and comparing a candidate measured now against a baseline '
                'measured hours ago is the confound this command exists to avoid. Run a larger tier, or '
                'lower round_items.',
                remaining,
                self.budget.round_items,
            )
            return rnd, alive, cursor

        cands, rejected = await refine_candidates(self.grid, self.profile_json, best, alive)
        await self.register([c for c in cands if c.sha not in self.candidates or self.candidates[c.sha] is c])
        await self.record_rejected(rejected)
        return await self.run_refine_rounds(rnd, cands, cursor)

    async def run_refine_rounds(self, rnd: int, alive: list[Candidate], cursor: int) -> tuple[int, list[Candidate], int]:
        """The refine round loop, split out so resume_sweep() can re-enter it.

        refine() calls this after generating this stage's candidates; a
        sweep resumed mid-refine already has them (loaded from
        benchmark_tune_candidate) and must run more rounds for the *same*
        candidates rather than regenerate a fresh set from a best() computed
        at resume time, which is only defined over whatever `alive` it is
        handed.
        """
        remaining = len(self.items) - cursor
        while alive and remaining >= self.budget.round_items and not self.stopped():
            rnd += 1
            lo = cursor
            hi = min(len(self.items), cursor + self.budget.round_items)
            alive = await self.run_round(rnd, 'refine', alive, lo, hi)
            cursor, remaining = hi, len(self.items) - hi
            if len(alive) <= 2:
                break
            if self.ended_reason != 'complete':
                break
        return rnd, alive, cursor

    async def best(self, alive: list[Candidate]) -> Candidate | None:
        scored = await self.scores(None)
        ranked = sorted(
            [c for c in alive if not c.baseline and scored.get(c.sha, (None, 0))[0] is not None],
            key=lambda c: scored[c.sha][0],
            reverse=True,
        )
        return ranked[0] if ranked else None

    # -- the guard -----------------------------------------------------------
    async def sweep_records(self) -> list[dict]:
        runs = {self.suite_run_id(c) for c in self.candidates.values()}
        results = await BenchmarkResults.list_results()
        return [r.model_dump() for r in results if r.suite_run_id in runs]

    async def audit(self) -> tuple[str, 'report.Audit', list]:
        """The refusal contract, borrowed whole from benchmarks.report.

        A second, more forgiving implementation of "can this design support
        this claim" is exactly the thing that would let a sweep announce a
        winner across a power cap, so the sweep's rows go through the same
        blocks_of()/audit_design() the report module uses, on the config_id
        axis.
        """
        records = await self.sweep_records()
        blocks = report.blocks_of(records, level_factor='config_id')
        run_ids = {r['run_id'] for r in records if r.get('run_id')}
        samples_by_run: dict[int, list[dict]] = {}
        for run_id in run_ids:
            samples = await BenchmarkGpuSamples.list_samples(run_id)
            samples_by_run[run_id] = [s.model_dump() for s in samples]
        regimes = report.Regimes(samples_by_run)
        design, audit = report.audit_design(records, blocks, regimes)
        return design, audit, blocks

    async def guard(self, winner: Candidate) -> dict:
        """Winner vs incumbent, once, at the end. Never the word 'better'."""
        matrix = await tune_matrix(self.sweep_id, counted_only=False)
        items = sorted({(k[0], k[1]) for k in matrix})
        b = c = both = wp = bp = 0
        for key in items:
            w = matrix.get((*key, winner.sha))
            i = matrix.get((*key, self.baseline.sha))
            if not w or not i:
                continue
            if w['outcome'] not in GRADED or i['outcome'] not in GRADED:
                continue
            both += 1
            wpass = w['outcome'] == PASS
            ipass = i['outcome'] == PASS
            wp += wpass
            bp += ipass
            if ipass and not wpass:
                b += 1
            elif wpass and not ipass:
                c += 1
        p, disc = report.mcnemar_exact(b, c)
        psi = disc / both if both else 0.0
        return {
            'n': both,
            'b': b,
            'c': c,
            'p': p,
            'discordant': disc,
            'psi': psi,
            'winner_passed': wp,
            'incumbent_passed': bp,
            'mde': report.detectable_effect(both, psi) if both else None,
            'needed': report.items_needed(psi, 0.05) if psi else None,
            'regressed': (p < self.o.alpha and wp < bp),
        }

    # -- the verdict -----------------------------------------------------------
    async def finish(self, alive: list[Candidate]) -> dict:
        """Decide, record and return a structured report. Three verdicts,
        matching the original's three exit codes (0 adopted/rejected, 1
        incomplete, 3 indeterminate) via the `verdict` field instead."""
        design, audit, blocks = await self.audit()
        blocked = [r for b in blocks for r in audit.blocked(f'throughput:{b.key}')]
        winner = await self.best(alive)
        scored = await self.scores(None)

        if self.ended_reason in ('budget', 'visits', 'interrupted', 'cancelled', 'drift'):
            verdict = 'incomplete'
            reason = {
                'budget': 'the time budget ran out mid-sweep',
                'visits': 'the max-visits cap was reached',
                'interrupted': 'stopped by a task cancellation',
                'cancelled': 'a cancellation was requested',
                'drift': 'gave up on drift recovery before finishing',
            }[self.ended_reason]
        elif blocked:
            verdict, reason = 'indeterminate', '; '.join(blocked)[:400]
        elif winner is None:
            verdict, reason = 'rejected', 'no candidate beat the profile defaults'
        else:
            verdict, reason = 'adopted', ''

        guard = None
        if verdict == 'adopted':
            # Pre-registered, and it walks down the ranking rather than
            # stopping: rejecting the fastest candidate is a reason to look
            # at the next one, not a reason to abandon the sweep.
            ranked = sorted(
                [c for c in alive if not c.baseline and scored.get(c.sha, (None, 0))[0] is not None],
                key=lambda c: scored[c.sha][0],
                reverse=True,
            )
            for cand in ranked:
                guard = await self.guard(cand)
                if not guard['regressed']:
                    winner = cand
                    break
                await BenchmarkTuneCandidates.set_candidate(
                    self.sweep_id,
                    cand.sha,
                    status='rejected',
                    status_reason=f"correctness regression (McNemar p = {guard['p']:.3f})",
                )
                winner = None
            if winner is None:
                verdict, reason = 'rejected', 'every faster candidate showed a correctness regression'

        if winner is not None and verdict == 'adopted':
            await BenchmarkTuneCandidates.set_candidate(self.sweep_id, winner.sha, status='winner')
        await BenchmarkTuneSweeps.close_sweep(
            self.sweep_id,
            reason=self.ended_reason,
            verdict=verdict,
            verdict_reason=reason,
            winner=winner.sha if winner and verdict == 'adopted' else None,
        )

        return await sweep_report(self.sweep_id, design=design, blocked=blocked, guard=guard)


def _guard(fn, *args, **kwargs):
    """Call a helper that may still raise bare SystemExit, and convert it.

    load_suite()/build_suite() (benchmarks.suites) still raise bare
    SystemExit on error, ported faithfully from a CLI tool where that just
    exited the process. Mirrors benchmarks.runner's own `_guard()` exactly
    (duplicated rather than imported: it is eight lines, and importing a
    module-private helper across files would couple two independently
    portable modules for no real gain), converting the caught SystemExit
    into SuiteLoadError -- an ordinary RuntimeError a router can catch and
    turn into an HTTP 4xx/5xx.
    """
    try:
        return fn(*args, **kwargs)
    except SystemExit as exc:
        raise SuiteLoadError(str(exc)) from exc


# ---------------------------------------------------------------------------
# reporting -- structured data for a router to shape, never printed here
# ---------------------------------------------------------------------------
def pct(value: float | None) -> str:
    return '-' if value is None else f'{math.exp(value) - 1:+.1%}'


def adoption_note(winner: BenchmarkTuneCandidateModel) -> dict:
    """What to put in llama-env.sh, and the flag diff, for a human to commit.

    Adoption is not automated on purpose. llama-env.sh (in the outer local-llm
    repo) is that project's source of truth for serving configuration, and a
    tool that rewrote it would be application code in a repo that
    deliberately holds none. Nothing here writes any file; a future router
    hands this dict to a UI for a human to act on.
    """
    overrides = winner.overrides or {}
    return {
        'candidate_sha': winner.candidate_sha,
        'config_id': winner.config_id,
        'overrides': overrides,
        'flags': ' '.join(f'{k}={v}' for k, v in sorted(overrides.items())),
        'reduces_context': 'LLAMA_CTX' in overrides,
        'note': (
            'Adoption is manual: edit scripts/llama-env.sh (outer repo) by hand and log the decision. '
            'Nothing in this engine writes that file.'
        ),
    }


async def sweep_report(sweep_id: str, *, design: str = '', blocked: list[str] | None = None, guard: dict | None = None) -> dict:
    """Everything operational about one sweep, as structured data.

    Ported from the original's report_sweep(), with printing replaced by a
    dict a router shapes into a response -- the same "no argparse, no
    printing" contract benchmarks.compare/report already keep. The pause
    table and the wall-clock split are included ahead of the ranking for the
    same reason the original printed them first: a sweep that spent 40% of
    an overnight budget cooling is a fact about the hardware a reader should
    see before forming a conclusion from the ranking.
    """
    row = await BenchmarkTuneSweeps.get_by_id(sweep_id)
    if not row:
        raise TuneRefused(f"llama-tune: no sweep '{sweep_id}'")
    cands = {c.candidate_sha: c for c in await BenchmarkTuneCandidates.sweep_candidates(sweep_id)}
    rounds = await BenchmarkTuneRounds.sweep_rounds(sweep_id)
    visits = await BenchmarkTuneVisits.sweep_visits(sweep_id)
    pauses = await BenchmarkTunePauses.sweep_pauses(sweep_id)

    serve = sum((v.load_ms or 0) / 1000.0 for v in visits)
    cooled = sum(p['seconds'] or 0 for p in pauses)

    matrix = await tune_matrix(sweep_id)
    items = sorted({(k[0], k[1]) for k in matrix})
    base = next((c for c in cands.values() if c.is_baseline), None)
    ranking = []
    for sha, c in cands.items():
        if c.status == 'infeasible':
            continue
        n = 0
        score = c.score
        if base:
            paired = log_ratios(matrix, sha, base.candidate_sha, items)
            n = len(paired)
            score = median(paired) if paired else score
        ranking.append(
            {
                'candidate_sha': sha,
                'is_baseline': c.is_baseline,
                'stage': c.stage,
                'config_id': c.config_id,
                'score': score,
                'score_pct': pct(score),
                'paired_items': n,
                'status': c.status,
                'status_reason': c.status_reason,
                'overrides': c.overrides,
            }
        )
    ranking.sort(key=lambda r: -(r['score'] if r['score'] is not None else -math.inf))

    not_measured = [{'candidate_sha': sha, 'reason': c.status_reason} for sha, c in cands.items() if c.status == 'infeasible']

    result = {
        'sweep_id': sweep_id,
        'profile': row.profile,
        'tier': row.tier,
        'benchmark': row.benchmark,
        'grid_path': row.grid_path,
        'grid_sha': row.grid_sha,
        'budget_mode': row.budget_mode,
        'seed': row.seed,
        'item_count': row.item_count,
        'item_order_sha': row.item_order_sha,
        'verdict': row.verdict,
        'ended_reason': row.ended_reason,
        'verdict_reason': row.verdict_reason,
        'resumable': row.verdict == 'incomplete',
        'time': {
            'reloading_seconds': serve,
            'cooling_seconds': cooled,
            'cooling_share': (cooled / row.budget_seconds) if row.budget_seconds else None,
            'pauses': len(pauses),
        },
        'rounds': [
            {
                'round': r.round,
                'stage': r.stage,
                'item_from': r.item_from,
                'item_to': r.item_to,
                'survivors': r.survivors,
                'baseline_gen_tps': r.baseline_gen_tps,
                'drift_ratio': r.drift_ratio,
                'decision': r.decision,
            }
            for r in rounds
        ],
        'pauses': pauses,
        'candidates': ranking,
        'not_measured': not_measured,
        'design_audit': design or None,
        'blocked': blocked or [],
        'guard': guard,
    }
    if row.verdict == 'adopted' and row.winner_candidate:
        result['adoption'] = adoption_note(cands[row.winner_candidate])
    return result


async def plan_sweep(sweep: Sweep) -> dict:
    """Plan and return the schedule; launch nothing, write nothing.

    Ported from the original's cmd_dry_run(). This is where the tier
    arithmetic and every grid constraint are checked, and it costs seconds
    rather than the hours a real sweep does. Deliberately does not call
    reclaim_stale_sweeps()/reclaim_stale_runs() -- unlike the original,
    whose dry-run silently reclaimed stale rows as a side effect of
    connecting to sqlite; a plan that "writes nothing" should mean it,
    which this port can honour precisely because reclaim is no longer an
    implicit side effect of opening a connection.
    """
    s = sweep.schedule
    cands, rejected = await sample_candidates(sweep.grid, sweep.profile_json, sweep.budget.candidates, sweep.o.seed)
    item_s, load_s = sweep.o.assume_item_seconds, sweep.o.assume_load_seconds
    reloads = sum(s.survivors)
    wall = s.item_visits * item_s + reloads * load_s
    return {
        'profile': sweep.profile_name,
        'tier': sweep.o.tier,
        'grid_path': str(sweep.grid.path),
        'grid_sha': sweep.grid.sha,
        'knob_count': len(sweep.grid.knobs),
        'constraint_count': len(sweep.grid.constraints),
        'item_count': len(sweep.items),
        'item_order_sha': sweep.order_sha,
        'rounds': [
            {'round': i + 1, 'item_from': lo, 'item_to': hi, 'items': hi - lo, 'candidates': n, 'item_visits': n * (hi - lo)}
            for i, (n, (lo, hi)) in enumerate(zip(s.survivors, s.slices))
        ],
        'winner_items': s.winner_items,
        'estimated_hours': (wall / 3600) if wall else 0.0,
        'estimated_reload_share': (reloads * load_s / wall) if wall else None,
        'candidates': [
            {'label': c.label, 'predicted_config_id': c.predicted_config_id, 'overrides': c.overrides} for c in cands
        ],
        'rejected': [{'candidate_sha': c.sha, 'reason': why} for c, why in rejected],
        'candidates_short_of_budget': max(0, sweep.budget.candidates - len(cands)),
    }


async def start_sweep(opts: SweepOptions, session: aiohttp.ClientSession) -> dict:
    """Create and run a brand-new sweep end to end. The async port of `cmd_run`
    (minus --dry-run, which is plan_sweep())."""
    await reclaim_stale_sweeps()
    await reclaim_stale_runs()
    sweep = await Sweep.create(opts, session)
    return await sweep.execute()


async def resume_sweep(sweep_id: str | None, defaults: SweepOptions, session: aiohttp.ClientSession) -> dict:
    """Continue the last (or named) sweep. The async port of `cmd_resume`.

    Only the parameters that define the search space itself are restored
    from the stored sweep row; everything operational (cooldown timing,
    drift tolerance, load timeout, the assume-*-seconds estimates, ...) has
    no column and a resumed sweep runs it at `defaults`' values instead --
    the same stance budget_seconds already takes (resuming does not hand
    back the time already spent).
    """
    await reclaim_stale_sweeps()
    await reclaim_stale_runs()

    if not sweep_id:
        sweep_id = await BenchmarkTuneSweeps.latest_sweep()
    if not sweep_id:
        raise TuneRefused('llama-tune: no sweep to resume')
    row = await BenchmarkTuneSweeps.get_by_id(sweep_id)
    if not row:
        raise TuneRefused(f"llama-tune: no sweep '{sweep_id}'")

    merged = dataclasses.replace(
        defaults,
        profile=row.profile,
        tier=row.tier,
        benchmark=row.benchmark,
        system=row.system_name,
        grid=row.grid_path,
        budget=row.budget_mode,
        max_visits=row.budget_visits,
        max_hours=None,
        eta=row.eta,
        round_items=row.round_items,
        candidates=row.candidates,
        stages=row.stages.split(','),
        alpha=row.alpha,
        on_drift=row.on_drift,
        seed=row.seed,
        sweep_id=sweep_id,
    )
    sweep = await Sweep.create(merged, session)
    # The grid is the search space, and a search space edited mid-sweep
    # makes the finished rounds measurements of a different experiment.
    # Refuse and name both, rather than resuming into a silent
    # discontinuity.
    if sweep.grid.sha != row.grid_sha:
        raise TuneRefused(
            f"llama-tune: {row.grid_path} has changed since this sweep started "
            f'({row.grid_sha} then, {sweep.grid.sha} now). The rounds already run measured the old '
            f'space. Restore the file, or start a new sweep.'
        )
    if sweep.order_sha != row.item_order_sha:
        raise TuneRefused(
            f'llama-tune: the tier no longer resolves to the same items ({row.item_order_sha} then, '
            f'{sweep.order_sha} now). A dataset or a calibration changed underneath this sweep.'
        )
    if row.budget_seconds:
        # Resuming does not hand back the time already spent; the mode's
        # promise is about wall clock, and a resumed multiday sweep is why
        # the budget is None for that mode rather than large.
        sweep.budget.seconds = row.budget_seconds

    sweep.started = time.time()
    for c in await BenchmarkTuneCandidates.sweep_candidates(sweep_id):
        cand = Candidate(row.profile, c.overrides or {}, stage=c.stage, baseline=bool(c.is_baseline))
        cand.config_id = c.config_id
        cand.status = c.status
        sweep.candidates[cand.sha] = cand
        if cand.baseline:
            sweep.baseline = cand
    done = await BenchmarkTuneRounds.sweep_rounds(sweep_id)
    if done and done[0].baseline_gen_tps:
        sweep.baseline_tps = done[0].baseline_gen_tps

    log.info('resuming %s after %s rounds', sweep_id, len(done))
    # Re-probe before serving. A kill -9 during a cooldown leaves an open
    # pause and no evidence about the card: the sweep was waiting on it
    # precisely because it could not be trusted, and nothing was watching in
    # between.
    stale = [p for p in await BenchmarkTunePauses.sweep_pauses(sweep_id) if p['resolution'] == 'abandoned']
    if stale:
        log.warning('%s pause(s) were abandoned by a crash; re-probing before measuring anything', len(stale))
        await sweep.cool.run(trigger='drift', rnd=(len(done) or 1), visit_id=None, drift_ratio=None)

    alive = [c for c in sweep.candidates.values() if c.status in ('pending', 'active', 'marginal', 'winner')]
    await sweep.guard_port()
    # A round row whose own `ended_at` is still NULL means the budget ran
    # out mid-round last time -- run_round() withholds close_round() for
    # exactly that reason, leaving nobody eliminated. rnd/cursor are
    # rewound to that round's own start (its item_from, not item_to) so the
    # loops below re-enter and finish *that* round under its original slice
    # bounds, rather than reading its partial visits as a finished round and
    # moving the schedule on to a slice it never intended.
    unfinished = done[-1] if done and done[-1].ended_at is None else None
    closed = done[:-1] if unfinished else done
    last = unfinished or (closed[-1] if closed else None)
    # A round row's own stage says where the sweep actually was, not the
    # cursor arithmetic below -- and it is what tells this resume whether to
    # continue explore (and possibly fall into refine after) or to re-enter
    # refine directly with the refine candidates already loaded above.
    last_stage = last.stage if last else 'explore'
    cursor = unfinished.item_from if unfinished else (closed[-1].item_to if closed else 0)
    rnd = len(closed)

    try:
        if last_stage == 'explore':
            for i, (lo, hi) in enumerate(sweep.schedule.slices, 1):
                if i <= rnd:
                    continue
                if sweep.spent():
                    sweep.ended_reason = sweep.spent()
                    break
                rnd = i
                alive = await sweep.run_round(rnd, 'explore', alive, lo, hi)
                if sweep.ended_reason != 'complete' or not alive:
                    break
            else:
                cursor = sweep.schedule.slices[-1][1] if sweep.schedule.slices else cursor
            if 'refine' in sweep.o.stages and not sweep.stopped():
                rnd, alive, cursor = await sweep.refine(rnd, alive, cursor)
        else:
            rnd, alive, cursor = await sweep.run_refine_rounds(rnd, alive, cursor)
    except asyncio.CancelledError:
        # Mirrors the original's `except KeyboardInterrupt:` in cmd_resume --
        # caught, not re-raised, so a forced cancellation still reaches
        # finish() and leaves the sweep resumable.
        sweep.ended_reason = 'interrupted'
        log.warning('interrupted. resume sweep %s', sweep_id)

    return await sweep.finish(alive)


async def sweep_status(sweep_id: str | None = None) -> dict:
    """One sweep's tables, without a design audit or guard (those only run
    at finish() time). The async port of `cmd_status`/`cmd_report`."""
    if not sweep_id:
        sweep_id = await BenchmarkTuneSweeps.latest_sweep()
        if not sweep_id:
            raise TuneRefused('llama-tune: no sweeps recorded')
    return await sweep_report(sweep_id)


async def list_recent_sweeps(limit: int = 20) -> list[dict]:
    """Every sweep, newest first. The async port of `cmd_list`."""
    rows = await BenchmarkTuneSweeps.list_sweeps(limit=limit)
    return [
        {
            'sweep_id': r.sweep_id,
            'started_at': r.started_at,
            'profile': r.profile,
            'tier': r.tier,
            'budget_mode': r.budget_mode,
            'verdict': r.verdict or (r.ended_reason or 'running'),
            'winner_candidate': r.winner_candidate,
        }
        for r in rows
    ]
