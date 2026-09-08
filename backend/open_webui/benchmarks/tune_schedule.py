"""benchmarks/tune_schedule.py - budget/round arithmetic and the search space.

Ported from local-llm's scripts/llama_tune.py (outer repo, the `lllm-tune`
configuration-search engine) -- specifically the half of it that is pure
arithmetic plus one shell-out, split out of benchmarks/tune.py (the round-
elimination orchestrator) and benchmarks/tune_probe.py (server lifecycle and
GPU cooldown probing) because it is the one third of the original with almost
no process/DB state of its own:

  * `Budget`/`Schedule`/`plan_schedule()` -- how much machine time a sweep may
    spend and the successive-halving round layout that buys, computed once
    before anything is launched.
  * `Grid`/`Candidate`/`sample_candidates()`/`refine_candidates()` -- the
    declared search space, and the two ways of drawing points from it (a
    broad seeded sample for `explore`, the leader's own neighbourhood for
    `refine`).
  * `median()`/`log_ratios()`/`sign_test()` -- the paired-ratio statistics the
    elimination rule in tune.py's `Sweep.eliminate()` is built on.

`config_id_of()` is the one function here with I/O: it shells out to
scripts/llama-env.sh (via LLAMA_ENV_SH, the same env var benchmarks/proc.py
and benchmarks/env_profile.py already require, for the same reason -- that
script is the outer local-llm repo's own serving-profile layer and stays
outside this app) to ask what config fingerprint a candidate's overrides
would produce, *without serving it*. It has to live here rather than in
env_profile.py because it needs a candidate's full override environment, not
just a profile name.

Nothing here touches the database and nothing here imports FastAPI.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import tomllib

GRID_DIR = Path(__file__).resolve().parent / 'data' / 'tuning'


class TuneRefused(RuntimeError):
    """The engine declined to run at all: a bad grid, an impossible schedule, ...

    Mirrors benchmarks/runner.py's SuiteLoadError -- the ported role of
    llama_tune.py's own `refuse()` / `raise SystemExit(2)` contract. The
    original distinguished exit 2 ("I will not do that, here is why") from
    exit 0 with a verdict and exit 1 with a resume line; there is no process
    exit code here; a future router catches this and returns a 4xx with the
    message, exactly as it does for SuiteLoadError.
    """


def _env_sh() -> Path:
    configured = os.environ.get('LLAMA_ENV_SH')
    if not configured:
        raise RuntimeError(
            'LLAMA_ENV_SH is not set. It must point at the outer local-llm '
            "repo's scripts/llama-env.sh -- the serving-profile shell layer "
            'this app resolves a candidate config fingerprint through.'
        )
    return Path(configured)


# ---------------------------------------------------------------------------
# budgets and the schedule they buy
# ---------------------------------------------------------------------------
class Budget:
    """How much machine time a sweep may spend, and what shape that buys.

    `seconds` is wall clock from the first launch, and **cooling counts
    against it**. `--budget overnight` is a promise that the machine is free
    in the morning; making up cooled time by running into the next day would
    break the one thing the mode exists to say. A sweep that spends its
    budget cooling finishes fewer rounds and says so.
    """

    def __init__(self, mode: str, *, seconds: float | None, visits: int | None, candidates: int, round_items: int):
        self.mode = mode
        self.seconds = seconds
        self.visits = visits
        self.candidates = candidates
        self.round_items = round_items

    def spent(self, started: float, visits: int) -> str | None:
        """Why the sweep must stop now, or None to keep going."""
        if self.seconds is not None and time.time() - started >= self.seconds:
            return 'budget'
        if self.visits is not None and visits >= self.visits:
            return 'visits'
        return None

    def remaining(self, started: float) -> float | None:
        if self.seconds is None:
            return None
        return max(0.0, self.seconds - (time.time() - started))


#: The four modes, with the candidate count and round size each can afford at
#: the measured ~42 s/item (2026-09-05, qwen36) plus reload overhead. They are
#: starting points: candidates/round_items in SweepOptions override any of
#: them, and plan_schedule refuses the combination if the tier cannot carry
#: it.
MODES = {
    'interactive': dict(seconds=2 * 3600, visits=None, candidates=6, round_items=6),
    'overnight': dict(seconds=8 * 3600, visits=None, candidates=16, round_items=8),
    # No deadline: multiday is resumed by hand, so what bounds it is the tier.
    'multiday': dict(seconds=None, visits=None, candidates=32, round_items=8),
    # No clock at all, a hard cap on visits. The mode for a stub run or a
    # deliberately short experiment.
    'trials': dict(seconds=None, visits=24, candidates=4, round_items=6),
}

BUDGET_MODES = tuple(MODES)


class Schedule:
    """The rounds a sweep will run, computed before anything is launched.

    Successive halving with eta=2 and doubling slices: every round costs
    about the same number of item-visits, and the survivor accumulates
    k0 * (2^R - 1) items -- which is the number the correctness guard gets,
    and therefore the number worth printing before a sweep rather than after
    it.
    """

    def __init__(
        self,
        candidates: int,
        round_items: int,
        eta: int,
        rounds: int,
        slices: list[tuple[int, int]],
        survivors: list[int],
    ):
        self.candidates = candidates
        self.round_items = round_items
        self.eta = eta
        self.rounds = rounds
        self.slices = slices
        self.survivors = survivors

    @property
    def winner_items(self) -> int:
        return self.slices[-1][1] if self.slices else 0

    @property
    def item_visits(self) -> int:
        return sum(n * (hi - lo) for n, (lo, hi) in zip(self.survivors, self.slices))

    def rows(self) -> list[list[str]]:
        return [
            [str(i + 1), f'{lo}:{hi}', str(hi - lo), str(n), str(n * (hi - lo))]
            for i, (n, (lo, hi)) in enumerate(zip(self.survivors, self.slices))
        ]


def plan_schedule(tier_size: int, candidates: int, round_items: int, eta: int = 2) -> Schedule:
    """Lay out the rounds, or refuse.

    Refuses rather than truncating. A sweep asked for more rounds than the
    tier has items would otherwise quietly pick a "winner" on three items,
    which is the failure mode this whole module exists to avoid -- and
    `smoke` is 24 items, so it is the common case rather than a corner one.
    """
    if candidates < 2:
        raise TuneRefused('llama-tune: candidates must be at least 2 (the baseline plus something to compare it with)')
    if round_items < 1 or eta < 2:
        raise TuneRefused('llama-tune: round_items must be >= 1 and eta >= 2')

    rounds = max(1, math.ceil(math.log(candidates, eta)))
    slices, survivors = [], []
    lo = 0
    alive = candidates
    for r in range(rounds):
        width = round_items * (eta**r)
        slices.append((lo, lo + width))
        survivors.append(alive)
        lo += width
        alive = max(2, math.ceil(alive / eta))

    if lo > tier_size:
        rows = '\n'.join(
            f'  round {i + 1}: items {a}:{b} ({b - a}), {n} candidates'
            for i, (n, (a, b)) in enumerate(zip(survivors, slices))
        )
        raise TuneRefused(
            f'llama-tune: this tier cannot carry that search.\n'
            f'  {candidates} candidates at eta {eta} needs {rounds} rounds;\n'
            f'  rounds of {round_items} items doubling need {lo} items, and\n'
            f'  the tier has {tier_size}.\n{rows}\n'
            f'  Either lower candidates, lower round_items, or run a larger tier. '
            f'Truncating instead would pick a winner on a handful of items.'
        )
    return Schedule(candidates, round_items, eta, rounds, slices, survivors)


# ---------------------------------------------------------------------------
# the search space
# ---------------------------------------------------------------------------
COMPARISONS = [
    ('<=', lambda a, b: a <= b),
    ('>=', lambda a, b: a >= b),
    ('==', lambda a, b: a == b),
    ('!=', lambda a, b: a != b),
    ('<', lambda a, b: a < b),
    ('>', lambda a, b: a > b),
]


def profile_key(var: str) -> str:
    """LLAMA_UBATCH -> ubatch, the key llama-profile-json reports it under.

    The same transform the original's web routes used to decide whether a
    Serve form field is an override. Spelling it once, here, is why the
    profile-json keys had to be exactly this and not something more
    readable.
    """
    return var[len('LLAMA_'):].lower() if var.startswith('LLAMA_') else var.lower()


def _numeric(value):
    """int/float where the text is one, the text itself otherwise."""
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


class Grid:
    """A declared search space: knob values, constraints, and its fingerprint.

    A grid declares *overrides only* and carries no base values. Every base
    value comes from llama-env.sh, which stays the single source of truth; a
    grid that could set a baseline would be a second profile table.
    """

    def __init__(self, path: Path, data: dict):
        self.path = path
        self.data = data
        self.id = str(data.get('id') or path.stem)
        self.profile = str(data.get('profile') or self.id)
        self.knobs: dict[str, list] = {}
        self.refine_step: dict[str, object] = {}
        self.ordered: dict[str, bool] = {}
        for var, spec in (data.get('knob') or {}).items():
            if not var.startswith('LLAMA_'):
                raise TuneRefused(
                    f"llama-tune: {path}: knob '{var}' is not an LLAMA_* override. A grid may only "
                    f'name the documented override variables; anything else would not reach llama-serve.'
                )
            values = spec.get('values')
            if not isinstance(values, list) or not values:
                raise TuneRefused(f"llama-tune: {path}: knob '{var}' has no values")
            self.knobs[var] = list(values)
            self.refine_step[var] = spec.get('refine_step', 1)
            self.ordered[var] = bool(spec.get('ordered', True))
        if not self.knobs:
            raise TuneRefused(f'llama-tune: {path} declares no knobs')
        self.constraints = [str(c['expr']) for c in (data.get('constraint') or []) if c.get('expr')]
        self.scalars = {k: v for k, v in data.items() if isinstance(v, (int, float, str))}
        # Over the parsed structure, not the file's bytes: a grid carries the
        # prose explaining why it is shaped as it is, and hashing that would
        # file a comment edit as a change of search space. Same choice, for
        # the same reason, as adapters.adapter_sha().
        payload = json.dumps(
            {
                'id': self.id,
                'profile': self.profile,
                'knobs': {k: self.knobs[k] for k in sorted(self.knobs)},
                'constraints': sorted(self.constraints),
                'scalars': {k: self.scalars[k] for k in sorted(self.scalars)},
            },
            sort_keys=True,
        )
        self.sha = hashlib.sha1(payload.encode()).hexdigest()[:12]

    # -- constraints -------------------------------------------------------
    def _resolve(self, token: str, overrides: dict, defaults: dict):
        token = token.strip()
        if token in overrides:
            return _numeric(overrides[token])
        if token in self.scalars:
            return _numeric(self.scalars[token])
        key = profile_key(token)
        if token.startswith('LLAMA_') and key in defaults:
            return _numeric(defaults[key])
        return _numeric(token)

    def violated(self, overrides: dict, defaults: dict) -> str | None:
        """The first constraint this candidate breaks, in the grid's own words.

        Deliberately a small comparison parser rather than eval(): a grid is
        a file on disk, and this process runs benchmarks that already
        execute model-generated code under process isolation. Nothing here
        needs to also be an expression evaluator.
        """
        for expr in self.constraints:
            for symbol, test in COMPARISONS:
                if symbol not in expr:
                    continue
                left, right = expr.split(symbol, 1)
                a = self._resolve(left, overrides, defaults)
                b = self._resolve(right, overrides, defaults)
                if a is None or b is None:
                    break
                try:
                    if not test(a, b):
                        return expr
                except TypeError:
                    break
                break
            else:
                raise TuneRefused(f"llama-tune: {self.path}: constraint '{expr}' has no comparison operator")
        return None


def load_grid(path_or_name: str | None, profile: str) -> Grid:
    if path_or_name:
        path = Path(path_or_name)
        if not path.is_file():
            path = GRID_DIR / f'{path_or_name}.toml'
    else:
        path = GRID_DIR / f'{profile}.toml'
    if not path.is_file():
        known = ', '.join(sorted(p.stem for p in GRID_DIR.glob('*.toml'))) or 'none'
        raise TuneRefused(f'llama-tune: no search space at {path} (defined: {known})')
    with path.open('rb') as fh:
        data = tomllib.load(fh)
    return Grid(path, data)


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------
class Candidate:
    """One serving configuration under test.

    Identified by `candidate_sha` over its overrides, **not** by config_id: a
    candidate that fails to load never reaches BenchmarkConfigs.upsert() and
    so never has a config_id at all, and a sweep has to be able to record
    that it tried it.

    Two config_id fields, deliberately not one. `predicted_config_id` comes
    from `config_id_of()` before launch and is what dedup and the dry-run
    table use -- it costs nothing and needs nothing running. `config_id` is
    filled in only from the recorder's own observation once a server has
    actually bound under these flags, because `BenchmarkTuneCandidate.config_id`
    and `BenchmarkTuneVisit.config_id` are foreign keys into
    `benchmark_config`, and no row exists there until BenchmarkConfigs.upsert()
    runs -- a candidate that OOMs never gets one at all. Writing the
    prediction into either column would violate the constraint on a
    candidate that was only ever planned, and silently misattribute rows on
    the rarer candidate that predicts one id and is recorded under another
    (tune.Sweep.visit() warns on that mismatch rather than hiding it behind
    a `config_id or predicted_config_id` fallback).
    """

    def __init__(self, profile: str, overrides: dict, *, stage: str, baseline: bool = False):
        self.profile = profile
        self.overrides = {k: str(v) for k, v in sorted(overrides.items())}
        self.stage = stage
        self.baseline = baseline
        self.predicted_config_id: str | None = None
        self.config_id: str | None = None
        # The full config_id_of() answer, kept beside the id it was drawn
        # from -- tune.py's telemetry-recorder spawn (Recorder, in
        # tune_probe.py) wants the 'config_lines' llama-config-id already
        # computed to produce predicted_config_id, and re-shelling out for
        # them a second time per visit would double the per-candidate
        # subprocess cost for data this object already has in hand.
        self.config_json: dict | None = None
        self.status = 'pending'
        self.reason = ''
        self.score: float | None = None
        payload = json.dumps({'profile': profile, 'overrides': self.overrides}, sort_keys=True)
        self.sha = hashlib.sha1(payload.encode()).hexdigest()[:12]

    @property
    def label(self) -> str:
        return 'baseline' if self.baseline else self.sha[:8]

    def flags(self) -> str:
        if not self.overrides:
            return '(profile defaults)'
        return ' '.join(f'{k}={v}' for k, v in self.overrides.items())

    def env(self) -> dict:
        env = dict(os.environ)
        env.update(self.overrides)
        return env


async def config_id_of(candidate: Candidate) -> dict:
    """The fingerprint this candidate would be recorded under, without serving.

    Asks scripts/llama-env.sh's `config-id` subcommand, which asks the same
    fingerprinting function the telemetry recorder uses (see
    benchmarks/telemetry_recorder.py's upsert_config() / benchmarks/stats.py's
    parse_config_text() -- both read the same six lines this shells out to
    produce). Computing it in Python instead would put a second copy of the
    fingerprint beside the shell one, and the day they disagreed two
    different configurations would be filed under one id.
    """
    proc = await asyncio.create_subprocess_exec(
        str(_env_sh()),
        'config-id',
        candidate.profile,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=candidate.env(),
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise TuneRefused(f'llama-tune: llama-config-id timed out for {candidate.flags()}')
    if proc.returncode != 0:
        raise TuneRefused(
            f'llama-tune: llama-config-id failed for {candidate.flags()}: '
            f"{stderr.decode(errors='replace').strip()[:300]}"
        )
    return json.loads(stdout)


async def sample_candidates(
    grid: Grid, profile_json: dict, want: int, seed: int
) -> tuple[list[Candidate], list[tuple[Candidate, str]]]:
    """The explore stage: a broad, deduplicated draw over the whole grid.

    Random over the cross product rather than one-knob-at-a-time, because the
    knobs interact -- ubatch against batch, threads against how much of the
    model is CPU-resident -- and a one-at-a-time screen would miss exactly
    the combinations worth finding. Narrowing is the refine stage's job, and
    it is coordinate descent precisely because by then there is a point to
    descend from.

    Deduplication is on config_id, not on the override dict: setting a knob
    to the value the profile already uses produces a different dict and the
    same server, and measuring it twice would spend a round learning
    nothing.
    """
    rng = random.Random(seed)
    baseline = Candidate(grid.profile, {}, stage='explore', baseline=True)
    chosen = [baseline]
    rejected: list[tuple[Candidate, str]] = []
    seen_config = set()
    seen_sha = {baseline.sha}

    baseline.predicted_config_id = (await config_id_of(baseline))['config_id']
    seen_config.add(baseline.predicted_config_id)

    knobs = sorted(grid.knobs)
    # Bounded rather than while-True: a heavily constrained grid can have far
    # fewer legal points than the caller asked for, and spinning forever
    # looking for them would be worse than saying so.
    for _ in range(want * 200):
        if len(chosen) >= want:
            break
        overrides = {k: rng.choice(grid.knobs[k]) for k in knobs}
        cand = Candidate(grid.profile, overrides, stage='explore')
        if cand.sha in seen_sha:
            continue
        seen_sha.add(cand.sha)
        why = infeasible_reason(cand, grid, profile_json)
        if why:
            rejected.append((cand, why))
            continue
        cand.predicted_config_id = (await config_id_of(cand))['config_id']
        if cand.predicted_config_id in seen_config:
            rejected.append((cand, f'duplicate of config {cand.predicted_config_id}'))
            continue
        seen_config.add(cand.predicted_config_id)
        chosen.append(cand)
    return chosen, rejected


async def refine_candidates(
    grid: Grid, profile_json: dict, winner: Candidate, keep: list[Candidate]
) -> tuple[list[Candidate], list[tuple[Candidate, str]]]:
    """The refine stage: the winner's immediate neighbourhood, knob by knob.

    This is the half of "start broad and narrow" that halving cannot do on
    its own: successive halving narrows the candidate *set* and never the
    parameter *values*, so without a stage that generates new points near
    the leader, the search stops at whatever the first draw happened to
    contain.
    """
    out: list[Candidate] = []
    rejected: list[tuple[Candidate, str]] = []
    seen_config = set()
    seen_sha = set()
    for cand in keep:
        seen_sha.add(cand.sha)
        if cand.config_id or cand.predicted_config_id:
            seen_config.add(cand.config_id or cand.predicted_config_id)
        out.append(cand)

    base = dict(winner.overrides)
    for var in sorted(grid.knobs):
        values = grid.knobs[var]
        current = base.get(var, profile_json.get(profile_key(var)))
        neighbours = []
        if grid.ordered.get(var, True):
            index = next((i for i, v in enumerate(values) if str(v) == str(current)), None)
            if index is None:
                neighbours = list(values)
            else:
                for j in (index - 1, index + 1):
                    if 0 <= j < len(values):
                        neighbours.append(values[j])
        else:
            # An unordered knob (a cache type) has no adjacency, so the
            # neighbourhood is every other value it can take.
            neighbours = [v for v in values if str(v) != str(current)]

        for value in neighbours:
            overrides = dict(base)
            overrides[var] = value
            cand = Candidate(grid.profile, overrides, stage='refine')
            if cand.sha in seen_sha:
                continue
            seen_sha.add(cand.sha)
            why = infeasible_reason(cand, grid, profile_json)
            if why:
                rejected.append((cand, why))
                continue
            cand.predicted_config_id = (await config_id_of(cand))['config_id']
            if cand.predicted_config_id in seen_config:
                rejected.append((cand, f'duplicate of config {cand.predicted_config_id}'))
                continue
            seen_config.add(cand.predicted_config_id)
            out.append(cand)
    return out, rejected


def infeasible_reason(cand: Candidate, grid: Grid, profile_json: dict) -> str | None:
    """Why this candidate cannot be measured, before anything is launched.

    Every check here is one that would otherwise cost a full server load to
    discover, and two of them would not announce themselves at all:
    llama-env.sh silently blanks --n-cpu-moe for a dense profile (so two
    candidates would collapse onto one configuration and be measured as if
    they were two), and a --parallel inside LLAMA_SPEC is passed twice and
    recorded wrong.
    """
    if profile_json.get('arch') == 'dense' and cand.overrides.get('LLAMA_MOE'):
        return 'dense profile: --n-cpu-moe is dropped by llama-env.sh'
    spec = cand.overrides.get('LLAMA_SPEC', '')
    if '--parallel' in spec:
        return '--parallel inside LLAMA_SPEC is passed twice; use LLAMA_PARALLEL'
    broken = grid.violated(cand.overrides, profile_json)
    if broken:
        return f'constraint: {broken}'
    return None


# ---------------------------------------------------------------------------
# statistics: the elimination rule
# ---------------------------------------------------------------------------
def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def log_ratios(matrix: dict, candidate: str, baseline: str, items: list[tuple[str, str]]) -> list[float]:
    """ln(candidate t/s / baseline t/s), per item both of them served.

    A ratio rather than a difference, and a median rather than a mean, for
    two separate reasons. The ratio is invariant to a machine-wide slowdown
    that hits both arms -- which is what a thermal cap is, and what made a
    naive throughput comparison unanalysable when a power cap struck
    mid-sweep. The median survives one pathological item, which on DS-1000
    is a real prospect.
    """
    out = []
    for key in items:
        a = matrix.get((*key, candidate))
        b = matrix.get((*key, baseline))
        if not a or not b:
            continue
        ta, tb = a.get('gen_tps'), b.get('gen_tps')
        if ta and tb and ta > 0 and tb > 0:
            out.append(math.log(ta / tb))
    return out


def sign_test(values: list[float]) -> tuple[float, int]:
    """Two-sided exact sign test on paired differences. Returns (p, n non-zero).

    Used before a cut, not to justify it: if the two candidates either side
    of the line cannot be told apart, the cut is recorded as unsupported
    rather than presented as a finding. Elimination on throughput alone is
    not unconditionally safe, and the honest thing is to say which cuts were
    guesses.
    """
    nonzero = [v for v in values if v != 0]
    n = len(nonzero)
    if n == 0:
        return 1.0, 0
    k = sum(1 for v in nonzero if v > 0)
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2 * tail / (2**n)), n
