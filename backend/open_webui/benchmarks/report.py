"""benchmarks/report.py - a statistical report over the measurement store.

Ported from local-llm's scripts/llama_report.py (outer repo, 2226 lines --
the largest and most intricate module in this migration). `compare.py`
ranks (model, config, tier, system, adapter) groups by pass rate and by
passes per minute. It is a *ranking*, and a ranking has no way to say
whether the difference it shows is real. This builds the document that
can: it audits the design first, then runs the tests that design supports,
and says what it refuses to test and why.

Three things in the store are the reason it exists, and each one shaped a
section here:

  * The system-prompt ablation is a **paired** design and a flat table shows
    it as unrelated rows. Tiers are seeded so the same items are drawn for
    every configuration, so a level-to-level comparison is within-item. The
    correct tests for a binary outcome measured repeatedly on the same items
    are Cochran's Q and exact McNemar, not ANOVA and not a chi-square over
    the marginal counts -- those throw away the pairing, which at this n is
    most of the information there is.

  * Most items are constant across every level. A section that reported a
    p-value without saying that would be reporting the wrong n.

  * The throughput data can contain a confound a one-way ANOVA walks
    straight into: levels run sequentially, one suite per level, and a GPU
    that hits a software power cap partway through leaves the cap perfectly
    confounded with a system prompt in that block. The naive test is
    computed anyway, printed beside the stratified one as the wrong answer,
    because a reader who does not see it will run it themselves.

Every test carries its assumption check and its n. A test whose assumptions
fail is printed as refused, with the reason, and never silently omitted.

Read-only by design: every fetch below goes through the async Table-wrapper
query methods on open_webui.models.benchmark_tests/benchmark_telemetry/
benchmark_configs (list_results, list_samples, get_by_id, list_runs, ...),
none of which write. This module itself never calls a write method, so the
"a report can never be the thing it reports on" property the original
enforced by opening its own sqlite connection with `mode=ro` holds here
automatically, by construction rather than by a connection flag.

Structural change from the original: the CLI's `at` columns were sqlite TEXT
timestamps ("YYYY-MM-DDTHH:MM:SSZ"), sortable and parseable as strings. This
port's schema stores `at` as BigInteger unix seconds (see benchmark_tests.py,
benchmark_telemetry.py), so every place the original parsed or lexically
sorted a timestamp string here compares or formats an int instead (`_iso()`
is the formatter, kept for markdown display only -- nothing parses it back).

Hard scipy dependency, same as the original: binomtest, chi2 and norm have
no adequate replacement without hand-rolling a special function, and a
report whose p-values come from a hand-rolled incomplete beta is worth less
than no report. Unlike the CLI, which called `raise SystemExit(2)` at import
time on a missing scipy, this module must stay importable even when scipy is
absent -- a router imports it at FastAPI startup to register endpoints, and
crashing the whole process because one optional analysis feature is missing
would take the rest of the backend down with it. So the import is attempted
once at module load, `HAVE_SCIPY` records the outcome, and `build_report()`
-- the only function that actually needs scipy -- raises `SciPyUnavailable`
as its first act when the import failed. A router catches that and returns
503; nothing before that point in `build_report()` has touched the database.

matplotlib is soft, exactly as in the original: report_figures.py degrades
every figure to a unicode text plot when it is not installed, so no analysis
is ever silently hidden behind a missing wheel. That module is imported
lazily, inside `build_report()`, not at the top of this one -- report_figures
imports the labelling and statistics helpers defined below from this module,
and importing it back at module scope here would be a real import cycle
during module load; deferring it until `build_report()` runs (by which time
this module has finished loading) avoids that without either module needing
to guess at the other's load order.

Output contract, preserved from the original: this module hands back one
markdown string, plus the bytes of whatever figures it drew. It does not
also emit a structured-JSON analysis path -- that would be a second, less
audited way to get a number out of the same statistics, which the original
deliberately avoided. The router (not part of this module) decides what to
do with the markdown and the figure bytes, including where -- or whether --
to write them to disk.
"""

from __future__ import annotations

import datetime as dt
import itertools
import math
import random
import textwrap
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from open_webui.benchmarks import stats as lstats
from open_webui.models.benchmark_configs import BenchmarkRuns
from open_webui.models.benchmark_telemetry import BenchmarkGpuSamples
from open_webui.models.benchmark_tests import BenchmarkResults, BenchmarkSchemaNotes

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime import cycle
    from open_webui.benchmarks.report_figures import Figures

# scipy is a hard requirement for this module's actual analyses (see the
# module docstring for why the import failure does not raise here). Checked
# against statsmodels/brute-force per the original's own comments; nothing
# in the statistics below has been re-derived independently of that check.
try:
    from scipy import stats as sps

    HAVE_SCIPY = True
except ImportError:  # pragma: no cover - exercised by uninstalling scipy
    sps = None
    HAVE_SCIPY = False


class SciPyUnavailable(RuntimeError):
    """Raised by build_report() when scipy could not be imported.

    Every statistical test in this module -- Wilson intervals aside, which
    are hand-rolled arithmetic -- calls into scipy.stats (binomtest, chi2,
    norm, kruskal, f_oneway, mannwhitneyu). Rather than hand-rolling those
    too, or letting an AttributeError on `None` surface from deep inside a
    block loop, build_report() checks once, up front, and raises this before
    it fetches a single row. A router can catch it and return a 503 with the
    install hint, instead of the whole app failing to start.
    """


class NoBenchmarkResults(RuntimeError):
    """Raised by build_report() when the tier/model/benchmark filters match nothing.

    The original's CLI wrote "no results match those filters" to stderr and
    exited 1. There is no stderr to write to here, so the same condition is
    raised as an exception a router can turn into a 404.
    """


TS = '%Y-%m-%dT%H:%M:%SZ'
Z95 = 1.959963984540054


def _iso(ts: int | None) -> str:
    """An epoch second as an ISO-8601 UTC string, or '-' when there isn't one.

    Every `at` column in this port is a BigInteger of unix seconds, unlike
    the original sqlite TEXT timestamp -- formatted here purely for display,
    the same way compare.py's `_iso()` does it, so a report reads and a
    reader's eye tracks it exactly like the original's did. Nothing parses
    this back.
    """
    if ts is None:
        return '-'
    return dt.datetime.fromtimestamp(int(ts), tz=dt.timezone.utc).strftime(TS)


def _epoch_dt(ts: int | None) -> dt.datetime | None:
    """An epoch second as an aware UTC datetime, for arithmetic (not display)."""
    if ts is None:
        return None
    return dt.datetime.fromtimestamp(int(ts), tz=dt.timezone.utc)


# ---------------------------------------------------------------------------
# outcome vocabulary -- ported from llama_results.py, which this app has no
# equivalent module for. Reimplemented here the same way compare.py does it,
# rather than imported from a nonexistent dependency or from compare.py
# itself (this module must not depend on compare.py, and duplicating six
# lines is cheaper than an import that couples two independently-portable
# analyses).
# ---------------------------------------------------------------------------
PASS = 'pass'
SKIPPED = 'skipped'
FAILURES = {'fail_assert', 'fail_error', 'fail_timeout', 'no_code'}
GRADED = FAILURES | {PASS}


def graded(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get('outcome') in GRADED]


# ---------------------------------------------------------------------------
# labels and ordering, shared by the figures and the tables beside them
# ---------------------------------------------------------------------------
def natural_key(item_id: str):
    """Sort benchmark item ids the way a reader reads them.

    MBPP names its items with bare integers, and a plain string sort puts 74
    between 641 and 750, which reads as a hole in the sample. Numeric where
    the id is a number, string otherwise, so `HumanEval/9` and
    `HumanEval/117` also land in the order their authors numbered them.
    """
    parts = []
    for chunk in str(item_id).replace('/', ' ').split():
        parts.append((0, int(chunk), '') if chunk.isdigit() else (1, 0, chunk))
    return parts


def short_level(level: str, levels: list[str]) -> str:
    """A level label short enough to draw and still able to name the level.

    A system prompt's identity is the sha of its bytes -- that is the whole
    reason `system_sha` and not `system_name` is in the grouping key -- so a
    label truncated to a fixed width is the one thing this must not do. The
    name alone is unambiguous whenever it is unique within the block, and
    the sha comes back, shortened but whole-prefixed, when it is not.
    """
    name = level.split('@', 1)[0]
    clash = sum(1 for other in levels if other.split('@', 1)[0] == name) > 1
    if not clash or '@' not in level:
        return name
    return f'{name}@{level.split("@", 1)[1][:6]}'


def wrap_title(text: str, width_in: float, fontsize: float = 11.0) -> str:
    """Wrap a figure title to the canvas it is drawn on.

    Figure widths scale with the number of levels, and a two-level block is
    narrow enough that an unwrapped title runs off the right edge and is
    saved that way -- matplotlib clips it without complaining, and
    `tight_layout` does not shrink a title to fit. Characters-per-inch is
    approximate on purpose: erring toward a wrap costs a line of height, and
    erring the other way costs the end of the sentence.
    """
    per_inch = 72.0 / (fontsize * 0.55)
    chars = max(24, int((width_in - 1.8) * per_inch))
    return '\n'.join(textwrap.wrap(text, chars)) or text


# ---------------------------------------------------------------------------
# statistics scipy does not have
# ---------------------------------------------------------------------------
def wilson(passed: int, attempted: int, z: float = Z95) -> tuple[float, float] | None:
    """Wilson score interval for a binomial proportion.

    Not the normal approximation: at n=8 with 8 passes the Wald interval is
    [1.0, 1.0], which claims certainty from the one sample size where there
    is least of it. Wilson stays inside [0,1] and does not collapse at the
    boundary, which is where most of this data sits.
    """
    if attempted <= 0:
        return None
    p = passed / attempted
    d = 1.0 + z * z / attempted
    centre = (p + z * z / (2 * attempted)) / d
    half = z * math.sqrt(p * (1 - p) / attempted + z * z / (4 * attempted * attempted)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def cochran_q(matrix: list[list[int]]) -> tuple[float, int, float, int]:
    """Cochran's Q over an items x treatments matrix of 0/1.

    The k-sample extension of McNemar: the null is that the treatment has no
    effect on the probability of success, tested within items. Returns
    (Q, df, asymptotic p, n_effective).

    Rows that are constant across every treatment contribute nothing to
    either the numerator or the denominator -- an item that passes under all
    six prompts is uninformative about which prompt is better -- so
    `n_effective` is the count of rows that vary, and it is the n this test
    actually has. scipy has no Cochran's Q, hence the hand implementation;
    it is checked against the closed form
    Q = (k-1)(k*sum(Cj^2) - (sum Cj)^2) / (k*sum(Ri) - sum(Ri^2)).
    """
    if not matrix:
        return (0.0, 0, 1.0, 0)
    k = len(matrix[0])
    rows = [r for r in matrix if len(r) == k]
    col = [sum(r[j] for r in rows) for j in range(k)]
    row = [sum(r) for r in rows]
    varying = sum(1 for r in row if 0 < r < k)
    denom = k * sum(row) - sum(r * r for r in row)
    if k < 2 or denom == 0:
        # Every item constant: the statistic is 0/0, which is "no evidence"
        # rather than "no effect", and is reported as such.
        return (0.0, k - 1, 1.0, varying)
    num = (k - 1) * (k * sum(c * c for c in col) - sum(col) ** 2)
    q = num / denom
    return (q, k - 1, float(sps.chi2.sf(q, k - 1)), varying)


def cochran_permutation_p(
    matrix: list[list[int]], observed: float, cap: int = 200_000, draws: int = 20_000, seed: int = 0
) -> tuple[float, str, int]:
    """p for Cochran's Q by permuting treatment labels within each item.

    The asymptotic chi-square p is the wrong instrument here and the reason
    is the same one that makes the design good: the exchangeable unit is the
    item, and there are usually few of them that vary. Under the null the k
    labels within an item are exchangeable, so the reference distribution
    can be built rather than assumed.

    An item with r successes out of k has C(k, r) distinguishable
    arrangements, so when the product over the varying rows is small the
    null distribution is enumerated **exactly**; otherwise it is sampled.
    Returns (p, how, size).
    """
    k = len(matrix[0]) if matrix else 0
    varying = [r for r in matrix if 0 < sum(r) < k]
    fixed = [r for r in matrix if not (0 < sum(r) < k)]
    if not varying or k < 2:
        return (1.0, 'not run (no item varies across levels)', 0)

    total = 1
    for r in varying:
        total *= math.comb(k, sum(r))
        if total > cap:
            break

    def q_of(rows):
        return cochran_q(rows + fixed)[0]

    if total <= cap:
        arrangements = []
        for r in varying:
            ones = sum(r)
            per = []
            for spots in itertools.combinations(range(k), ones):
                row = [0] * k
                for s in spots:
                    row[s] = 1
                per.append(row)
            arrangements.append(per)
        hits = 0
        size = 0
        for combo in itertools.product(*arrangements):
            size += 1
            if q_of(list(combo)) >= observed - 1e-12:
                hits += 1
        return (hits / size, f'exact, all {size} arrangements enumerated', size)

    rng = random.Random(seed)
    hits = 0
    for _ in range(draws):
        shuffled = []
        for r in varying:
            row = list(r)
            rng.shuffle(row)
            shuffled.append(row)
        if q_of(shuffled) >= observed - 1e-12:
            hits += 1
    # The +1 is not cosmetic: a Monte Carlo p of exactly 0 asserts an
    # impossibility the sampling cannot establish.
    return ((hits + 1) / (draws + 1), f'monte carlo, {draws} draws (seed {seed})', draws)


def mcnemar_exact(b: int, c: int) -> tuple[float, int]:
    """Exact McNemar for one paired 2x2: the binomial sign test on discordants.

    b and c are the two discordant counts (pass-then-fail, fail-then-pass).
    Concordant pairs carry no information about a difference and are not in
    the denominator -- which is the whole point of using this rather than a
    chi-square on the four cells. The exact form is used unconditionally
    rather than only when b+c is small, because b+c here is never large.
    """
    n = b + c
    if n == 0:
        return (1.0, 0)
    return (float(sps.binomtest(b, n, 0.5, alternative='two-sided').pvalue), n)


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order.

    Several comparisons against one baseline is several chances to find a
    difference that is not there. Holm rather than Bonferroni because it is
    uniformly more powerful and makes the same assumption (none), and this
    data has no power to spare.
    """
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        value = (m - rank) * pvalues[i]
        running = max(running, value)
        adjusted[i] = min(1.0, running)
    return adjusted


def mcnemar_power(n: int, discordance: float, effect: float, alpha: float = 0.05) -> float:
    """Power of an exact-ish paired binary test, by the normal approximation.

    n pairs, `discordance` the probability an item disagrees between the two
    conditions, `effect` the difference in pass rate. This is Connor's
    approximation, inverted below to answer the only question worth asking
    of a small-n paired experiment: how many items would it take.
    """
    if discordance <= 0 or effect <= 0 or effect > discordance or n <= 0:
        return 0.0
    z_a = float(sps.norm.ppf(1 - alpha / 2))
    num = effect * math.sqrt(n) - z_a * math.sqrt(discordance)
    den = math.sqrt(max(discordance - effect * effect, 1e-12))
    return float(sps.norm.cdf(num / den))


def items_needed(discordance: float, effect: float, power: float = 0.80, alpha: float = 0.05, cap: int = 100_000) -> int | None:
    """Smallest n with at least `power` to detect `effect`. None if unreachable."""
    if discordance <= 0 or effect <= 0 or effect > discordance:
        return None
    lo, hi = 1, 8
    while hi <= cap and mcnemar_power(hi, discordance, effect, alpha) < power:
        lo, hi = hi, hi * 2
    if hi > cap:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if mcnemar_power(mid, discordance, effect, alpha) >= power:
            hi = mid
        else:
            lo = mid + 1
    return lo


def detectable_effect(n: int, discordance: float, power: float = 0.80, alpha: float = 0.05) -> float | None:
    """The smallest effect n items can detect: the MDE curve's y value."""
    if discordance <= 0 or n <= 0:
        return None
    lo, hi = 1e-4, discordance
    if mcnemar_power(n, discordance, hi * 0.999, alpha) < power:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        if mcnemar_power(n, discordance, mid, alpha) >= power:
            hi = mid
        else:
            lo = mid
    return hi


# ---------------------------------------------------------------------------
# the design: what was actually run, and what that permits
# ---------------------------------------------------------------------------
FACTORS = ['model', 'config_id', 'tier', 'benchmark', 'adapter_sha', 'system_sha']

# system_sha NULL means no system prompt was sent -- a real level, per the
# outer repo's migration 2 schema note. adapter_sha NULL means the adapter
# was not recorded, which is unknown and is never pooled with a known one,
# per migration 3's. The two NULLs are opposites and the labels keep them
# apart.
NONE_LEVEL = '(none)'
UNKNOWN = '?'


def level_of(record: dict, factor: str) -> str:
    value = record.get(factor)
    if factor == 'system_sha':
        if not value:
            return NONE_LEVEL
        return f'{record.get("system_name") or "?"}@{value[:12]}'
    if factor == 'adapter_sha':
        return value[:12] if value else UNKNOWN
    if factor == 'config_id':
        return value or 'unrecorded'
    return str(value) if value is not None else UNKNOWN


class Block:
    """One (tier, model, config, benchmark, adapter) with its system levels.

    The block is the unit the paired tests run on, because it is the largest
    set of rows inside which the only thing that differs is the system
    prompt. Crossing a benchmark or an adapter boundary would change the
    items or the wrapper around them, and pooling across either would be
    comparing prompts by comparing questions.
    """

    def __init__(self, key: tuple[str, str, str, str, str]):
        self.tier, self.model, self.config, self.benchmark, self.adapter = key
        self.cells: dict[tuple[str, str], list[dict]] = {}

    @property
    def key(self):
        return (self.tier, self.model, self.config, self.benchmark, self.adapter)

    def label(self) -> str:
        return f'{self.benchmark} x adapter {self.adapter} ({self.model}, tier {self.tier})'

    def slug(self) -> str:
        """A filename for this block's figures.

        `?` is the label for an unrecorded adapter and is also a shell glob
        and an illegal character on some filesystems, so it is spelled out
        here. The figure and the table have to name the same block, and a
        link that does not resolve is worse than an ugly filename.
        """
        adapter = 'unknown-adapter' if self.adapter == UNKNOWN else self.adapter
        return f'{self.benchmark}-{adapter}-{self.tier}'

    def levels(self) -> list[str]:
        seen: dict[str, int] = {}
        for (_item, level), records in self.cells.items():
            first = min((r.get('at') or 0) for r in records)
            if level not in seen or first < seen[level]:
                seen[level] = first
        # Ordered by when the level was first measured, so the table reads
        # in the order the experiment was run -- which is the order the
        # sequential confound below is about.
        return sorted(seen, key=lambda lv: (seen[lv], lv))

    def items(self) -> list[str]:
        # Natural order, not lexical: see natural_key. The figure and the
        # markdown table above it both read this list, so they cannot
        # disagree.
        return sorted({item for item, _ in self.cells}, key=natural_key)

    def outcome(self, item: str, level: str) -> int | None:
        """1 pass, 0 graded-and-not-pass, None absent.

        A cell measured more than once takes its earliest measurement, and
        the duplication is reported by the reliability section rather than
        averaged away here: averaging two contradictory verdicts into 0.5
        would put a number in the matrix that no run produced.
        """
        records = self.cells.get((item, level))
        if not records:
            return None
        first = sorted(records, key=lambda r: (r.get('at') or 0))[0]
        if first.get('outcome') not in GRADED:
            return None
        return 1 if first.get('outcome') == PASS else 0

    def matrix(self) -> tuple[list[str], list[str], list[list[int]]]:
        """The balanced item x level matrix, dropping items missing any level."""
        levels = self.levels()
        items = [i for i in self.items() if all(self.outcome(i, lv) is not None for lv in levels)]
        return (items, levels, [[self.outcome(i, lv) for lv in levels] for i in items])

    def complete(self) -> bool:
        items, levels, _ = self.matrix()
        return len(levels) >= 2 and len(items) >= 2

    def holes(self) -> int:
        levels = self.levels()
        return sum(1 for i in self.items() for lv in levels if self.outcome(i, lv) is None)

    def spans(self) -> dict[str, tuple[int, int]]:
        out: dict[str, list[int]] = {}
        for (_item, level), records in self.cells.items():
            for r in records:
                out.setdefault(level, []).append(r.get('at') or 0)
        return {lv: (min(v), max(v)) for lv, v in out.items() if v}

    def sequential(self) -> bool:
        """True when no two levels overlap in time.

        Sequential is not a defect by itself; it becomes one the moment
        anything drifts during the run, because then the drift and the
        level are the same variable. That is checked mechanically here
        rather than remembered.
        """
        spans = sorted(self.spans().values())
        return all(spans[i][1] < spans[i + 1][0] for i in range(len(spans) - 1))

    def replicated(self) -> int:
        return sum(1 for records in self.cells.values() if len(records) > 1)


#: The block key when the levels are system prompts, which is what every
#: caller wants by default: hold everything else fixed and vary the prompt.
BLOCK_KEY = ['tier', 'model', 'config_id', 'benchmark', 'adapter_sha']


def blocks_of(records: list[dict], level_factor: str = 'system_sha', key_factors: list[str] | None = None) -> list[Block]:
    """Group graded records into blocks whose columns are levels of one factor.

    `level_factor` is a parameter because a tuning-style question asks the
    same thing of a different axis: its levels would be serving
    configurations, not system prompts. Block itself never assumed what a
    level is -- only that levels are strings -- so the axis is genuinely all
    that changes, and the paired tests, the discordance figures and the
    refusal machinery are reused rather than rewritten against a second,
    less careful implementation.

    The default keeps every current caller on the system-prompt axis.
    """
    if key_factors is None:
        key_factors = [f for f in BLOCK_KEY if f != level_factor]
        if 'system_sha' not in key_factors and level_factor != 'system_sha':
            key_factors.append('system_sha')
    blocks: dict[tuple, Block] = {}
    for r in records:
        if r.get('outcome') not in GRADED:
            continue
        key = tuple(level_of(r, f) for f in key_factors)
        block = blocks.setdefault(key, Block(key))
        cell = (str(r.get('item_id')), level_of(r, level_factor))
        block.cells.setdefault(cell, []).append(r)
    return sorted(blocks.values(), key=lambda b: b.key)


# ---------------------------------------------------------------------------
# throttle regimes: which physical machine state served each request
# ---------------------------------------------------------------------------
class Regimes:
    """Per-result GPU regime, from the samples taken while it was being served.

    A throughput number is a statement about a machine as much as about a
    model, and a GPU under a software power cap for part of its life makes
    that literal. Classifying each request by the throttle word observed
    during its own wall-clock window is what makes "the prompt was slower"
    separable from "the card was slower by then".

    Built from a pre-fetched `run_id -> samples` map rather than a live
    connection: the original queried the whole `gpu_sample` table directly
    (`SELECT ... FROM gpu_sample ORDER BY run_id, at`), which a report built
    against an async Table-wrapper cannot do without adding a new
    unfiltered-scan method to that wrapper. Instead build_report() fetches
    `BenchmarkGpuSamples.list_samples(run_id)` for exactly the run ids
    referenced by the records in scope -- every run id a lookup here could
    ever be asked about -- during its async data-fetch phase, and this class
    only ever reads that map.
    """

    def __init__(self, samples_by_run: dict[int, list[dict]]):
        # Each list is assumed pre-sorted by `at` ascending -- true of
        # BenchmarkGpuSamples.list_samples()'s own ORDER BY.
        self.by_run = samples_by_run

    def window(self, record: dict) -> list[dict]:
        run_id = record.get('run_id')
        end = _epoch_dt(record.get('at'))
        if run_id is None or end is None:
            return []
        wall = float(record.get('wall_ms') or 0.0)
        start = end - dt.timedelta(milliseconds=wall)
        inside = [s for s in self.by_run.get(run_id, []) if (t := _epoch_dt(s.get('at'))) and start <= t <= end]
        if inside:
            return inside
        # A request shorter than the sampling interval can fall between two
        # samples. The nearest sample is the honest answer there; an empty
        # one would read as "no throttling seen", which is a different
        # claim.
        nearest = min(
            self.by_run.get(run_id, []),
            key=lambda s: abs(((_epoch_dt(s.get('at')) or end) - end).total_seconds()),
            default=None,
        )
        return [nearest] if nearest else []

    def label(self, record: dict) -> str:
        samples = self.window(record)
        if not samples:
            return 'no telemetry'
        return lstats.throttle_reasons(samples) or 'none'

    def power(self, record: dict) -> float | None:
        values = [s['power_w'] for s in self.window(record) if s.get('power_w')]
        return lstats.mean(values) if values else None


# ---------------------------------------------------------------------------
# markdown helpers
# ---------------------------------------------------------------------------
def table(columns: list[str], rows: list[list[str]]) -> str:
    """A markdown table, through the renderer the rest of the repo uses."""
    return '\n'.join(lstats.render_table(columns, rows))


def rate(passed: int, attempted: int) -> str:
    """`passed/attempted = xx.x%`, never a bare percentage.

    Load-bearing at these sample sizes: at n=8 one item is 12.5 points, and
    a lone `87.5%` invites a comparison the sample cannot support.
    """
    if attempted <= 0:
        return '0/0 = -'
    return f'{passed}/{attempted} = {100.0 * passed / attempted:.1f}%'


def interval(passed: int, attempted: int) -> str:
    ci = wilson(passed, attempted)
    return '-' if ci is None else f'[{100 * ci[0]:.0f}, {100 * ci[1]:.0f}]'


def pv(p: float | None) -> str:
    if p is None:
        return '-'
    if p < 0.001:
        return '<0.001'
    return f'{p:.3f}'


def num(value, spec: str = '{:.2f}') -> str:
    return '-' if value is None else spec.format(value)


def h(level: int, text: str) -> str:
    return f'{"#" * level} {text}'


# ---------------------------------------------------------------------------
# section 2: the design audit
# ---------------------------------------------------------------------------
class Audit:
    """What the design permits, decided before any test is run.

    Every entry here is derived from the rows, not remembered from a
    decisions log, so it stays true as the store grows. `vetoes` is
    consulted by the sections below; an analysis whose key appears there is
    printed as refused with these reasons attached, rather than run and
    caveated.
    """

    def __init__(self):
        self.notes: list[str] = []
        self.vetoes: dict[str, list[str]] = {}

    def veto(self, key: str, reason: str) -> None:
        self.vetoes.setdefault(key, []).append(reason)

    def blocked(self, key: str) -> list[str]:
        return self.vetoes.get(key, [])


def determines(records: list[dict], a: str, b: str) -> bool:
    """True when each level of `a` occurs with exactly one level of `b`."""
    seen: dict[str, set[str]] = {}
    for r in records:
        seen.setdefault(level_of(r, a), set()).add(level_of(r, b))
    return all(len(v) == 1 for v in seen.values())


def audit_design(records: list[dict], blocks: list[Block], regimes: Regimes) -> tuple[str, Audit]:
    audit = Audit()
    # Every factor is audited, including whichever one blocks_of put on the
    # level axis. Excluding the level factor was considered, on the theory
    # that a factor cannot be confounded with itself -- but it cannot be,
    # since aliasing only ever compares distinct pairs, and excluding it
    # would suppress the one veto that matters most when sweeping down to a
    # single surviving config_id: that has one level and no contrast, and
    # has to be told so, not quietly analysed as though it had two.
    factors = list(FACTORS)
    out = [
        h(2, '2. Design audit'),
        '',
        'What follows is derived from the rows, not asserted. A factor with one',
        'level cannot be tested; two factors that move together cannot be told',
        'apart; and a comparison whose levels ran under different machine states',
        'is a comparison of machine states. Each finding below either permits an',
        'analysis or refuses one, and the refusals are honoured by the sections',
        'that follow.',
        '',
    ]

    graded_records = graded(records)
    rows = []
    for factor in factors:
        levels = sorted({level_of(r, factor) for r in graded_records})
        shown = ', '.join(levels[:6]) + (' ...' if len(levels) > 6 else '')
        rows.append([factor, str(len(levels)), shown])
    out += [h(3, 'Factors and their levels'), '', table(['factor', 'levels', 'values'], rows), '']

    # --- constant factors -------------------------------------------------
    constant = [f for f in factors if len({level_of(r, f) for r in graded_records}) < 2]
    if constant:
        out += [
            '',
            f'**Constant, so untestable:** {", ".join(constant)}. '
            'A factor with one level contributes no contrast; it is part of '
            'the conditions every number here is conditional on, not a '
            'variable.',
            '',
        ]
        for f in constant:
            audit.veto(f'factor:{f}', 'only one level was ever measured')

    # --- aliasing ---------------------------------------------------------
    aliased = []
    for a, b in itertools.combinations(factors, 2):
        if a in constant or b in constant:
            continue
        if determines(graded_records, a, b) and determines(graded_records, b, a):
            aliased.append((a, b))
    if aliased:
        lines = []
        for a, b in aliased:
            lines.append(
                f'- **`{a}` and `{b}` are perfectly confounded.** Every '
                f'level of one occurs with exactly one level of the other, '
                f'so no data separates them: an apparent `{a}` effect is '
                f'equally an apparent `{b}` effect, and this report will '
                f'not attribute a difference to either.'
            )
            audit.veto(f'factor:{a}', f'perfectly confounded with `{b}`')
            audit.veto(f'factor:{b}', f'perfectly confounded with `{a}`')
        out += [h(3, 'Aliased factors'), ''] + lines + ['']
    else:
        out += [h(3, 'Aliased factors'), '', 'None: every pair of varying factors is crossed at least partly.', '']

    # --- cold/warm ----------------------------------------------------------
    # The original queried a `v_request` view's `is_cold` column, joined
    # over every request row regardless of grading. This port has no
    # separate request table exposed to it -- `BenchmarkResultModel.timings`
    # already carries the same llama.cpp timings dict a request row would,
    # denormalised onto the result -- so cold/warm is read straight off the
    # in-scope records via stats.is_cold(), over every record that carries
    # timings at all (graded or not, matching the original's scope).
    cold_counts: dict[bool, int] = {}
    for r in records:
        if not r.get('timings'):
            continue
        cold = lstats.is_cold(r)
        cold_counts[cold] = cold_counts.get(cold, 0) + 1
    if len(cold_counts) < 2:
        only = next(iter(cold_counts), None)
        total = sum(cold_counts.values())
        out += [
            f'**Prompt cache:** `is_cold` is {only} for all {total} requests with '
            f'timings recorded in this scope, so cold-versus-warm is not a factor '
            f'here -- every measurement is (or none is) a cold prefill.',
            '',
        ]
        audit.veto('factor:is_cold', 'constant across every request in scope')

    # --- per-block structure ---------------------------------------------
    rows = []
    for b in blocks:
        items, levels, _ = b.matrix()
        seq = 'sequential' if b.sequential() else 'interleaved'
        regs = sorted({regimes.label(r) for records_ in b.cells.values() for r in records_})
        # The `!` marks a block whose *levels* were served under different
        # machine states. A single-level block cannot have that problem --
        # it has no contrast to confound -- so it is never flagged, however
        # many regimes its own requests spanned.
        flag = f'{len(regs)} !' if len(regs) > 1 and len(b.levels()) > 1 else str(len(regs))
        rows.append(
            [
                b.benchmark,
                b.adapter,
                b.model,
                b.tier,
                str(len(b.levels())),
                str(len(items)),
                str(b.holes()),
                str(b.replicated()),
                seq,
                flag,
            ]
        )
    out += [
        h(3, 'Blocks'),
        '',
        'A *block* is one (tier, model, config, benchmark, adapter): the largest',
        'set of rows inside which the only thing that differs is the system',
        'prompt. `holes` counts item x level cells with no graded result;',
        '`repl` counts cells measured more than once; `regimes` is how many',
        'distinct GPU throttle states the block\'s requests were served under,',
        'and a `!` marks a block whose levels did not share one.',
        '',
        table(['benchmark', 'adapter', 'model', 'tier', 'levels', 'items', 'holes', 'repl', 'level order', 'regimes'], rows),
        '',
    ]

    for b in blocks:
        if not b.complete():
            continue
        if b.sequential():
            audit.veto(
                f'throughput:{b.key}',
                'levels ran sequentially, never interleaved, so anything '
                'that drifted during the run is confounded with the level',
            )
        regs: dict[str, set[str]] = {}
        for cell, records_ in b.cells.items():
            for r in records_:
                regs.setdefault(regimes.label(r), set()).add(cell[1])
        if len(regs) > 1:
            detail = '; '.join(f'`{k}` covers {len(v)} of {len(b.levels())} levels' for k, v in sorted(regs.items()))
            audit.veto(f'throughput:{b.key}', f'its levels were not served under one GPU state ({detail})')
        if b.replicated() == 0:
            audit.veto(
                f'anova:{b.key}',
                'no cell was measured twice, so the design carries no '
                'within-cell error term and any F it produced would be '
                'testing the residual against itself',
            )

    refused = [k for k in audit.vetoes if k.startswith('throughput:')]
    if refused:
        out += [
            h(3, 'Refused analyses'),
            '',
            'These are computed nowhere below except where the report shows a',
            'naive test deliberately, labelled as wrong:',
            '',
        ]
        for key in sorted(refused):
            name = key.split(':', 1)[1]
            out.append(f'- throughput by system prompt within `{name}`:')
            for reason in audit.vetoes[key]:
                out.append(f'  - {reason}')
        out.append('')
    return ('\n'.join(out), audit)


# ---------------------------------------------------------------------------
# section 1: provenance
# ---------------------------------------------------------------------------
def section_provenance(records: list[dict], filters: dict, runs_by_id: dict[int, dict], schema_notes: list[dict]) -> str:
    """What was in scope, and what filters produced it.

    A reduced relative of the original's provenance section: the CLI opened
    a raw sqlite connection and could run `PRAGMA user_version` and
    `SELECT count(*) FROM <table>` against every table in the store, whole-
    store counts included. The async Table-wrapper layer this module reads
    through exposes typed query methods, not an arbitrary-table row count or
    a schema-version pragma, and adding one is a models/ change outside this
    module's charter -- so the whole-store row-count table is dropped rather
    than approximated, and the remaining fields are the ones this module can
    state honestly from what it actually fetched.
    """
    stamps = [r.get('at') for r in records if r.get('at') is not None]
    span = f'{_iso(min(stamps))} .. {_iso(max(stamps))}' if stamps else '-'
    builds = sorted({row['build'] for row in runs_by_id.values() if row.get('build')})
    models = sorted({r.get('model') or '?' for r in records})
    revisions = sorted({r.get('dataset_revision') or '?' for r in records})
    applied = [f'- {n["noted_on"]}: {n["note"]}' for n in sorted(schema_notes, key=lambda n: n['note_id'], reverse=True)[:3]]

    used = ', '.join(f'`{k}={v}`' for k, v in filters.items() if v) or 'none'
    out = [
        h(2, '1. Provenance'),
        '',
        'Which query produced these numbers, and what was in it. A report',
        'without this is not checkable.',
        '',
        table(
            ['field', 'value'],
            [
                ['data source', 'Postgres, via read-only async Table-wrapper query methods'],
                ['writes issued by this report', 'none'],
                ['filters applied', used],
                ['results in scope', str(len(records))],
                ['graded in scope', str(len(graded(records)))],
                ['result timespan (UTC)', span],
                ['models', ', '.join(models) or '-'],
                ['llama.cpp builds', ', '.join(builds) or '-'],
                ['dataset revisions', ', '.join(revisions) or '-'],
            ],
        ),
        '',
    ]
    if applied:
        out += [
            h(3, 'Most recent schema notes'),
            '',
            "The store's own account of its discontinuities. A number either "
            'side of one of these is not necessarily comparable.',
            '',
        ] + applied + ['']
    return '\n'.join(out)


# ---------------------------------------------------------------------------
# section 3: the reliability floor
# ---------------------------------------------------------------------------
def repeat_cells(records: list[dict]) -> dict[tuple, list[dict]]:
    cells: dict[tuple, list[dict]] = {}
    for r in graded(records):
        key = (
            level_of(r, 'model'),
            level_of(r, 'config_id'),
            level_of(r, 'tier'),
            level_of(r, 'benchmark'),
            str(r.get('item_id')),
            level_of(r, 'system_sha'),
            level_of(r, 'adapter_sha'),
        )
        cells.setdefault(key, []).append(r)
    return {k: v for k, v in cells.items() if len(v) > 1}


def section_reliability(records: list[dict]) -> tuple[str, float | None]:
    """The floor every later effect has to clear.

    Same item, same prompt, same adapter, same configuration, measured
    twice. Anything that differs between the two is the sampler, and any
    between-level difference smaller than this is not distinguishable from
    re-running the same condition. Stated once here, up front, because every
    section after it has to be read against it.
    """
    repeats = repeat_cells(records)
    flipped = {k: v for k, v in repeats.items() if len({r.get('outcome') == PASS for r in v}) > 1}
    temps = sorted(
        {
            str(r.get('params', {}).get('temperature'))
            for r in graded(records)
            if isinstance(r.get('params'), dict) and r.get('params', {}).get('temperature') is not None
        }
    )

    out = [
        h(2, '3. Reliability floor'),
        '',
        'Before comparing conditions, measure the noise inside one. These are',
        'cells -- one (model, config, tier, benchmark, item, system prompt,',
        'adapter) -- that were measured more than once. Nothing distinguishes',
        'the two measurements except the sampler, so a cell that changed its',
        "verdict is the instrument's own error rate.",
        '',
    ]
    loose: dict[tuple, int] = {}
    for r in graded(records):
        key = (level_of(r, 'benchmark'), str(r.get('item_id')), level_of(r, 'system_sha'), level_of(r, 'adapter_sha'))
        loose[key] = loose.get(key, 0) + 1
    loose_repeats = sum(1 for v in loose.values() if v > 1)

    if not repeats:
        out += [
            'No cell in scope was measured twice, so there is **no estimate of '
            'run-to-run variability at all**. Every difference reported below '
            'is therefore an upper bound on a real effect and a lower bound on '
            'nothing: it cannot be told from noise, because the noise has not '
            'been measured. Re-running one existing suite unchanged is the '
            'cheapest experiment available and should come before any new '
            'condition.',
            '',
        ]
        if loose_repeats:
            out += [
                f'There *are* {loose_repeats} cells that repeat on a looser key '
                '-- same benchmark item, same system prompt, same adapter, but '
                'a different model, configuration or tier. They are deliberately '
                'not counted above. A different model answering the same '
                'question is a different condition, so a verdict that changes '
                'between them measures the models, not the instrument; counting '
                'those as run-to-run noise would put a between-model effect on '
                'the noise floor and then use that floor to dismiss '
                'between-model effects.',
                '',
            ]
        return ('\n'.join(out), None)

    n, k = len(repeats), len(flipped)
    ci = wilson(k, n)
    floor = 100.0 * k / n
    out += [
        table(
            ['field', 'value'],
            [
                ['cells measured more than once', str(n)],
                ['cells that changed verdict', rate(k, n)],
                ['95% Wilson interval', interval(k, n) + ' %'],
                ['sampler temperature(s)', ', '.join(temps) or '-'],
            ],
        ),
        '',
    ]
    out += [
        f'**The floor is {floor:.0f} percentage points.** Re-running an '
        'identical condition changes the verdict on about that share of '
        'items, so a difference between two conditions smaller than it is '
        'indistinguishable from running the same condition twice. Every '
        'effect below is reported against this number, not against zero.'
    ]
    if ci:
        out += [
            '',
            f'The floor is itself estimated from {n} cells, so it is known '
            f'only to within {interval(k, n)} %. It is a rough bound, and it '
            'is the *tightest* statement this scope supports about its own '
            'repeatability.',
        ]
    if temps and any(t not in ('0', '0.0') for t in temps):
        out += [
            '',
            'The temperature above is why. Sampling at a non-zero temperature '
            'makes the model a random variable, which is legitimate -- it is '
            'how the server actually serves -- but it means a single run per '
            'cell measures one draw from it, not the cell.',
        ]
    out.append('')

    rows = []
    for key, group in sorted(flipped.items()):
        outcomes = [r.get('outcome') for r in sorted(group, key=lambda r: (r.get('at') or 0))]
        rows.append([key[3], key[4], key[5], key[6], ' -> '.join(outcomes)])
    if rows:
        out += [h(3, 'The cells that flipped'), '', table(['benchmark', 'item', 'system', 'adapter', 'outcomes'], rows), '']
    return ('\n'.join(out), k / n)


# ---------------------------------------------------------------------------
# section 4: paired accuracy
# ---------------------------------------------------------------------------
MATRIX_CAP = 40


def baseline_of(levels: list[str]) -> str:
    """The level everything is compared against.

    "no system prompt" when it was measured, because that is the condition
    every run before the feature existed used and the one a difference is
    actually a difference *from*. Otherwise the earliest-measured level, so
    the choice is at least not arbitrary.
    """
    return NONE_LEVEL if NONE_LEVEL in levels else levels[0]


def block_accuracy(b: Block, figs: 'Figures', audit: Audit) -> tuple[str, dict]:
    items, levels, matrix = b.matrix()
    k = len(levels)
    base = baseline_of(levels)
    bi = levels.index(base)

    out = [h(3, b.label()), '']

    # --- the matrix -------------------------------------------------------
    discordant = [i for i, row in enumerate(matrix) if 0 < sum(row) < k]
    shown = discordant + [i for i in range(len(items)) if i not in discordant]
    shown = sorted(shown[:MATRIX_CAP])
    rows = []
    for i in shown:
        marks = ['1' if v else '0' for v in matrix[i]]
        rows.append([items[i]] + marks + ['varies' if i in discordant else 'constant'])
    out += [table(['item'] + levels + ['across levels'], rows), '']
    if len(items) > MATRIX_CAP:
        out += [f'({len(items) - MATRIX_CAP} further items omitted; all {len(discordant)} that vary are shown.)', '']
    out += [
        '1 is a pass, 0 is any graded failure. The rightmost column is the '
        'one that matters: an item with the same verdict under every level '
        'contributes nothing to a within-item test, so the effective sample '
        f'here is **{len(discordant)} of {len(items)} items**, not {len(items)}.',
        '',
    ]

    figs.paired_matrix(b, items, levels, matrix, discordant, out)

    # --- per-level rates --------------------------------------------------
    rows = []
    for j, lv in enumerate(levels):
        passed = sum(row[j] for row in matrix)
        rows.append([lv, rate(passed, len(items)), interval(passed, len(items)) + ' %', 'baseline' if j == bi else ''])
    out += [h(4, 'Per level'), '', table(['system prompt', 'passed/attempted', '95% Wilson', ''], rows), '']
    figs.level_discordance(b, levels, matrix, bi, out)

    # --- Cochran's Q ------------------------------------------------------
    q, dfree, p_asym, n_eff = cochran_q(matrix)
    p_perm, how, _size = cochran_permutation_p(matrix, q)
    out += [
        h(4, "Cochran's Q -- is any level different from any other?"),
        '',
        'The k-sample McNemar. The null is that the pass probability does not',
        'depend on which system prompt was sent, tested *within* each item, so',
        'item difficulty cancels instead of being averaged over. This is the',
        f'test the design supports; a chi-square over the {k} marginal totals',
        'would discard the pairing, and a one-way ANOVA would model a binary',
        'outcome as normal with equal variance, which it is not at either end',
        'of the range where nearly all of this data sits.',
        '',
        table(
            ['field', 'value'],
            [
                ['Q', f'{q:.4f}'],
                ['df', str(dfree)],
                ['items with any variation', f'{n_eff}/{len(items)}'],
                ['asymptotic p (chi-square)', pv(p_asym)],
                ['permutation p', pv(p_perm)],
                ['permutation method', how],
            ],
        ),
        '',
    ]
    if n_eff == 0:
        out += [
            '**No item varies across the levels here**, so there is no '
            'within-item comparison to make: Q is 0 by construction and the '
            'permutation test has nothing to permute. This is not evidence '
            'that the levels agree -- it is the same result a broken '
            'harness would produce, and the two are told apart by the '
            'manipulation check in section 6, not by this table.',
            '',
        ]
    if n_eff < 5:
        out += [
            'The asymptotic p is quoted for completeness and should not be '
            'believed: the chi-square approximation to Q needs many items '
            f'that vary, and there are {n_eff}. The permutation p is exact '
            'under the design\'s own randomisation and is the one to read.',
            '',
        ]

    # --- pairwise McNemar -------------------------------------------------
    raw, detail = [], []
    for j, lv in enumerate(levels):
        if j == bi:
            continue
        bb = sum(1 for row in matrix if row[bi] == 1 and row[j] == 0)
        cc = sum(1 for row in matrix if row[bi] == 0 and row[j] == 1)
        p, ndisc = mcnemar_exact(bb, cc)
        raw.append(p)
        detail.append([lv, str(bb), str(cc), str(ndisc), pv(p)])
    adjusted = holm(raw)
    for row, a in zip(detail, adjusted):
        row.append(pv(a))
    if detail:
        out += [
            h(4, f'Exact McNemar against `{base}`, Holm-corrected'),
            '',
            '`b` is items the baseline passed and this level failed, `c` the',
            'reverse. Concordant items are not in the denominator, which is',
            'the point of the test: they carry no information about a',
            'difference. `n disc` is what each p-value actually rests on.',
            '',
            table(['system prompt', 'b', 'c', 'n disc', 'exact p', 'Holm p'], detail),
            '',
        ]

    verdict = (
        f'No level differs from the baseline at any conventional level, '
        f'and the permutation test finds no difference among the {k} either.'
    )
    if p_perm < 0.05 or any(a < 0.05 for a in adjusted):
        verdict = (
            'At least one contrast survives correction. Read it against '
            'the reliability floor in section 3 before acting on it.'
        )
    out += [f'**Verdict.** {verdict}', '']

    return (
        '\n'.join(out),
        {
            'block': b,
            'items': items,
            'levels': levels,
            'matrix': matrix,
            'baseline': bi,
            'q': q,
            'p_perm': p_perm,
            'n_eff': n_eff,
            'discordant': len(discordant),
        },
    )


def section_accuracy(blocks: list[Block], figs: 'Figures', audit: Audit) -> tuple[str, list[dict]]:
    complete = [b for b in blocks if b.complete()]
    out = [
        h(2, '4. Paired accuracy'),
        '',
        'Tiers are seeded so the same items are drawn for every',
        'configuration. That makes this a **repeated-measures design**: each',
        'item is measured under every system prompt, and the comparison is',
        'within an item rather than between two samples of items. Item',
        'difficulty is the largest source of variance in a coding benchmark and',
        'a paired test removes it entirely, which at these sample sizes is the',
        'difference between a test with some power and one with none.',
        '',
        'So: Cochran\'s Q with a permutation reference distribution across the',
        'levels, exact McNemar against the baseline pairwise. Not ANOVA -- the',
        'response is binary, the cells hold one observation each, and the design',
        'carries no within-cell error term.',
        '',
    ]
    if not complete:
        out += [
            '**No complete block in scope.** A block needs at least two system '
            'levels and at least two items measured under all of them. Nothing '
            'in this scope has that, so there is no paired comparison to run.',
            '',
        ]
        return ('\n'.join(out), [])

    results, extra = [], []
    for b in complete:
        text, summary = block_accuracy(b, figs, audit)
        out.append(text)
        results.append(summary)

    # --- pooling ----------------------------------------------------------
    # Blocks are pooled only with blocks that ran the *same* levels. Grouping
    # by the level signature rather than requiring every block in scope to
    # match is what makes this useful in general: two six-level blocks are
    # poolable even when a two-level block sits beside them, and the
    # two-level one is not quietly dropped into the same test.
    # Keyed on the *set* of levels, not the sequence: a block's columns are
    # ordered by when each level was first measured, and two blocks that ran
    # the same prompts in a different order are still the same experiment --
    # so the signature is order-free and each block's rows are re-indexed
    # into one canonical column order before they are stacked.
    families: dict[tuple, list[dict]] = {}
    for r in results:
        families.setdefault(tuple(sorted(r['levels'])), []).append(r)
    poolable = [(sig, group) for sig, group in families.items() if len(group) > 1]
    for signature_levels, group in sorted(poolable, key=lambda kv: -len(kv[1])):
        results_ = group
        pooled = []
        for r in results_:
            order = [r['levels'].index(lv) for lv in signature_levels]
            pooled += [[row[j] for j in order] for row in r['matrix']]
        q, dfree, p_asym, n_eff = cochran_q(pooled)
        p_perm, how, _ = cochran_permutation_p(pooled, q)
        names = ', '.join(r['block'].label() for r in results_ if r.get('block'))
        out += [
            h(3, f'Pooled: {len(results_)} blocks sharing {len(signature_levels)} levels'),
            '',
            'Legitimate only because these blocks ran the *same* levels, so',
            'the columns mean the same thing in each. Items stay distinct rows',
            '-- an mbpp item and a ds1000 item are never averaged together --',
            'so pooling adds rows to a within-item test rather than mixing two',
            'populations.',
            '',
            f'Pooled here: {names}.',
            '',
            table(
                ['field', 'value'],
                [
                    ['blocks pooled', str(len(results_))],
                    ['items', str(len(pooled))],
                    ['items with any variation', str(n_eff)],
                    ['Q', f'{q:.4f}'],
                    ['df', str(dfree)],
                    ['asymptotic p', pv(p_asym)],
                    ['permutation p', pv(p_perm)],
                    ['permutation method', how],
                ],
            ),
            '',
        ]
        base = baseline_of(list(signature_levels))
        extra.append(
            {
                'pooled': True,
                'matrix': pooled,
                'levels': list(signature_levels),
                'baseline': list(signature_levels).index(base),
                'q': q,
                'p_perm': p_perm,
                'n_eff': n_eff,
                'items': [f'pooled{i}' for i in range(len(pooled))],
                'discordant': n_eff,
                'block': None,
            }
        )
    unpoolable = [sig for sig, group in families.items() if len(group) == 1]
    if len(families) > 1:
        out += [
            '',
            f'**Not pooled across all blocks.** {len(families)} distinct level',
            'sets are present in scope, and stacking blocks that ran different',
            'levels would invent a comparison nobody made. The level sets that',
            'appear once each are analysed above and nowhere else:',
            '',
        ]
        for sig in unpoolable:
            out.append(f'- {len(sig)} levels: {", ".join(sig)}')
        out.append('')
    return ('\n'.join(out), results + extra)


# ---------------------------------------------------------------------------
# section 5: power, and what to run next
# ---------------------------------------------------------------------------
EFFECTS = [0.05, 0.10, 0.15, 0.20]


def section_power(results: list[dict], figs: 'Figures', floor: float | None) -> str:
    out = [
        h(2, '5. Power, and what to run next'),
        '',
        'A null result is only informative if the experiment could have found',
        'something. This section answers the question section 4 raises and',
        'cannot settle: how large an effect would this design have detected,',
        'and how many items would it take to detect a smaller one.',
        '',
        'For a paired binary test the sample size depends on the **discordance',
        'rate** -- the share of items that change verdict between two',
        'conditions -- and not on the pass rate. Concordant items cost a run',
        'and contribute nothing, so a benchmark where nearly every item gives',
        'the same answer under every prompt is an expensive way to learn very',
        'little, however many items it has.',
        '',
    ]

    per_block = [r for r in results if not r.get('pooled')]
    if not per_block:
        out += ['No complete block, so no discordance rate to estimate and no power calculation to make.', '']
        return '\n'.join(out)

    pairs = disc = 0
    for r in per_block:
        matrix, bi = r['matrix'], r['baseline']
        for row in matrix:
            for j in range(len(row)):
                if j == bi:
                    continue
                pairs += 1
                disc += 1 if row[j] != row[bi] else 0
    if pairs == 0:
        out += ['No baseline comparison exists in scope.', '']
        return '\n'.join(out)

    psi = disc / pairs
    ci = wilson(disc, pairs) or (0.0, 1.0)
    items_now = sum(len(r['items']) for r in per_block)
    out += [
        table(
            ['field', 'value'],
            [
                ['baseline-vs-level item comparisons', str(pairs)],
                ['of those, discordant', rate(disc, pairs)],
                ['discordance rate psi', f'{psi:.3f}'],
                ['95% Wilson interval on psi', f'[{ci[0]:.3f}, {ci[1]:.3f}]'],
                ['items entering a baseline comparison', str(items_now)],
            ],
        ),
        '',
    ]

    if psi <= 0:
        out += [
            '**Discordance is zero.** Not one item changed verdict between '
            'the baseline and any other level. No paired test can detect '
            'anything at any sample size under that estimate, because there '
            'is nothing for it to count. What that means practically is that '
            'these items are saturated -- they answer the same way whatever '
            'is put in front of them -- and the next experiment should change '
            'the items, not add more of them.',
            '',
        ]
        return '\n'.join(out)

    rows = []
    for effect in EFFECTS:
        cells = []
        for label, p in (('psi', psi), ('psi lo', ci[0]), ('psi hi', ci[1])):
            n = items_needed(p, effect)
            cells.append('impossible' if effect > p else ('> 100k' if n is None else str(n)))
        rows.append([f'{100 * effect:.0f} pp'] + cells)
    out += [
        h(3, 'Items needed for 80% power at alpha 0.05'),
        '',
        table(['effect to detect', 'at psi', 'at psi low', 'at psi high'], rows),
        '',
        '`impossible` is not a rounding: in a paired binary design the',
        'difference in pass rate cannot exceed the discordance rate, since',
        'every unit of difference has to come from an item that changed. An',
        'effect larger than psi cannot exist under the observed discordance,',
        'so no sample size detects it -- the estimate of psi would have to be',
        'wrong first.',
        '',
    ]

    mde = detectable_effect(items_now, psi)
    out += [h(3, 'What this experiment could have found'), '']
    if mde is None:
        ceiling = mcnemar_power(items_now, psi, psi * 0.999)
        out += [
            f'**Nothing.** At {items_now} items and psi = {psi:.3f}, even a '
            f'difference of {100 * psi:.1f} percentage points -- the largest '
            f'one that can exist under this discordance rate -- would be '
            f'detected only {100 * ceiling:.0f}% of the time. There is no '
            f'effect size this experiment had an 80% chance of finding, so '
            f'its null result is a statement about the experiment and not '
            f'about the prompts.',
            '',
        ]
    else:
        out += [
            f'With {items_now} items and psi = {psi:.3f}, the smallest '
            f'difference detectable at 80% power is **about {100 * mde:.0f} percentage points**.',
            '',
        ]
    if floor is not None:
        out += [
            f'Compare that with the reliability floor of {100 * floor:.0f} percentage points from section 3. '
            + (
                'The experiment cannot resolve anything smaller than its own '
                're-run noise, so a null result here is the expected outcome '
                'of the design and is not evidence that the prompts are equivalent.'
                if mde is None or mde >= floor
                else 'The design can resolve effects below its own re-run noise, so a null here is informative.'
            ),
            '',
        ]
    figs.mde_curve(psi, ci, items_now, out)
    out += [
        h(3, 'What to run'),
        '',
        (
            '- Re-run one existing condition unchanged. Nothing in this scope '
            'measures run-to-run variability at all, so there is currently no '
            'scale against which to read any difference below.'
            if floor is None
            else (
                f'- Re-run one existing condition unchanged, to put a tighter '
                f'interval on the {100 * floor:.0f} pp reliability floor. It is the '
                f'largest number in this document and rests on the fewest observations.'
            )
        ),
        f'- If the system-prompt question is worth settling, it needs the item '
        f'count in the table above at the tier that supplies it, not more levels '
        f'at a small one.',
        f'- Cheaper alternative: drop the levels that are indistinguishable '
        f'from the baseline and spend the runs on two levels with more items. '
        f'Six levels at 8 items has less power than two levels at 24 for the '
        f'same number of requests.',
        '',
    ]
    return '\n'.join(out)


# ---------------------------------------------------------------------------
# section 6: throughput
# ---------------------------------------------------------------------------
def timing(record: dict, field: str) -> float | None:
    value = (record.get('timings') or {}).get(field)
    return float(value) if isinstance(value, (int, float)) else None


RESPONSES = [
    ('predicted_per_second', 'generation t/s', 'a rate, so it is a property of the machine as much as of the answer'),
    ('prompt_per_second', 'prefill t/s', 'same: a rate, and the most sensitive thing here to the clock'),
    ('predicted_n', 'output tokens', 'a property of the response itself; unaffected by how fast it was produced'),
    ('prompt_n', 'prompt tokens', 'a property of the request; the closest thing here to a manipulation check'),
]


def kruskal_and_anova(samples: list[list[float]]) -> tuple[str, str]:
    """Both tests, or the reason neither ran. Kruskal first: it is the one to read."""
    usable = [s for s in samples if len(s) >= 2]
    if len(usable) < 2:
        return ('refused: fewer than two levels have two observations', '')
    flat = [v for s in usable for v in s]
    if len(set(flat)) < 2:
        return ('refused: every observation is identical', '')
    try:
        kh, kp = sps.kruskal(*usable)
        krow = f'H = {kh:.3f}, p = {pv(float(kp))}, k = {len(usable)}'
    except ValueError as exc:
        krow = f'refused: {exc}'
    try:
        f, fp = sps.f_oneway(*usable)
        frow = f'F = {f:.3f}, p = {pv(float(fp))}, k = {len(usable)}'
    except ValueError as exc:
        frow = f'refused: {exc}'
    return (krow, frow)


def section_throughput(records: list[dict], blocks: list[Block], regimes: Regimes, audit: Audit, figs: 'Figures') -> str:
    out = [
        h(2, '6. Throughput'),
        '',
        'Two of the four responses below are rates, and a rate measured on a',
        'machine that changed speed partway through the experiment is a',
        'measurement of the machine. Levels that ran sequentially -- one whole',
        'suite per system prompt -- have anything that drifted with time',
        'confounded with the level by construction, and section 7 is where any',
        "drift in the GPU's power state shows up.",
        '',
        'So each contrast is shown twice: once as the naive one-way test a',
        'reader would otherwise run, and once stratified by the GPU regime the',
        'requests were actually served under. Where the design audit refused',
        'the contrast, the naive test is still printed -- labelled -- because a',
        'refusal a reader cannot see the consequences of does not persuade',
        'anybody.',
        '',
    ]

    figs.timeline(records, regimes, out)

    complete = [b for b in blocks if b.complete()]
    if not complete:
        out += ['No complete block, so no level contrast to make.', '']
        return '\n'.join(out)

    for b in complete:
        _items, levels, _ = b.matrix()
        out += [h(3, b.label()), '']
        by_level: dict[str, list[dict]] = {}
        for (_item, level), group in b.cells.items():
            by_level.setdefault(level, []).extend(group)

        regime_of = {lv: sorted({regimes.label(r) for r in group}) for lv, group in by_level.items()}
        # " / " between whole regime labels, because a label is itself a
        # comma-separated list of bit names and joining with a comma reads
        # as one multi-bit state rather than two states.
        rows = [[lv, str(len(by_level.get(lv, []))), ' / '.join(regime_of.get(lv, []))] for lv in levels]
        out += [table(['system prompt', 'requests', 'GPU regime(s) during them'], rows), '']

        blocked = audit.blocked(f'throughput:{b.key}')
        if blocked:
            # Not str.capitalize(): it lowercases the rest of the string,
            # and the rest of these strings is throttle bit names read
            # straight out of nvml. `SwPowerCap` is the name; `swpowercap`
            # is not.
            out += ['**This contrast is refused.** ' + ' '.join(f'{r[0].upper()}{r[1:]}.' for r in blocked), '']

        rows = []
        for field_, name, why in RESPONSES:
            samples = [[v for r in by_level.get(lv, []) if (v := timing(r, field_)) is not None] for lv in levels]
            kr, fr = kruskal_and_anova(samples)
            rows.append([name, kr, fr])
        out += [
            h(4, 'Naive one-way tests by system prompt -- the wrong answer'),
            '',
            table(['response', 'Kruskal-Wallis', 'one-way ANOVA'], rows),
            '',
            'Read these as a demonstration, not a result. Both assume the',
            'observations are exchangeable across levels given the null, and',
            'they are not when the level and the elapsed time are the same',
            'variable. ANOVA additionally assumes normal residuals with equal',
            'variance across groups, and a distribution with a large step in it',
            'has neither.',
            '',
        ]

        # --- stratified -----------------------------------------------------
        buckets: dict[str, dict[str, list[float]]] = {}
        for lv in levels:
            for r in by_level.get(lv, []):
                v = timing(r, 'predicted_per_second')
                if v is not None:
                    buckets.setdefault(regimes.label(r), {}).setdefault(lv, []).append(v)
        strat_rows = []
        for regime, levels_here in sorted(buckets.items()):
            usable = {lv: v for lv, v in levels_here.items() if len(v) >= 2}
            if len(usable) < 2:
                strat_rows.append(
                    [regime, str(len(levels_here)), f'{sum(len(v) for v in levels_here.values())}', 'collapses: fewer than two levels survive here']
                )
                continue
            kr, _ = kruskal_and_anova(list(usable.values()))
            strat_rows.append([regime, str(len(usable)), str(sum(len(v) for v in usable.values())), kr])
        out += [
            h(4, 'Stratified by GPU regime -- generation t/s'),
            '',
            table(['regime', 'levels present', 'requests', 'Kruskal-Wallis'], strat_rows),
            '',
        ]
        if all('collapses' in row[-1] for row in strat_rows):
            out += [
                'Every stratum collapses. That is the finding: once the',
                'requests are separated by the machine state they ran under,',
                'no GPU regime contains more than one system prompt, so there',
                'is no throughput comparison left to make. The naive tests',
                'above were comparing power states.',
                '',
            ]
        elif blocked:
            out += [
                'Stratifying removes the *power-state* confound and not the',
                '*ordering* one. The levels still ran one after another inside',
                'each stratum, so a surviving p-value here says the requests',
                'differed, not that the prompt is why. The contrast stays',
                'refused; interleaving the levels within one server session is',
                'what would settle it.',
                '',
            ]

        rows = []
        for field_, name, why in RESPONSES:
            values = [v for lv in levels for r in by_level.get(lv, []) if (v := timing(r, field_)) is not None]
            rows.append(
                [
                    name,
                    str(len(values)),
                    num(lstats.mean(values)) if values else '-',
                    num(lstats.percentile(values, 0.5)) if values else '-',
                    num(min(values)) if values else '-',
                    num(max(values)) if values else '-',
                    why,
                ]
            )
        out += [
            h(4, 'The responses themselves'),
            '',
            table(['response', 'n', 'mean', 'p50', 'min', 'max', 'note'], rows),
            '',
            '`output tokens` and `prompt tokens` are the defensible responses',
            'of the four: they are counts the model produced, not divisions by',
            'a wall clock, so a throttled GPU changes how long they took and',
            'not what they were.',
            '',
        ]
        # `prompt tokens` doubles as a manipulation check, and it is the only
        # check in this report that the independent variable was applied at
        # all. A system prompt that reached the server must show up as more
        # prompt tokens; if it does not, the run measured an empty
        # manipulation and every accuracy result above is a null by
        # construction.
        base = baseline_of(levels)
        base_n = [v for r in by_level.get(base, []) if (v := timing(r, 'prompt_n')) is not None]
        others = [(lv, [v for r in by_level.get(lv, []) if (v := timing(r, 'prompt_n')) is not None]) for lv in levels if lv != base]
        checks = []
        for lv, values in others:
            if not values or not base_n:
                checks.append([lv, '-', '-', 'no data'])
                continue
            delta = lstats.mean(values) - lstats.mean(base_n)
            try:
                _u, up = sps.mannwhitneyu(values, base_n, alternative='two-sided')
                verdict = 'prompt reached the server' if delta > 0 and up < 0.05 else 'not distinguishable from the baseline'
                checks.append([lv, num(delta, '{:+.1f}'), pv(float(up)), verdict])
            except ValueError as exc:
                checks.append([lv, num(delta, '{:+.1f}'), '-', str(exc)])
        if checks:
            out += [
                h(4, 'Manipulation check: did the prompt reach the server?'),
                '',
                'A system prompt is extra tokens in the request body, so it has',
                'to show up as more prompt tokens than the baseline. This is',
                'the only check in this report that the independent variable',
                'was applied at all, so it is checked rather than assumed.',
                '',
                table(['system prompt', 'mean prompt-token delta vs baseline', 'Mann-Whitney p', 'reading'], checks),
                '',
            ]
    return '\n'.join(out)


# ---------------------------------------------------------------------------
# section 7: throttle audit
# ---------------------------------------------------------------------------
def section_throttle(records: list[dict], regimes: Regimes, blocks: list[Block], samples_by_run: dict[int, list[dict]]) -> str:
    out = [
        h(2, '7. Throttle audit'),
        '',
        'Every serving run records GPU telemetry on a fixed interval, and every',
        'sample carries `clocks_throttle_reasons.active` as a bitmask. This',
        'section decodes it per run and per request, because a throughput',
        'number taken under a software power cap and one taken without it are',
        'measurements of two different machines that happen to share a',
        'hostname.',
        '',
    ]

    runs = sorted({r.get('run_id') for r in records if r.get('run_id') is not None})
    rows = []
    for run_id in runs:
        samples = samples_by_run.get(run_id, [])
        if not samples:
            continue
        stats_ = lstats.gpu_stats(samples)
        rows.append(
            [
                str(run_id),
                str(stats_.get('samples', len(samples))),
                num(stats_.get('power_avg'), '{:.1f}'),
                num(stats_.get('power_p95'), '{:.1f}'),
                num(stats_.get('sm_p50'), '{:.0f}'),
                num(stats_.get('temp_max'), '{:.0f}'),
                num(stats_.get('vram_headroom_mib'), '{:.0f}'),
                stats_.get('throttle') or '-',
            ]
        )
    if rows:
        out += [
            h(3, 'Per serving run'),
            '',
            table(
                ['run', 'samples', 'power avg W', 'power p95 W', 'sm p50 MHz', 'temp max C', 'min free VRAM MiB', 'throttle reasons seen'],
                rows,
            ),
            '',
        ]

    # --- regime transitions ------------------------------------------------
    for run_id in runs:
        samples = samples_by_run.get(run_id, [])
        if not samples:
            continue
        transitions = []
        previous = object()
        for s in samples:
            word = s.get('throttle')
            if word != previous:
                transitions.append((s.get('at'), word, lstats.throttle_reasons([s]) or 'none'))
                previous = word
        if len(transitions) < 2:
            continue
        out += [
            h(3, f'Run {run_id}: when the machine changed'),
            '',
            table(['from (UTC)', 'throttle word', 'decoded'], [[_iso(a), str(w), d] for a, w, d in transitions[:20]]),
            '',
        ]
        if len(transitions) > 20:
            out += [f'({len(transitions) - 20} further transitions omitted.)', '']

    # --- per request --------------------------------------------------------
    graded_records = [r for r in graded(records) if r.get('run_id') is not None]
    if graded_records:
        rows = []
        for r in sorted(graded_records, key=lambda r: (r.get('at') or 0))[-60:]:
            rows.append(
                [
                    _iso(r.get('at')),
                    str(r.get('run_id')),
                    r.get('benchmark', '-'),
                    level_of(r, 'system_sha'),
                    num(timing(r, 'predicted_per_second')),
                    num(regimes.power(r), '{:.1f}'),
                    regimes.label(r),
                ]
            )
        out += [
            h(3, 'Per request (most recent 60)'),
            '',
            'Each request joined to the GPU samples taken inside its own',
            "wall-clock window. This is the join the report's refusals rest on.",
            '',
            table(['at (UTC)', 'run', 'benchmark', 'system', 'gen t/s', 'power W', 'regime'], rows),
            '',
        ]

    # --- the flag -----------------------------------------------------------
    flagged = []
    for b in blocks:
        if not b.complete():
            continue
        by_regime: dict[str, set[str]] = {}
        for (_item, level), group in b.cells.items():
            for r in group:
                by_regime.setdefault(regimes.label(r), set()).add(level)
        if len(by_regime) > 1:
            flagged.append((b, by_regime))
    if flagged:
        out += [h(3, 'Blocks whose levels did not share a machine state'), '']
        for b, by_regime in flagged:
            out.append(f'- **{b.label()}**')
            for regime, levels in sorted(by_regime.items()):
                out.append(f'  - `{regime}`: {", ".join(sorted(levels))}')
        out += [
            '',
            'Any throughput difference between levels listed under different',
            'regimes is a difference between regimes. The design audit refused',
            'those contrasts and section 6 shows what the naive test would have',
            'claimed instead.',
            '',
        ]
    else:
        out += [
            "Every complete block's levels were served under one GPU regime, "
            'so no throughput contrast in this scope is confounded with the '
            "machine's power state.",
            '',
        ]
    return '\n'.join(out)


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def headline(results: list[dict], floor: float | None, audit: Audit, blocks: list[Block]) -> str:
    """What the document concludes, before the reader has to earn it.

    Placed first because the sections are ordered by what gates what, not by
    what matters, and a reader who stops after one screen should still leave
    with the finding rather than a row count.
    """
    lines = [h(2, 'Headline'), '']
    complete = [b for b in blocks if b.complete()]
    pooled = next((r for r in results if r.get('pooled')), None)
    per_block = [r for r in results if not r.get('pooled')]

    if not complete:
        lines += [
            '- No block in scope is complete enough for a paired test, so '
            'no comparison of system prompts is made here. Section 4 says '
            'what would produce one.',
            '',
        ]
    else:
        best = pooled or (per_block[0] if per_block else None)
        if best:
            varied = best['n_eff']
            total = len(best['items'])
            lines += [
                f"- **No system prompt differs from any other.** Cochran's Q "
                f"= {best['q']:.3f} on {len(best['levels']) - 1} df, "
                f"permutation p = {pv(best['p_perm'])}, over {total} items of "
                f"which **{varied} vary at all**. The comparison rests on "
                f"those {varied} items and nothing else."
            ]
    if floor is not None:
        lines += [
            f'- **Re-running an identical condition changes the verdict '
            f'{100 * floor:.0f}% of the time.** Every difference in this '
            f'document is smaller than that, so none of them is '
            f'distinguishable from noise.'
        ]
    refused = [k for k in audit.vetoes if k.startswith('throughput:')]
    if refused:
        lines += [
            f'- **{len(refused)} throughput contrast(s) are refused**, not '
            'computed and caveated. Section 6 shows what the naive test '
            'would have claimed.'
        ]
    aliases = sorted({k.split(':', 1)[1] for k in audit.vetoes if k.startswith('factor:')})
    if aliases:
        lines += [
            f'- Factors that cannot be tested in this scope at all: '
            f'{", ".join("`" + a + "`" for a in aliases)} -- either constant '
            f'or perfectly confounded with another factor.'
        ]
    lines += ['', 'Every claim above is derived below, with its n and its assumption check beside it.', '']
    return '\n'.join(lines)


@dataclass
class Figure:
    """One rendered figure: a filename to save it under, its PNG bytes, and alt text.

    The router decides where `name` gets written (e.g. under
    DATA_DIR/benchmarks/reports/<run>/) -- this module never touches a
    filesystem path, matching its read-only-database discipline. The
    markdown already references `name` as a relative image link, so writing
    each Figure's `data` to a file called `name` beside the markdown is all
    a caller has to do to make the links resolve.
    """

    name: str
    data: bytes
    alt: str


@dataclass
class ReportResult:
    """build_report()'s return value: the markdown, and the figures it references.

    `figures` is empty whenever matplotlib is unavailable or figure
    generation was disabled -- the markdown still has every analysis in that
    case, just with unicode text plots in fenced code blocks in place of the
    image links.
    """

    markdown: str
    figures: list[Figure] = field(default_factory=list)


async def _fetch(tier: str | None, model: str | None, benchmark: str | None):
    """The async data-fetch phase: every DB read this report needs, up front.

    Nothing after this point in build_report() awaits anything -- the
    statistics and the figures are pure functions of what is fetched here,
    which is what lets them stay synchronous (scipy and matplotlib have no
    async API to offer them anyway).
    """
    rows = await BenchmarkResults.list_results(tier=tier, model=model)
    records = [r.model_dump() for r in rows]
    if benchmark:
        records = [r for r in records if r.get('benchmark') == benchmark]

    run_ids = sorted({r['run_id'] for r in records if r.get('run_id') is not None})
    samples_by_run: dict[int, list[dict]] = {}
    runs_by_id: dict[int, dict] = {}
    for run_id in run_ids:
        samples = await BenchmarkGpuSamples.list_samples(run_id)
        samples_by_run[run_id] = [s.model_dump() for s in samples]
        run_row = await BenchmarkRuns.get_by_id(run_id)
        if run_row is not None:
            runs_by_id[run_id] = run_row.model_dump()
    notes = [n.model_dump() for n in await BenchmarkSchemaNotes.list_notes()]
    return records, samples_by_run, runs_by_id, notes


def _assemble(
    records: list[dict],
    samples_by_run: dict[int, list[dict]],
    runs_by_id: dict[int, dict],
    schema_notes: list[dict],
    filters: dict,
    figs: 'Figures',
) -> ReportResult:
    """The sync compute-and-render phase: pure given what _fetch() returned.

    Split out from build_report() so it can be unit-tested directly against
    fixture dicts, with no event loop and no database involved -- the same
    reason the original's `build()` took an already-open connection and
    already-fetched records rather than doing its own I/O.
    """
    regimes = Regimes(samples_by_run)
    blocks = blocks_of(records)
    design, audit = audit_design(records, blocks, regimes)
    reliability, floor = section_reliability(records)
    accuracy, section_results = section_accuracy(blocks, figs, audit)

    now = dt.datetime.now(dt.timezone.utc).strftime(TS)
    used = ', '.join(f'{k}={v}' for k, v in filters.items() if v) or 'none'
    head = [
        f'# Measurement report - {now}',
        '',
        f'Filters: {used}.',
        '',
        'This document is output. Nothing reads it back, so it can be pasted,',
        'edited or thrown away without consequence; the database it was',
        'derived from is the record.',
        '',
        '**How to read it.** The sections are ordered so that each gates the',
        'next. The design audit (2) can refuse an analysis, and where it does,',
        'the later section says so instead of running it. The reliability floor',
        '(3) is the scale everything after it is measured against. Counts are',
        'always printed beside rates, because at these sample sizes one item is',
        'several percentage points and a bare percentage invites a conclusion',
        'the sample cannot support.',
        '',
    ]
    if figs.reason:
        head += [f'> Figures are rendered as text plots in fenced blocks: {figs.reason}.', '']

    body = [
        headline(section_results, floor, audit, blocks),
        section_provenance(records, filters, runs_by_id, schema_notes),
        design,
        reliability,
        accuracy,
        section_power(section_results, figs, floor),
        section_throughput(records, blocks, regimes, audit, figs),
        section_throttle(records, regimes, blocks, samples_by_run),
    ]
    markdown = '\n'.join(head) + '\n' + '\n\n'.join(body) + '\n'
    return ReportResult(markdown=markdown, figures=list(figs.figures))


async def build_report(
    *,
    tier: str | None = None,
    model: str | None = None,
    benchmark: str | None = None,
    generate_figures: bool = True,
) -> ReportResult:
    """Build the statistical markdown report, this module's one entry point.

    Async because the data it reads comes through the async Table-wrapper
    query methods; everything after the data is in hand runs synchronously,
    per the module docstring's async-fetch/sync-compute split. Raises
    `SciPyUnavailable` immediately, before any query, if scipy failed to
    import; raises `NoBenchmarkResults` if the tier/model/benchmark filters
    match nothing in scope. Neither of those depends on matplotlib: figures
    always degrade to unicode text plots rather than failing the whole
    report, controlled by `generate_figures` (True still degrades to text
    when matplotlib itself is missing; False always skips figures, even
    when matplotlib is present, mirroring the original's `--no-figures`).
    """
    if not HAVE_SCIPY:
        raise SciPyUnavailable(
            'benchmarks.report needs scipy (binomtest, fisher_exact, wilcoxon, '
            'mannwhitneyu, kruskal, f_oneway, chi2, norm) for its exact tests '
            'and power calculations. Install it into this environment before '
            'requesting a report.'
        )

    records, samples_by_run, runs_by_id, notes = await _fetch(tier, model, benchmark)
    if not records:
        raise NoBenchmarkResults('no results match those filters')

    # Imported here, not at module scope: report_figures.py imports the
    # labelling and statistics helpers defined above it in this module, and
    # this module only ever needs report_figures inside this one function,
    # after it has finished loading -- see the module docstring.
    from open_webui.benchmarks.report_figures import Figures

    filters = {'tier': tier, 'model': model, 'benchmark': benchmark}
    figs = Figures(enabled=generate_figures)
    return _assemble(records, samples_by_run, runs_by_id, notes, filters, figs)
