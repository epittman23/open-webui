"""benchmarks/compare.py - compare models and serving configurations on test results.

Ported from local-llm's scripts/llama_compare.py (outer repo, the `lllm-test
compare` engine). The serving telemetry already ranks configurations by
throughput. This ranks them by whether they were *right*, which is the
question that decides whether local inference can replace OpenRouter -- a
configuration that generates faster and answers worse has not improved
anything.

Groups results by (model, config-id, tier, system prompt, adapter) and joins
each group to the configuration that produced it, using the flag helpers
already ported 1:1 into stats.py (short_spec, short_ot, config_value) rather
than reimplemented, so a row here reads the same as it always did.

Three rules the output obeys, each of them learned from a mistake recorded in
the outer repo's CLAUDE.md decisions log:

  * Counts, never bare percentages. At the smoke tier one item is about four
    points, and `88%` invites a comparison that 24 items cannot support.
  * Tiers do not mix. A 24-item pass rate and a 164-item one are not
    comparable, so rows from different tiers are never ranked against each
    other.
  * A pair of configurations differing in more than one flag is flagged as
    such. This repo lost a measurement to exactly that (the 2026-08-23
    `--parallel` entry): three flags changed at once and the comparison meant
    nothing.

Unlike the original, everything here is async: the sqlite connection and its
plain function calls (db.results(), db.serving_summary(), ...) are replaced
by the async Table-wrapper methods in open_webui.models.benchmark_configs,
.benchmark_tests and .benchmark_telemetry. The grouping and rendering logic
(Group, group_records, flags_of, differing, caveats, and the row builders) is
pure and stays synchronous, exactly as in the original, operating on plain
dicts already fetched from the database.

No FastAPI here, no argparse, no printing: this module returns structured
data (dicts and lists) for a router to shape into a response.
"""

from __future__ import annotations

import datetime as dt
import re

from open_webui.benchmarks.stats import (
    config_value,
    effective_bandwidth,
    headroom_in_layers,
    headroom_warning,
    ngl_fit,
    short_ot,
    short_spec,
)
from open_webui.models.benchmark_configs import BenchmarkConfigs, BenchmarkRunLoadInfos, BenchmarkRuns
from open_webui.models.benchmark_telemetry import BenchmarkGpuSamples, BenchmarkMetricsScrapes
from open_webui.models.benchmark_tests import BenchmarkRequests, BenchmarkResults, BenchmarkSuiteExclusions

# ---------------------------------------------------------------------------
# outcomes -- ported from llama_results.py, which this app has no equivalent
# module for (its two functions used here, graded() and pass_rate(), are
# reimplemented below rather than imported from a nonexistent dependency).
# ---------------------------------------------------------------------------
PASS = 'pass'
SKIPPED = 'skipped'
FAILURES = {'fail_assert', 'fail_error', 'fail_timeout', 'no_code'}
GRADED = FAILURES | {PASS}


def graded(records: list[dict]) -> list[dict]:
    """Only the items that were actually attempted and judged.

    With exclusions held in their own table this is nearly the identity
    function over result rows, and it is kept anyway: it is the one place
    that states the rule, and a skipped row written by some future path
    would still be caught here rather than silently entering a denominator.
    """
    return [r for r in records if r.get('outcome') in GRADED]


def pass_rate(records: list[dict]) -> tuple[int, int, float | None]:
    """(passed, attempted, rate) -- always returned together.

    Never a bare rate. At the smoke tier one item is about four points, and
    a percentage printed without its denominator invites a comparison the
    sample size does not support.
    """
    g = graded(records)
    passed = sum(1 for r in g if r.get('outcome') == PASS)
    return passed, len(g), (passed / len(g) if g else None)


# ---------------------------------------------------------------------------
# joining test results to the serving configuration that produced them
# ---------------------------------------------------------------------------
# These are the keys `_vramlog_config` writes, not the CLI spellings of the
# flags behind them: `-c` is recorded as `ctx`, `--n-cpu-moe` as `moe`, and
# the two cache types share one `cache: k=... v=...` line. A key that is not
# in the recorded lines is read as absent, so naming a flag the way
# llama-server spells it drops it silently -- and a dropped flag is one
# `differing()` cannot report, which is the whole reason this list exists.
FLAG_KEYS = ['ngl', 'parallel', 'ctx', 'threads', 'moe', 'batch', 'ubatch']


def flags_of(config_lines: list[str]) -> dict[str, str]:
    """The flag values this comparison cares about, from a block's config lines."""
    out = {}
    for key in FLAG_KEYS:
        value = config_value(config_lines, key)
        if value is not None:
            out[key] = value
    cache = config_value(config_lines, 'cache') or ''
    # `fa` shares the cache line and so is extracted here rather than
    # through FLAG_KEYS. It is a flag a tuning grid can set, and a flag this
    # warning cannot see reads as an all-clear.
    for name, pattern in (('cache-k', r'k=(\S+)'), ('cache-v', r'v=(\S+)'), ('fa', r'fa: (\S+)')):
        hit = re.search(pattern, cache)
        if hit:
            out[name] = hit.group(1)
    out['spec'] = short_spec(config_value(config_lines, 'speculative'))
    out['-ot'] = short_ot(config_value(config_lines, 'override-tensors'))
    return out


def differing(a: dict, b: dict) -> list[str]:
    """Which flags two configurations disagree on."""
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


# ---------------------------------------------------------------------------
# grouping
# ---------------------------------------------------------------------------
def _iso(ts: int | None) -> str:
    """An epoch second as an ISO-8601 UTC string, or '' when there isn't one.

    result.at is stored as BigInteger (unix seconds) in this port, unlike
    the original sqlite TEXT timestamp -- formatted here so `when()` still
    reads and sorts like the original's "YYYY-MM-DDTHH:MM:SSZ" string.
    """
    if not ts:
        return ''
    return dt.datetime.fromtimestamp(int(ts), tz=dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


class Group:
    """One (model, config-id, tier, system prompt, adapter)."""

    def __init__(self, model: str, config_id: str, tier: str, system: str, adapter: str):
        self.model = model
        self.config_id = config_id
        self.tier = tier
        # The system prompt's sha, or "" for none. Part of the key rather
        # than a column alone: a prompt changes what the model was asked,
        # so averaging a run that carried one with a run that did not would
        # hide exactly the difference someone ran both to measure.
        self.system = system
        # The adapter fingerprint, or "" for a row recorded before adapters
        # were fingerprinted. Unlike `system`, "" here is genuinely unknown
        # rather than "none", so an empty one is labelled `?` and never
        # silently pooled with a known adapter.
        self.adapter = adapter
        self.records: list[dict] = []

    @property
    def key(self):
        return (self.model, self.config_id, self.tier, self.system, self.adapter)

    def system_label(self) -> str:
        """name@sha for a prompt, "-" for none.

        The name comes from the records rather than the key because the key
        is the sha: two rows with the same name and different shas are a
        file that was edited between runs, and they must stay two rows.
        """
        if not self.system:
            return '-'
        names = {r.get('system_name') for r in self.records if r.get('system_name')}
        name = sorted(names)[0] if len(names) == 1 else '?'
        return f'{name}@{self.system}'

    def adapter_label(self) -> str:
        """The adapter sha, or `?` for a row that predates the fingerprint."""
        return self.adapter or '?'

    def graded(self):
        return graded(self.records)

    def rate(self):
        return pass_rate(self.records)

    def by_benchmark(self) -> dict[str, tuple[int, int, float | None]]:
        out = {}
        for name in sorted({r.get('benchmark', '') for r in self.records}):
            subset = [r for r in self.records if r.get('benchmark') == name]
            out[name] = pass_rate(subset)
        return out

    def timing(self, field: str) -> list[float]:
        vals = []
        for r in self.graded():
            t = r.get('timings') or {}
            if isinstance(t.get(field), (int, float)):
                vals.append(float(t[field]))
        return vals

    def gen_tps(self) -> float | None:
        n, ms = self.timing('predicted_n'), self.timing('predicted_ms')
        total = sum(ms)
        return (sum(n) / total) * 1000 if total else None

    def prefill_tps(self) -> float | None:
        """Cold prefill only. A cached prefill measures the cache, not the flags.

        cache_prompt is false by default in llama-test for exactly this
        reason; a request that ran with it on is excluded here rather than
        blended in.
        """
        n = ms = 0.0
        for r in self.graded():
            if (r.get('params') or {}).get('cache_prompt'):
                continue
            t = r.get('timings') or {}
            if isinstance(t.get('prompt_n'), (int, float)) and t.get('prompt_ms'):
                n += float(t['prompt_n'])
                ms += float(t['prompt_ms'])
        return (n / ms) * 1000 if ms else None

    def acceptance(self) -> float | None:
        drafted = sum(self.timing('draft_n'))
        accepted = sum(self.timing('draft_n_accepted'))
        return accepted / drafted if drafted else None

    def wall_minutes(self) -> float:
        return sum(float(r.get('wall_ms') or 0) for r in self.graded()) / 60000

    def passes_per_minute(self) -> float | None:
        """The honest combined metric on this hardware.

        Pass rate alone would rank a configuration that answers correctly
        at one token a second above a usable one; throughput alone is what
        the serving log already reports. This is the quantity being
        optimised: correct answers per unit of wall clock.
        """
        minutes = self.wall_minutes()
        if minutes <= 0:
            return None
        return self.rate()[0] / minutes

    def revision(self) -> str:
        revs = {r.get('dataset_revision', '') for r in self.records if r.get('dataset_revision')}
        return sorted(revs)[0][:12] if len(revs) == 1 else ('mixed' if revs else '-')

    def revision_map(self) -> dict[str, str]:
        """benchmark -> dataset revision, for the disagreement check.

        The `revision` property collapses to "mixed" whenever a row spans
        more than one benchmark, which every tier does. Comparing that
        collapsed string between rows would report a disagreement whenever
        a single-benchmark row sits beside a tier -- so the check is made
        per benchmark instead, and only fires when two rows asked the same
        benchmark at different revisions.
        """
        out: dict[str, str] = {}
        for r in self.records:
            rev = r.get('dataset_revision')
            if rev:
                out[r.get('benchmark', '')] = rev[:12]
        return out

    def when(self) -> str:
        stamps = [r.get('at') for r in self.records if r.get('at')]
        return _iso(max(stamps)) if stamps else ''


def group_records(records: list[dict], by_benchmark: bool = False) -> list[Group]:
    """Group graded records by (model, config-id, tier, system, adapter).

    `by_benchmark` is accepted for signature parity with the original (which
    never used it either) rather than dropped -- a future caller that reads
    the original's history should not find a parameter silently removed.
    """
    groups: dict[tuple, Group] = {}
    for r in records:
        # A result whose server was started by hand has config_id NULL;
        # "unrecorded" is the name it is given here, for display.
        # system_sha is NULL for a result measured without a system prompt,
        # which the schema defines as "none sent" rather than "unknown" --
        # so "" is a real group here, not a missing value. adapter_sha is
        # NULL only for rows written before adapter fingerprinting, where it
        # means "not recorded" -- so it is its own group, and rows either
        # side of the fingerprint are never averaged together.
        key = (
            r.get('model', '?'),
            r.get('config_id') or 'unrecorded',
            r.get('tier', '?'),
            r.get('system_sha') or '',
            r.get('adapter_sha') or '',
        )
        groups.setdefault(key, Group(*key)).records.append(r)
    return list(groups.values())


# ---------------------------------------------------------------------------
# fetching -- the only functions here that touch the database
# ---------------------------------------------------------------------------
async def fetch_results(tier: str | None = None, model: str | None = None) -> list[dict]:
    """Every result, shaped like the records group_records() understands."""
    rows = await BenchmarkResults.list_results(tier=tier, model=model)
    return [r.model_dump() for r in rows]


async def fetch_config_blocks() -> dict[str, list[str]]:
    """config-id -> its configuration lines.

    Replaces the original's `db.configs(conn)` -- the join that used to mean
    globbing logs/*.log and parsing every markdown block in every file to
    rebuild a config-id -> flags dict is now one query against the config
    table.
    """
    configs = await BenchmarkConfigs.list_all()
    return {c.config_id: c.config_text.split('\n') for c in configs}


async def exclusion_note() -> list[str]:
    """What no configuration could attempt, said once rather than per row.

    An item is excluded when it falls outside its adapter's library filter,
    when its library is not installed here, or when the benchmark's own
    reference solution does not pass in this environment. Counting any of
    those as a model failure would measure the box rather than the model --
    on DS-1000 it would have understated every model by about 14 points
    forever -- so they are dropped from the pool before sampling, and the
    denominator beside every rate is the number of items actually asked.
    """
    rows = await BenchmarkSuiteExclusions.list_all()
    if not rows:
        return []
    counts: dict[str, int] = {}
    kinds: dict[str, int] = {}
    for row in rows:
        counts[row.benchmark] = counts.get(row.benchmark, 0) + 1
        kind = row.kind or 'unattemptable'
        kinds[kind] = kinds.get(kind, 0) + 1
    shown = ', '.join(f'{name} {n}' for name, n in sorted(counts.items()))
    # Split by kind, because the two are invalidated by different things:
    # the library filter by editing an adapter, the calibration by a
    # dataset refetch or a library upgrade on this box.
    why = ', '.join(f'{n} {k.replace("_", " ")}' for k, n in sorted(kinds.items(), key=lambda kv: -kv[1]))
    return [
        f'> note: {sum(counts.values())} items are excluded from the pool '
        f'before sampling ({shown}): {why}. They are not in any '
        f'denominator above.'
    ]


# ---------------------------------------------------------------------------
# row builders -- pure, structured data for a router to shape
# ---------------------------------------------------------------------------
def rows_for(groups: list[Group], blocks: dict[str, list[str]], baseline: str | None = None) -> list[dict]:
    """One row per group, ranked. Groups with nothing graded sort last.

    Sorting them last rather than as zero matches the serving comparison: a
    configuration with no measurement is not a configuration that scored
    nothing.

    `baseline`, when given, is a config-id to add a `vs_baseline` pass-rate
    delta against -- computed only within a tier (see the original's
    `_with_deltas`): across tiers the difference of two pass rates is not a
    delta, it is a category error.
    """

    def sort_key(g: Group):
        passed, attempted, rate = g.rate()
        if attempted == 0:
            return (1, 0.0, 0.0)
        return (0, -(rate or 0), -(g.gen_tps() or 0))

    ordered = sorted(groups, key=sort_key)
    base = next((g for g in groups if g.config_id == baseline), None) if baseline else None
    brate = base.rate()[2] if base is not None else None

    rows = []
    for g in ordered:
        passed, attempted, rate = g.rate()
        row = {
            'model': g.model,
            'config_id': g.config_id,
            'tier': g.tier,
            'system': g.system_label(),
            'adapter': g.adapter_label(),
            'flags': flags_of(blocks.get(g.config_id, [])),
            'dataset_revision': g.revision(),
            'passed': passed,
            'attempted': attempted,
            'pass_rate': rate,
            'per_benchmark': {
                name: {'passed': p, 'attempted': a, 'pass_rate': r} for name, (p, a, r) in g.by_benchmark().items()
            },
            'prefill_tps': g.prefill_tps(),
            'gen_tps': g.gen_tps(),
            'acceptance': g.acceptance(),
            'passes_per_minute': g.passes_per_minute(),
            'last_run': g.when(),
        }
        if base is not None:
            if g.tier != base.tier:
                row['vs_baseline'] = None
                row['vs_baseline_note'] = 'different tier'
            elif rate is None or brate is None:
                row['vs_baseline'] = None
            else:
                row['vs_baseline'] = rate - brate
        rows.append(row)
    return rows


def benchmark_rows(groups: list[Group]) -> tuple[list[str], list[dict]]:
    """The pass rate per benchmark, pivoted: (benchmark names seen, one row per group)."""
    names = sorted({r.get('benchmark', '') for g in groups for r in g.records if r.get('benchmark')})
    rows = []
    for g in sorted(groups, key=lambda x: -(x.rate()[2] or 0)):
        per = g.by_benchmark()
        passed, attempted, rate = g.rate()
        rows.append(
            {
                'model': g.model,
                'config_id': g.config_id,
                'tier': g.tier,
                'system': g.system_label(),
                'adapter': g.adapter_label(),
                'per_benchmark': {
                    name: dict(zip(('passed', 'attempted', 'pass_rate'), per.get(name, (0, 0, None))))
                    for name in names
                },
                'passed': passed,
                'attempted': attempted,
                'pass_rate': rate,
            }
        )
    return names, rows


def failure_rows(groups: list[Group]) -> list[dict]:
    """Every item that did not pass, newest group first.

    This is what the per-group sections of the original's logs/tests.log
    carried, and it is the part worth keeping: passes are not listed
    individually because the count is the result, while the failures are
    the list of things to go and look at.
    """
    rows = []
    for g in sorted(groups, key=lambda x: x.when(), reverse=True):
        for r in g.records:
            if r.get('outcome') not in FAILURES:
                continue
            rows.append(
                {
                    'config_id': g.config_id,
                    'model': g.model,
                    'tier': g.tier,
                    'benchmark': r.get('benchmark', ''),
                    'item_id': str(r.get('item_id', '')),
                    'outcome': r.get('outcome', ''),
                    'reason': r.get('reason') or '',
                    'reasoning_chars': r.get('reasoning_chars'),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# caveats -- the warnings that keep a table from being read as more than it is
# ---------------------------------------------------------------------------
def caveats(groups: list[Group], blocks: dict[str, list[str]]) -> list[str]:
    """The warnings that keep a table from being read as more than it is."""
    out: list[str] = []
    tiers = {g.tier for g in groups}
    if len(tiers) > 1:
        out.append(
            f'> note: rows span {len(tiers)} tiers ({", ".join(sorted(tiers))}). '
            f'Pass rates are only comparable within a tier -- the tiers '
            f'ask different questions of different item counts.'
        )

    by_name: dict[str, set] = {}
    for g in groups:
        if g.system:
            names = {r.get('system_name') for r in g.records if r.get('system_name')}
            for name in names:
                by_name.setdefault(name, set()).add(g.system)
    edited = sorted(n for n, shas in by_name.items() if len(shas) > 1)
    if edited:
        out.append(
            f'> note: {", ".join(edited)} appears with more than one '
            f'sha, so prompts/system/ was edited between these runs. '
            f'They are separate rows because they are separate prompts; '
            f"the file's current text is only the most recent one."
        )

    small = sorted({g.tier for g in groups if 0 < g.rate()[1] < 50})
    if small:
        out.append(
            f'> note: {", ".join(small)} has fewer than 50 graded items, '
            f'so one item moves the rate by more than two points. Read '
            f'the counts, not the percentage.'
        )

    seen: dict[str, set[str]] = {}
    for g in groups:
        for b, rev in g.revision_map().items():
            seen.setdefault(b, set()).add(rev)
    split = sorted(f'{b} ({", ".join(sorted(revs))})' for b, revs in seen.items() if len(revs) > 1)
    if split:
        out.append(
            '> warning: rows were measured against different dataset '
            'revisions of ' + '; '.join(split) + '. They are not the '
            'same test.'
        )

    # Within one tier, two rows can still have been asked different
    # questions: `--benchmark humaneval` records tier "smoke" while
    # covering a third of it. The tier label alone would hide that, so the
    # benchmark sets are compared too.
    for tier in sorted(tiers):
        peers = [g for g in groups if g.tier == tier and g.rate()[1] > 0]
        sets = {frozenset(g.revision_map()) for g in peers}
        if len(sets) > 1:
            shown = sorted('/'.join(sorted(s_)) or 'none' for s_ in sets)
            out.append(
                f'> warning: rows in tier `{tier}` cover different '
                f'benchmarks ({"; ".join(shown)}), so their pass rates '
                f'are over different items. Use --by benchmark.'
            )

    # The multi-flag warning, per tier: a comparison is only a comparison
    # when one thing changed.
    for tier in sorted(tiers):
        peers = [g for g in groups if g.tier == tier and g.rate()[1] > 0]
        for i, a in enumerate(peers):
            for b in peers[i + 1 :]:
                if a.config_id == b.config_id:
                    continue
                diff = differing(flags_of(blocks.get(a.config_id, [])), flags_of(blocks.get(b.config_id, [])))
                if len(diff) > 1:
                    out.append(
                        f'> warning: `{a.config_id}` and `{b.config_id}` differ '
                        f'in {len(diff)} flags ({", ".join(diff)}). A difference '
                        f'between them cannot be attributed to any one of them.'
                    )
    unknown = sorted({g.config_id for g in groups if g.config_id not in blocks and g.config_id != 'unrecorded'})
    if unknown:
        out.append(
            f'> note: no config row for '
            f'{", ".join("`" + c + "`" for c in unknown)}, so their flag '
            f'columns are blank. The results are still valid; the flags '
            f'are simply not recorded in this database.'
        )
    if any(g.config_id == 'unrecorded' for g in groups):
        out.append(
            '> note: `unrecorded` means the server was started by hand, '
            'outside llama-serve, so no telemetry run was active and the '
            'flags in force were never captured.'
        )
    return out


# ---------------------------------------------------------------------------
# serving: throughput and GPU telemetry per configuration, independent of
# any test run
# ---------------------------------------------------------------------------
async def serving_summary() -> list[dict]:
    """One row per configuration: its most recent run, and what that run measured.

    Reconstructs the original's `v_config_latest` view plus its follow-on
    joins (request aggregate, GPU aggregate, run_load_info, /metrics
    fallback) as a sequence of async model calls, since this app keeps
    derived views out of the schema. Each row is the configuration's
    *latest* run rather than an average over its history, for the reason
    recorded in the original: an older run may predate a llama.cpp rebuild
    or have shared the machine with something else, and averaging would
    hide the change being looked for.

    Cold and warm prefills are aggregated separately: a request whose
    prompt was partly in a slot processed only the remainder, so its
    prompt_per_second measures a handful of tokens against fixed overhead;
    blending the two produces a prefill figure that belongs to no
    configuration.

    `source` says where the throughput came from: "requests" for the exact
    per-request timings llama-test recorded, "metrics" for the server's own
    counters, which cover every client and whatever prompts they sent. The
    fallback is worth having and is not the same measurement, so it is
    labelled rather than silently mixed in.
    """
    out: list[dict] = []
    latest_runs = await BenchmarkRuns.get_latest_run_per_config()
    for run in sorted(latest_runs, key=lambda r: r.started_at, reverse=True):
        config = await BenchmarkConfigs.get_by_id(run.config_id)
        if config is None:
            continue
        facts: dict = {
            'config_id': config.config_id,
            'alias': config.alias,
            'config_text': config.config_text,
            'run_id': run.run_id,
            'model': run.model,
            'quant': run.quant,
            'build': run.build,
            'started_at': run.started_at,
            'ended_at': run.ended_at,
        }

        agg = await BenchmarkRequests.get_run_request_aggregate(run.run_id)
        if agg:
            facts.update(agg)

        gpu = await BenchmarkGpuSamples.get_run_aggregates(run.run_id)
        if gpu:
            facts.update({k: v for k, v in gpu.items() if k != 'run_id'})

        info = await BenchmarkRunLoadInfos.get_by_run_id(run.run_id)
        if info:
            facts.update({k: v for k, v in info.model_dump().items() if k != 'run_id'})

        cold_n, cold_ms = facts.get('cold_prompt_n'), facts.get('cold_prompt_ms')
        pred_n, pred_ms = facts.get('predicted_n'), facts.get('predicted_ms')
        facts['prefill_tps'] = (cold_n / cold_ms * 1000) if cold_n and cold_ms else None
        facts['gen_tps'] = (pred_n / pred_ms * 1000) if pred_n and pred_ms else None
        drafted = facts.get('draft_n') or 0
        facts['acceptance'] = ((facts.get('draft_n_accepted') or 0) / drafted) if drafted else None
        facts['source'] = 'requests' if facts['gen_tps'] else None

        if facts['gen_tps'] is None:
            delta = await BenchmarkMetricsScrapes.get_delta(run.run_id)
            gen_n = delta.get('llamacpp:tokens_predicted_total')
            gen_s = delta.get('llamacpp:tokens_predicted_seconds_total')
            pp_n = delta.get('llamacpp:prompt_tokens_total')
            pp_s = delta.get('llamacpp:prompt_seconds_total')
            if gen_n and gen_s:
                facts['gen_tps'] = gen_n / gen_s
                facts['source'] = 'metrics'
            if pp_n and pp_s and not facts['prefill_tps']:
                facts['prefill_tps'] = pp_n / pp_s
        out.append(facts)
    return out


def serving_rows(facts: list[dict]) -> tuple[list[dict], list[dict], list[str]]:
    """The serving-side comparison: one row per configuration, fastest first.

    What the original's markdown log rendered at the head of every file, as
    structured data instead of a printed table. Rows are ordered by
    generation throughput; a configuration that has never been measured
    sorts last rather than as zero, because it is unknown, not slow.

    Returns (rows, derived, notes): `derived` is the ms/token and
    cpu-bandwidth figures per configuration, and `notes` carries the
    per-`-ngl`-family least-squares fit (see stats.ngl_fit) plus any
    headroom warnings.
    """
    ordered = sorted(facts, key=lambda r: (r.get('gen_tps') is None, -(r.get('gen_tps') or 0)))

    rows = []
    for r in ordered:
        lines = (r.get('config_text') or '').split('\n')
        rows.append(
            {
                'config_id': r.get('config_id', ''),
                'alias': r.get('alias', ''),
                'model': r.get('model', ''),
                'ngl': config_value(lines, 'ngl'),
                'parallel': config_value(lines, 'parallel'),
                'spec': short_spec(config_value(lines, 'speculative')),
                'override_tensors': short_ot(config_value(lines, 'override-tensors')),
                'fused_gdn': r.get('fused_gdn'),
                'prefill_tps': r.get('prefill_tps'),
                'gen_tps': r.get('gen_tps'),
                'acceptance': r.get('acceptance'),
                'source': r.get('source'),
                'peak_vram_mib': r.get('mem_max'),
                'vram_headroom_mib': r.get('vram_headroom_mib'),
                'build': r.get('build') or '',
                'started_at': r.get('started_at'),
            }
        )

    derived = []
    fit_input = []
    for r in ordered:
        lines = (r.get('config_text') or '').split('\n')
        arch = config_value(lines, 'arch')
        gen = r.get('gen_tps')
        # Derived rather than stored: the load log reports the split it
        # actually made, and the remainder is what generation speed on this
        # hardware tracks.
        cpu_layers = None
        if r.get('layers_total') is not None and r.get('layers_gpu') is not None:
            cpu_layers = r['layers_total'] - r['layers_gpu']
        derived.append(
            {
                'config_id': r.get('config_id', ''),
                'cpu_resident_layers': cpu_layers,
                'ms_per_token': (1000.0 / gen) if gen else None,
                'cpu_bandwidth_gib_s': effective_bandwidth(arch, r.get('cpu_buffer_mib'), gen),
                'headroom_in_layers': headroom_in_layers(r, r.get('vram_headroom_mib')),
            }
        )
        fit_input.append({'config_text': r.get('config_text', ''), 'gen': gen, 'cpu_layers': cpu_layers})

    notes = ['> note: ' + line for line in ngl_fit(fit_input)]
    # A run that fit with almost nothing to spare is a result that will not
    # reproduce after a context-size change, which is the whole point of an
    # -ngl sweep on a small card -- so it is called out rather than left to
    # be read off the headroom column.
    for r in ordered:
        warning = headroom_warning(
            f'{r.get("config_id", "?")} ({r.get("started_at") or ""})', r.get('vram_headroom_mib')
        )
        if warning:
            notes.append(warning)
    return rows, derived, notes
