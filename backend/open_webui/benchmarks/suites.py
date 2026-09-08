"""benchmarks/suites.py - suite TOML loading and seeded item selection.

Ported from local-llm's scripts/llama_tests.py (outer repo). A suite names a
tier (smoke/standard/full) as a list of (benchmark, n) pairs; build_suite()
resolves that into the concrete, seeded list of items a run will grade.

Parsed with stdlib tomllib (Python 3.11+), on purpose: no TOML dependency.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import tomllib

from open_webui.benchmarks import grading
from open_webui.benchmarks.datasets import dotted, item_id, read_items

SUITES_DIR = Path(__file__).resolve().parent / 'data' / 'suites'


def load_suite(name: str) -> dict:
    path = SUITES_DIR / f'{name}.toml'
    if not path.exists():
        known = ', '.join(sorted(p.stem for p in SUITES_DIR.glob('*.toml')))
        raise SystemExit(f"benchmarks: no suite '{name}' (known: {known})")
    with open(path, 'rb') as fh:
        return tomllib.load(fh)


# ---------------------------------------------------------------------------
# filtering and sampling
# ---------------------------------------------------------------------------
def filter_items(adapter: dict, items: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split into (runnable, skipped) by the adapter's library filter."""
    spec = adapter.get('filter')
    if not spec:
        return items, []
    field, wanted = spec['field'], set(spec['values'])
    keep, drop = [], []
    for it in items:
        (keep if dotted(it, field) in wanted else drop).append(it)
    return keep, drop


def sample(items: list[dict], n: int, seed: int, adapter: dict) -> list[dict]:
    """A seeded sample, stable across runs and across configurations.

    Sorted by the dataset's own id before sampling, so the selection does not
    depend on the order the rows happened to be written in. n <= 0 means every
    item -- that is what the `full` tier asks for.
    """
    ordered = sorted(items, key=lambda it: item_id(adapter, it))
    if n <= 0 or n >= len(ordered):
        return ordered
    return sorted(random.Random(seed).sample(ordered, n),
                  key=lambda it: item_id(adapter, it))


# ---------------------------------------------------------------------------
# suite assembly
# ---------------------------------------------------------------------------
def build_suite(suite: dict, adapters: dict[str, dict],
                only: str | None = None) -> tuple[list[dict], list[dict]]:
    """Resolve a tier into the concrete list of items to run.

    Returns (selected, skipped). Skipped entries carry their reason so a run
    records what it could not attempt rather than quietly shrinking.
    """
    seed = int(suite.get('seed', 0))
    selected: list[dict] = []
    skipped: list[dict] = []

    for entry in suite.get('benchmark', []):
        bench = entry['id']
        if only and bench != only:
            continue
        adapter = adapters.get(bench)
        if adapter is None:
            raise SystemExit(f"benchmarks: suite '{suite['id']}' names unknown "
                             f"benchmark '{bench}'")
        items = read_items(bench)
        keep, dropped = filter_items(adapter, items)

        # "kind" separates the two reasons an item can be out of the pool,
        # because they are invalidated by different things: the library filter
        # by editing the adapter, the calibration by a dataset refetch or a
        # library upgrade on this box.
        for it in dropped:
            skipped.append({'benchmark': bench, 'item_id': item_id(adapter, it),
                            'kind': 'library_filter',
                            'reason': "outside the adapter's library filter"})

        # Applied before sampling, not after. Dropping items from an already
        # drawn sample would give each benchmark a different effective n and
        # make the tier's advertised size a fiction; dropping them from the
        # pool keeps every tier exactly the size it says it is.
        blocked = grading.ungradeable(bench)
        if blocked:
            still = []
            for it in keep:
                iid = item_id(adapter, it)
                if iid in blocked:
                    skipped.append({'benchmark': bench, 'item_id': iid,
                                    'kind': 'ungradeable',
                                    'reason': f'ungradeable here ({blocked[iid]})'})
                else:
                    still.append(it)
            keep = still

        # The seed is offset per benchmark so the three do not draw parallel
        # index sequences, which would correlate the samples across benchmarks.
        chosen = sample(keep, int(entry.get('n', 0)),
                        seed + sum(map(ord, bench)), adapter)
        for it in chosen:
            selected.append({'benchmark': bench, 'adapter': adapter, 'item': it,
                             'item_id': item_id(adapter, it)})

    return selected, skipped


def interleave(selected: list[dict]) -> list[dict]:
    """Round-robin the selected items across benchmarks, each in item order.

    Exists for llama-tune, which measures a candidate over a prefix of the
    tier and lengthens that prefix each round. A contiguous prefix of the
    natural order would make round 1 a HumanEval-only measurement and a later
    round a DS-1000 one, confounding the round with prompt length and
    difficulty -- which is the same confound the interleaved *candidate*
    order exists to avoid, one level down.

    Ordering within a benchmark is the sample's own, not sorted: `sample` is
    seeded so every configuration draws the same items in the same sequence,
    and re-sorting here would throw that away for no gain.
    """
    by_bench: dict[str, list[dict]] = {}
    for row in selected:
        by_bench.setdefault(row['benchmark'], []).append(row)
    names = sorted(by_bench)
    out: list[dict] = []
    for i in range(max((len(v) for v in by_bench.values()), default=0)):
        for name in names:
            if i < len(by_bench[name]):
                out.append(by_bench[name][i])
    return out


def order_sha(selected: list[dict]) -> str:
    """A fingerprint of an item order, so a resume can prove it got the same one.

    Over (benchmark, item_id) pairs in sequence: the identity of the items and
    the order they run in are both part of what a sweep's rounds mean, since a
    round is a slice of this list.
    """
    text = '\n'.join(f"{r['benchmark']}/{r['item_id']}" for r in selected)
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def missing_libraries(selected: list[dict]) -> dict[str, str]:
    """Item ids that cannot be graded here, mapped to why.

    Checked once up front rather than per item: importing pandas costs about a
    second, and a `full` DS-1000 run would otherwise pay it 510 times to learn
    the same fact.
    """
    needs = {'ds1000': {'Pandas': 'pandas', 'Numpy': 'numpy',
                        'Scipy': 'scipy', 'Sklearn': 'sklearn',
                        'Matplotlib': 'matplotlib', 'Tensorflow': 'tensorflow',
                        'Pytorch': 'torch'}}
    wanted: set[str] = set()
    for row in selected:
        table = needs.get(row['benchmark'])
        if table:
            lib = dotted(row['item'], 'metadata.library')
            if lib in table:
                wanted.add(table[lib])
    if not wanted:
        return {}
    have = {n.lower() for n in grading.libraries_available(sorted(wanted))}

    out: dict[str, str] = {}
    for row in selected:
        table = needs.get(row['benchmark'])
        if not table:
            continue
        module = table.get(dotted(row['item'], 'metadata.library'))
        if module and module.lower() not in have:
            out[f"{row['benchmark']}/{row['item_id']}"] = (
                f'{module} is not installed in the grading environment')
    return out
