"""benchmarks/datasets.py - fetched dataset storage, and reading it back.

Ported from local-llm's scripts/llama_fetch.py and the dataset-reading half of
scripts/llama_tests.py (outer repo). Two things live here on purpose: the
fetch/manifest/content-hash logic that downloads a benchmark's rows once, and
the read_items()/read_manifest()/item_id() helpers every other module in this
package uses to read them back -- both sides of the same on-disk layout.

Datasets are fetched rather than vendored, same as upstream. They are
upstream-versioned, they carry three different licenses (MIT, CC-BY-4.0,
CC-BY-SA-4.0), and checking a copy into this repo would make every result
trace back to that copy instead of to a citable release.

Each benchmark lands as:

    <base>/<id>/items.jsonl     the rows, exactly as upstream published them
    <base>/<id>/MANIFEST.json   what was downloaded, and from which revision

The manifest is the point. A pass rate is only reproducible alongside the
dataset revision it was measured on. Every record this package writes carries
the revision recorded here.

Two revision identifiers are kept:

    revision        the upstream commit / dataset sha, when the host will say
    content_sha256  the hash of the bytes actually downloaded

The second always exists and is the stronger pin: it identifies the exact
content, whereas a branch name resolves to different content over time. The
first is what a human can look up.

Base path: <DATA_DIR>/benchmarks/datasets, where DATA_DIR is the fork's own
data directory (open_webui.env.DATA_DIR). Overridable with BENCHMARKS_DATA_DIR
for an alternate deployment, and every function additionally accepts an
explicit `base` argument so a caller or a test can point at a directory of its
own without touching the environment -- mirroring how llama_fetch.py and
llama_tests.py parameterized their own tests/data directory via
LLAMA_TESTS_DATA.

Stdlib only (urllib + json + gzip + tomllib elsewhere in this package): no
`datasets`, no `pyarrow`, no network client beyond what Python ships.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from open_webui.env import DATA_DIR

UA = {'User-Agent': 'open-webui/benchmarks (https://github.com/epittman23/local-llm)'}
TIMEOUT = 60


def datasets_base_dir() -> Path:
    """Where fetched datasets live. Overridable so a run can point at a copy."""
    override = os.environ.get('BENCHMARKS_DATA_DIR')
    return Path(override) if override else DATA_DIR / 'benchmarks' / 'datasets'


# ---------------------------------------------------------------------------
# reading what was fetched
# ---------------------------------------------------------------------------
def items_path(bench: str, base: Path | None = None) -> Path:
    return (base or datasets_base_dir()) / bench / 'items.jsonl'


def manifest_path(bench: str, base: Path | None = None) -> Path:
    return (base or datasets_base_dir()) / bench / 'MANIFEST.json'


def read_manifest(bench: str, base: Path | None = None) -> dict | None:
    path = manifest_path(bench, base)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def read_items(bench: str, base: Path | None = None) -> list[dict]:
    path = items_path(bench, base)
    if not path.exists():
        raise SystemExit(
            f'benchmarks: {bench} is not downloaded yet.\n'
            f'  fetch it first (see datasets.fetch_one)')
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def dotted(obj: dict, path: str):
    """Read "metadata.problem_id" out of a nested record."""
    cur = obj
    for part in path.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def item_id(adapter: dict, item: dict) -> str:
    """The dataset's own identifier for an item, never a positional index.

    Positional ids would silently change meaning when the upstream dataset
    gains or loses a row, and a result file outlives the dataset revision
    that made it.
    """
    raw = dotted(item, adapter['item']['id_field'])
    return str(raw)


# ---------------------------------------------------------------------------
# fetching: github_raw (a single file out of a repository)
# ---------------------------------------------------------------------------
def get(url: str, headers: dict | None = None) -> bytes:
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return resp.read()


def get_json(url: str) -> dict:
    return json.loads(get(url).decode('utf-8'))


def github_revision(repo: str, ref: str, path: str) -> str:
    """The commit that last touched this file, or "" if GitHub will not say.

    Unauthenticated GitHub allows 60 requests an hour, and this is a nicety
    rather than a requirement: content_sha256 pins the data either way. So a
    failure here is a warning, not an error.
    """
    url = (f'https://api.github.com/repos/{repo}/commits'
           f'?path={urllib.parse.quote(path)}&sha={urllib.parse.quote(ref)}&per_page=1')
    try:
        commits = get_json(url)
        if isinstance(commits, list) and commits:
            return commits[0].get('sha', '')
    except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
        print(f'  note: could not resolve the commit sha ({exc}); '
              f'pinning by content hash only', file=sys.stderr)
    return ''


def fetch_github_raw(spec: dict) -> tuple[list[dict], str, str, bytes]:
    repo, ref, path = spec['repo'], spec['ref'], spec['path']
    url = f'https://raw.githubusercontent.com/{repo}/{ref}/{path}'
    print(f'  GET {url}', file=sys.stderr)
    raw = get(url)

    body = gzip.decompress(raw) if spec['format'].endswith('.gz') else raw
    text = body.decode('utf-8')

    if spec['format'].startswith('jsonl'):
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        rows = json.loads(text)
        if not isinstance(rows, list):
            raise SystemExit(f'benchmarks: {url} is not a list of records')

    return rows, github_revision(repo, ref, path), url, raw


# ---------------------------------------------------------------------------
# fetching: hf_rows (the HuggingFace datasets-server, plain JSON without auth)
# ---------------------------------------------------------------------------
def hf_revision(dataset: str) -> str:
    try:
        return get_json(f'https://huggingface.co/api/datasets/{dataset}').get('sha', '')
    except (urllib.error.URLError, json.JSONDecodeError, OSError):
        return ''


def fetch_hf_rows(spec: dict) -> tuple[list[dict], str, str, bytes]:
    dataset, config, split = spec['dataset'], spec['config'], spec['split']
    base = ('https://datasets-server.huggingface.co/rows'
            f'?dataset={urllib.parse.quote(dataset, safe="")}'
            f'&config={urllib.parse.quote(config)}&split={urllib.parse.quote(split)}')
    print(f'  GET {base} (paginated)', file=sys.stderr)

    rows: list[dict] = []
    offset, page = 0, 100
    while True:
        # The endpoint rate-limits; a short pause between pages is cheaper than
        # restarting a download that got a 429 halfway through 1000 rows.
        for attempt in range(5):
            try:
                payload = get_json(f'{base}&offset={offset}&length={page}')
                break
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 502, 503) or attempt == 4:
                    raise
                time.sleep(2 * (attempt + 1))
        else:                                   # pragma: no cover - loop always breaks
            raise SystemExit('benchmarks: gave up on the datasets-server')

        batch = payload.get('rows', [])
        if not batch:
            break
        rows.extend(r['row'] for r in batch)
        print(f'\r  {len(rows)} rows', end='', file=sys.stderr, flush=True)
        offset += len(batch)
        if len(batch) < page:
            break
        time.sleep(0.2)

    print(file=sys.stderr)
    body = '\n'.join(json.dumps(r, sort_keys=True) for r in rows).encode('utf-8')
    return rows, hf_revision(dataset), base, body


# ---------------------------------------------------------------------------
FETCHERS = {'github_raw': fetch_github_raw, 'hf_rows': fetch_hf_rows}


def fetch_one(adapter: dict, force: bool = False, base: Path | None = None) -> dict:
    bench = adapter['id']
    out = (base or datasets_base_dir()) / bench
    manifest = out / 'MANIFEST.json'

    if manifest.exists() and not force:
        current = json.loads(manifest.read_text())
        print(f"{bench}: already downloaded "
              f"({current['n_items']} items, revision "
              f"{(current.get('revision') or current['content_sha256'])[:12]}); "
              f"--force to refetch", file=sys.stderr)
        return current

    spec = adapter['fetch']
    fetcher = FETCHERS.get(spec['kind'])
    if fetcher is None:
        raise SystemExit(f"benchmarks: unknown fetch kind '{spec['kind']}'")

    print(f'{bench}: downloading', file=sys.stderr)
    rows, revision, url, raw = fetcher(spec)
    if not rows:
        raise SystemExit(f'benchmarks: {bench} returned no rows')

    out.mkdir(parents=True, exist_ok=True)
    (out / 'items.jsonl').write_text(
        ''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))

    # Recorded even though nothing filters at fetch time: the filter is applied
    # when a suite is built, so widening it later must not require a refetch.
    counts: dict[str, int] = {}
    spec_filter = adapter.get('filter')
    if spec_filter:
        for row in rows:
            key = str(dotted(row, spec_filter['field']))
            counts[key] = counts.get(key, 0) + 1

    record = {
        'benchmark': bench,
        'name': adapter.get('name', bench),
        'license': adapter.get('license', 'unknown'),
        'homepage': adapter.get('homepage', ''),
        'citation': adapter.get('citation', ''),
        'source_url': url,
        'revision': revision,
        'content_sha256': hashlib.sha256(raw).hexdigest(),
        'fetched_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'n_items': len(rows),
    }
    if counts:
        record['by_' + spec_filter['field'].split('.')[-1]] = dict(sorted(counts.items()))

    manifest.write_text(json.dumps(record, indent=2) + '\n')
    print(f'{bench}: {len(rows)} items -> {out}', file=sys.stderr)
    return record
