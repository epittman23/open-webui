"""benchmarks/adapters.py - adapter TOML loading and prompt rendering.

Ported from local-llm's scripts/llama_tests.py (outer repo). An adapter
describes one benchmark: id, name, license, homepage, citation, the
prompt_template that turns a dataset item into a chat turn, and the
[fetch]/[item]/[filter]/[check] blocks the rest of this package reads.

Every test item in this project comes from a published benchmark that ships
its own ground truth -- an adapter never states an answer, only how a
completion-style stub becomes a prompt and which harness grades the reply.

Parsed with stdlib tomllib (Python 3.11+), on purpose: no TOML dependency.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import tomllib

from open_webui.benchmarks.datasets import dotted

ADAPTERS_DIR = Path(__file__).resolve().parent / 'data' / 'adapters'


def load_adapters() -> dict[str, dict]:
    out = {}
    for path in sorted(ADAPTERS_DIR.glob('*.toml')):
        with open(path, 'rb') as fh:
            adapter = tomllib.load(fh)
        adapter['_path'] = str(path)
        adapter['_sha'] = adapter_sha(adapter)
        out[adapter['id']] = adapter
    return out


# The fields that change what a model is asked or how its answer is graded. A
# benchmark's identity was `dataset_revision` alone, which covers the published
# items and nothing this package wraps around them -- so editing prompt_template
# silently made old and new results incomparable with nothing recording the
# discontinuity, which is the problem config_id and system_sha already solve
# for serving flags and system prompts. Recorded per result as adapter_sha.
FINGERPRINTED = ('prompt_template', 'item', 'filter', 'check')


def adapter_sha(adapter: dict) -> str:
    """Fingerprint an adapter over the fields that affect a measurement.

    Over the parsed subset rather than the file's bytes, which is the one place
    this deliberately differs from system_sha: an adapter carries the prose
    explaining why it is shaped the way it is, and hashing the bytes would file
    every comment edit as a measurement discontinuity. `id`, `name`, `license`,
    `homepage`, `citation` and `[fetch]` are excluded for the same reason --
    they say where the benchmark came from, not what was asked of the model.
    """
    payload = {k: adapter[k] for k in FINGERPRINTED if k in adapter}
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return hashlib.sha1(canonical.encode('utf-8')).hexdigest()[:12]


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------
def render_prompt(adapter: dict, item: dict) -> str:
    """Fill the adapter's template from the item's own fields.

    str.format is not used: benchmark prompts are full of braces (dict literals,
    f-strings, LaTeX) and would blow up or be silently mangled by it. Only the
    named placeholders the adapters actually declare are substituted.
    """
    text = adapter['prompt_template']
    body = dotted(item, adapter['item']['prompt_field']) or ''
    fields = {'prompt': body}
    if adapter['id'] == 'mbpp':
        fields['tests'] = '\n'.join(item.get('test_list', []))
    for key, value in fields.items():
        text = text.replace('{' + key + '}', value)
    return text
