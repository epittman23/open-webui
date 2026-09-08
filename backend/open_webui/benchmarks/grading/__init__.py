"""benchmarks/grading - answer extraction, program execution, and calibration.

Ported from local-llm's scripts/llama_tests.py (outer repo). Each benchmark's
harness (humaneval.py, mbpp.py, ds1000.py) builds a standalone program from the
benchmark's OWN test material and this package runs it. The rule throughout:
the dataset decides correctness, this package only arranges for its verdict to
be reached.

GRADING RUNS MODEL-GENERATED PYTHON. It runs in a subprocess, in a temporary
working directory, under a timeout. That is process isolation, not a sandbox
-- it is what the upstream benchmark runners do, and it is safe enough for a
single-admin local deployment and not safe against adversarial output. This is
an unchanged trust model from the original CLI tool; no new sandboxing was
added in this port.

REGISTRY dispatches by harness name (adapter['check']['harness']), which for
every adapter shipped today is also the benchmark id -- but the dispatch key
is the harness, since that is what determines how a program is built and
graded.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from open_webui.env import BACKEND_DIR
from open_webui.benchmarks.datasets import datasets_base_dir, read_manifest

from . import ds1000, humaneval, mbpp

# ---------------------------------------------------------------------------
# outcomes
#
# `skipped` is deliberately not a failure. A DS-1000 item whose library is not
# installed says nothing about the model or the serving configuration, and
# counting it as a miss would make installing requirements-extra.txt look like
# a quality improvement.
# ---------------------------------------------------------------------------
PASS = 'pass'
FAIL_ASSERT = 'fail_assert'
FAIL_ERROR = 'fail_error'
FAIL_TIMEOUT = 'fail_timeout'
NO_CODE = 'no_code'
SKIPPED = 'skipped'

FAILURES = {FAIL_ASSERT, FAIL_ERROR, FAIL_TIMEOUT, NO_CODE}
GRADED = FAILURES | {PASS}          # everything except skipped


# ---------------------------------------------------------------------------
# per-benchmark dispatch
# ---------------------------------------------------------------------------
REGISTRY = {
    ds1000.NAME: ds1000,
    humaneval.NAME: humaneval,
    mbpp.NAME: mbpp,
}


def _harness(adapter: dict):
    name = adapter['check']['harness']
    module = REGISTRY.get(name)
    if module is None:
        raise ValueError(f"unknown harness '{name}'")
    return module


# ---------------------------------------------------------------------------
# answer extraction
# ---------------------------------------------------------------------------
FENCE = re.compile(r'```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)```', re.DOTALL)


def extract_code(answer: str, want_def: bool = False) -> str:
    """The code the model meant as its answer.

    The last fenced block wins, because a thinking model that reconsiders puts
    the final version last. When a definition is required, the last block that
    actually contains one wins instead -- models routinely follow a complete
    solution with a short usage snippet, and taking that snippet would fail an
    answer that was correct.

    An unfenced answer is used whole. That is usually a model that ignored the
    formatting instruction rather than one that produced no code, and letting
    the grader reject it is more honest than declaring no_code here.
    """
    blocks = [body for _, body in FENCE.findall(answer)]
    if not blocks:
        return answer.strip()
    if want_def:
        with_def = [b for b in blocks if re.search(r'^\s*(async\s+)?def\s', b, re.M)]
        if with_def:
            return with_def[-1].strip('\n')
    return blocks[-1].strip('\n')


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------
def grader_python() -> str:
    """The interpreter graded code runs under.

    DS-1000 needs pandas/numpy from site-packages, so this must be the venv
    interpreter when there is one. LLAMA_GRADER_PYTHON overrides for the case
    where the venv running this package is not the one holding pandas.
    """
    override = os.environ.get('LLAMA_GRADER_PYTHON')
    if override:
        return override
    venv = BACKEND_DIR / '.venv' / 'bin' / 'python'
    return str(venv) if venv.exists() else sys.executable


def libraries_available(names: list[str]) -> set[str]:
    """Which of these import in the grading interpreter.

    Probed in a subprocess against the interpreter that will actually run the
    graded code, which is not necessarily this one.
    """
    probe = ('import importlib.util,json,sys;'
             'print(json.dumps([n for n in sys.argv[1:] '
             'if importlib.util.find_spec(n.lower()) is not None]))')
    try:
        out = subprocess.run([grader_python(), '-c', probe, *names],
                             capture_output=True, text=True, timeout=60)
        return set(json.loads(out.stdout or '[]'))
    except (OSError, ValueError, subprocess.SubprocessError):
        return set()


def build_program(adapter: dict, item: dict, code: str) -> str | None:
    """The full program to execute, or None when the answer has no usable code."""
    return _harness(adapter).build_program(item, code)


def run_program(source: str, timeout: int, isolated: bool) -> tuple[str, str]:
    """Execute a grading program. Returns (outcome, reason)."""
    with tempfile.TemporaryDirectory(prefix='llama-grade.') as tmp:
        path = Path(tmp) / 'grade.py'
        path.write_text(source)
        argv = [grader_python(), '-I']
        if isolated:
            # -S keeps site-packages out, so a HumanEval solution cannot pass by
            # importing something the benchmark never assumed was there.
            argv.append('-S')
        argv.append(str(path))
        try:
            proc = subprocess.run(argv, cwd=tmp, capture_output=True,
                                  text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return FAIL_TIMEOUT, f'no result within {timeout}s'
        except OSError as exc:
            return FAIL_ERROR, f'could not run grader: {exc}'

    if proc.returncode == 0:
        return PASS, ''

    err = (proc.stderr or '').strip()
    tail = err.splitlines()[-1] if err else f'exit {proc.returncode}'
    if 'AssertionError' in err:
        return FAIL_ASSERT, tail
    return FAIL_ERROR, tail


def grade(adapter: dict, item: dict, answer: str) -> tuple[str, str]:
    """Grade one answer against the benchmark's own tests."""
    module = _harness(adapter)
    code = extract_code(answer or '', want_def=module.WANT_DEF)
    program = module.build_program(item, code)
    if program is None:
        return NO_CODE, 'no code in the response'
    check = adapter['check']
    return run_program(program, int(check.get('timeout', 30)),
                       bool(check.get('isolated', True)))


# ---------------------------------------------------------------------------
# grader self-check
# ---------------------------------------------------------------------------
def reference_answer(adapter: dict, item: dict) -> str | None:
    """The benchmark's own solution, dressed as if a model had answered it.

    This is how the graders are tested without a model. A correct harness
    scores these at ~100%; anything less is a bug in this package, not in an
    answer. It is deliberately wrapped in a fence and pushed back through
    extract_code and build_program, so the extraction path is exercised too --
    a grader that only works on already-clean code would pass a check that
    skipped those steps.

    Returns None when the dataset does not ship a reference for this item.
    """
    return _harness(adapter).reference_answer(item)


def env_fingerprint() -> dict:
    """The grading environment, as the interpreter that will grade sees it.

    Recorded with every calibration because it is what a calibration is about:
    the same item is gradeable under one set of library versions and not under
    another. Probed in the grading subprocess, not in this one, since this
    package may run under a different interpreter than LLAMA_GRADER_PYTHON
    names.
    """
    probe = (
        "import json,sys\n"
        "import importlib.metadata as md\n"
        "out={'python': '%d.%d.%d' % sys.version_info[:3]}\n"
        "for name in ('numpy','pandas','scipy','scikit-learn','pyyaml'):\n"
        "    try: out[name]=md.version(name)\n"
        "    except Exception: pass\n"
        "print(json.dumps(out, sort_keys=True))\n")
    try:
        proc = subprocess.run([grader_python(), '-c', probe],
                              capture_output=True, text=True, timeout=60)
        return json.loads(proc.stdout or '{}')
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}


def calibration_path(bench: str, base: Path | None = None) -> Path:
    return (base or datasets_base_dir()) / bench / 'CALIBRATION.json'


def read_calibration(bench: str, base: Path | None = None) -> dict | None:
    path = calibration_path(bench, base)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def calibration_stale(bench: str, base: Path | None = None) -> str:
    """Why a stored calibration no longer applies, or "" when it still does."""
    cal = read_calibration(bench, base)
    if cal is None:
        return 'not calibrated yet'
    manifest = read_manifest(bench, base) or {}
    if cal.get('content_sha256') != manifest.get('content_sha256'):
        return 'the dataset was refetched since it was calibrated'
    if cal.get('environment') != env_fingerprint():
        return 'the grading environment changed since it was calibrated'
    return ''


def ungradeable(bench: str, base: Path | None = None) -> dict[str, str]:
    """Item ids whose own reference solution does not pass here, and why.

    This is the load-bearing idea behind calibration. DS-1000 was published in
    2022 against pandas 1.x; on pandas 3 a chunk of it fails before any model
    is involved, because DataFrame.append and replace(method=) no longer
    exist. Grading a model against a test its own dataset cannot pass measures
    the library version, not the model -- so those items are skipped, with the
    benchmark's own verdict as the evidence, rather than scored as failures.

    An empty result when nothing is calibrated: the check has to be run
    deliberately, and a missing calibration is reported by the caller rather
    than silently assumed to be all-clear.
    """
    cal = read_calibration(bench, base)
    if not cal or calibration_stale(bench, base):
        return {}
    return dict(cal.get('ungradeable', {}))


def calibrate(bench: str, adapter: dict, items: list[dict],
              on_item=None, base: Path | None = None) -> dict:
    """Grade every reference solution and record which ones this box can verify.

    Returns the calibration record; the caller writes it. Every item is pushed
    through the same extract_code/build_program/run_program path a model
    answer takes, so this tests the harness end to end and not just the exec
    call.
    """
    from open_webui.benchmarks.datasets import item_id

    ungradeable_: dict[str, str] = {}
    checked = 0
    for item in items:
        ref = reference_answer(adapter, item)
        iid = item_id(adapter, item)
        # Counted before the early return, not after: `n_checked` is the pool
        # this calibration covers and `n_checked - n_ungradeable` is read as
        # the gradeable count, so an item that lands in `ungradeable_` without
        # landing in `checked` makes that subtraction go negative.
        checked += 1
        if ref is None:
            ungradeable_[iid] = 'the dataset ships no reference solution'
            continue
        outcome, reason = grade(adapter, item, ref)
        if outcome != PASS:
            ungradeable_[iid] = f'{outcome}: {reason}'[:200]
        if on_item:
            on_item(iid, outcome, reason)

    manifest = read_manifest(bench, base) or {}
    return {
        'benchmark': bench,
        'checked_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'content_sha256': manifest.get('content_sha256', ''),
        'revision': manifest.get('revision', ''),
        'environment': env_fingerprint(),
        'grader_python': grader_python(),
        'n_checked': checked,
        'n_ungradeable': len(ungradeable_),
        'ungradeable': dict(sorted(ungradeable_.items())),
    }


def write_calibration(bench: str, record: dict, base: Path | None = None) -> Path:
    path = calibration_path(bench, base)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=False) + '\n')
    return path
