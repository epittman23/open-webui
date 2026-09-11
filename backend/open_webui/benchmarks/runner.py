"""benchmarks/runner.py - ask the server, grade the answer, record the result.

Ported from local-llm's scripts/llama_test.py (outer repo): the run/grade/
record loop behind `lllm-test`. The original was already factored so a web UI
could drive it in-process (the standalone dashboard calls prepare_suite() and
run_items() directly), so this is a mechanical async port rather than a
redesign -- the callback-driven shape of run_items() in particular is kept
exactly, since it is what lets a future SSE-streaming router endpoint push
live progress without threads.

Several of the original's decisions are load-bearing and are preserved here
deliberately (see the outer repo's llama_test.py docstring for the full
reasoning):

  * The model name is read from GET /v1/models, not taken from the profile
    (see env_profile.served_model()) -- the profile says what *would* be
    served, the server is already serving something.
  * temperature is pinned to 0 and cache_prompt defaults to false, so a
    repeated prompt measures the configuration rather than the prefix cache.
  * Request parameters are read back out of the built body rather than
    re-derived, so the record reflects the request that was actually sent.
  * ServerGone (a connection refused, or a stream that died mid-read) is
    raised instead of being recorded as a graded outcome: a result is a
    statement about a model's answer, and a dead server is not one. Recording
    it as fail_error would file a serving failure as a model failure and,
    because (suite_run_id, benchmark, item_id) is unique, make the item
    unresumable forever.

Two things the original CLI did that this port deliberately drops, since
neither means anything without a terminal:

  * Live rendering of the answer/thinking split to stdout/stderr as tokens
    arrive (the `show`/`con` parameters throughout llama_test.py). The
    reasoning/content split itself is kept in _read_sse() -- what is dropped
    is only the side effect of printing it. A future streaming router
    endpoint is free to build its own live view on top of the on_record (and,
    if wanted, a lower-level token) callback.
  * KeyboardInterrupt handling in the run loop. There is no Ctrl-C in an ASGI
    worker; cooperative cancellation is should_stop(), exactly as the
    original's docstring already anticipated for llama-tune.

GRADING EXECUTES MODEL-GENERATED PYTHON in a subprocess under a timeout (see
benchmarks/grading/__init__.py). That is process isolation, not a sandbox --
an unchanged trust model from the original CLI tool.

A known issue in the modules this file calls into: load_suite(), read_items()
and build_suite() (suites.py / datasets.py) still raise bare SystemExit on
error, ported faithfully from a CLI tool where that just exited the process.
SystemExit is a BaseException; left to propagate out of an async request
handler it would escape asyncio's task machinery and take the whole ASGI
worker down with it. Nothing in this module calls those functions directly --
every call goes through _guard(), which turns a caught SystemExit into
SuiteLoadError, an ordinary RuntimeError a router can catch and turn into an
HTTP 4xx/5xx.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Awaitable, Callable, Optional, Union

import aiohttp

from open_webui.benchmarks import env_profile
from open_webui.benchmarks.adapters import load_adapters, render_prompt
from open_webui.benchmarks.datasets import read_manifest
from open_webui.benchmarks.grading import calibration_stale, grade
from open_webui.benchmarks.suites import (
    build_suite,
    interleave,
    load_suite,
    missing_libraries,
)
from open_webui.models.benchmark_configs import BenchmarkRuns
from open_webui.models.benchmark_tests import BenchmarkResults, BenchmarkSuiteExclusions

log = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / 'data' / 'prompts'


# ---------------------------------------------------------------------------
# the SystemExit boundary
# ---------------------------------------------------------------------------
class SuiteLoadError(RuntimeError):
    """A suite, adapter or dataset could not be resolved.

    Raised in place of the bare SystemExit that load_suite(), read_items()
    and build_suite() still raise -- see the module docstring. A future
    router catches this and turns it into an HTTP 4xx/5xx; nothing upstream
    of _guard() ever sees a SystemExit.
    """


def _guard(fn, *args, **kwargs):
    """Call a helper that may still raise bare SystemExit, and convert it.

    The only sanctioned way anything in this module reaches load_suite(),
    read_items() or build_suite().
    """
    try:
        return fn(*args, **kwargs)
    except SystemExit as exc:
        raise SuiteLoadError(str(exc)) from exc


def _slice_bounds(text: str, total: int) -> tuple[int, int]:
    """Parse FROM:TO, where either side may be empty."""
    if ':' not in text:
        raise SuiteLoadError("--slice takes FROM:TO, e.g. 0:6 or 12:")
    lo_s, hi_s = text.split(':', 1)
    try:
        lo = int(lo_s) if lo_s.strip() else 0
        hi = int(hi_s) if hi_s.strip() else total
    except ValueError:
        raise SuiteLoadError(f"--slice '{text}' is not FROM:TO") from None
    if lo < 0 or hi < lo:
        raise SuiteLoadError(f"--slice '{text}' is not a forward range")
    return lo, min(hi, total)


# ---------------------------------------------------------------------------
# system prompts
#
# Named files under benchmarks/data/prompts/, not text in a request body: a
# run records which prompt it used, and a name is only worth recording if the
# bytes behind it can be recovered later. The sha travels beside the name for
# the case the name cannot cover -- a file edited in place is a different
# prompt under the same name, and grouping by name alone would average the
# two. Resolved the same way ADAPTERS_DIR/SUITES_DIR are: a package-relative
# path under benchmarks/data/.
#
# Nothing here is a test item and nothing here is graded. Ground truth still
# comes only from the published datasets.
# ---------------------------------------------------------------------------
def system_names() -> list[str]:
    if not PROMPTS_DIR.is_dir():
        return []
    return sorted(p.stem for p in PROMPTS_DIR.glob('*.txt'))


def load_system(name: str | None) -> dict:
    """Resolve a system-prompt name to {name, text, sha}, or {} for none.

    The text is used verbatim, trailing newline stripped and nothing else
    touched: a system prompt is bytes the model sees, so normalising
    whitespace here would mean the recorded sha did not describe what was
    sent.
    """
    if not name:
        return {}
    path = PROMPTS_DIR / f'{name}.txt'
    if not path.is_file():
        known = ', '.join(system_names()) or 'none defined'
        raise SuiteLoadError(f"no system prompt '{name}' in {PROMPTS_DIR} (known: {known})")
    text = path.read_text(encoding='utf-8').rstrip('\n')
    if not text.strip():
        raise SuiteLoadError(f"system prompt '{name}' is empty")
    return {'name': name, 'text': text, 'sha': hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]}


# ---------------------------------------------------------------------------
# request
# ---------------------------------------------------------------------------
def build_body(prompt: str, model: str, prof: dict, stream: bool, system: dict | None = None) -> dict:
    messages = []
    if system and system.get('text'):
        messages.append({'role': 'system', 'content': system['text']})
    messages.append({'role': 'user', 'content': prompt})
    body = {
        'model': model,
        'messages': messages,
        'max_tokens': int(os.environ.get('LLAMA_TEST_MAX_TOKENS', 2048)),
        'temperature': 0,
        'cache_prompt': os.environ.get('LLAMA_TEST_CACHE_PROMPT', '0') == '1',
    }
    # reasoning_effort only reaches models whose template actually serves it;
    # sending it elsewhere would silently mean nothing while looking configured.
    if 'reasoning_effort' in ' '.join(prof.get('extra', [])):
        body['chat_template_kwargs'] = {'reasoning_effort': os.environ.get('LLAMA_REASONING', 'medium')}
    if stream:
        body['stream'] = True
    return body


PARAM_KEYS = ('temperature', 'top_p', 'top_k', 'max_tokens', 'cache_prompt', 'stream', 'chat_template_kwargs')


def system_of(body: dict) -> str:
    """The system message actually in the body, or "".

    Read back out of the request rather than carried alongside it, for the
    same reason params_of is: what gets recorded should be what was sent.
    """
    for message in body.get('messages', []):
        if message.get('role') == 'system':
            return message.get('content', '')
    return ''


def params_of(body: dict) -> dict:
    """What the log should record: read back out of the body, not re-derived."""
    params = {k: body[k] for k in PARAM_KEYS if body.get(k) is not None}
    text = system_of(body)
    if text:
        params['system_sha'] = hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]
    return params


class ServerGone(RuntimeError):
    """The server stopped answering, so there is nothing to grade.

    Raised instead of recording an outcome. A benchmark result is a statement
    about a model's answer, and a connection refused is not one: recording it
    as fail_error files a serving failure as a model failure, and because
    (suite_run_id, benchmark, item_id) is unique, a resume then skips the item
    forever. Raising leaves the item genuinely undone.
    """


class Answer:
    def __init__(self):
        self.content = ''
        self.reasoning = ''
        self.timings: dict = {}
        self.wall_ms = 0.0
        self.error = ''
        # True when the request never produced a completion because the
        # connection failed, as opposed to the model answering badly.
        self.gone = False


async def ask(
    session: aiohttp.ClientSession,
    prompt: str,
    model: str,
    prof: dict,
    port: int,
    *,
    stream: bool = True,
    system: dict | None = None,
) -> Answer:
    """One chat completion against the serving llama.cpp server. Streams by default."""
    body = build_body(prompt, model, prof, stream, system)
    timeout = aiohttp.ClientTimeout(total=float(os.environ.get('LLAMA_TEST_TIMEOUT', 900)))
    url = f'http://127.0.0.1:{port}/v1/chat/completions'

    out = Answer()
    t0 = time.monotonic()
    try:
        async with session.post(url, json=body, timeout=timeout) as resp:
            if resp.status >= 400:
                text = (await resp.text())[:400]
                out.error = f'HTTP {resp.status}: {text}'
            elif stream:
                await _read_sse(resp, out)
            else:
                payload = await resp.json(content_type=None)
                message = (payload.get('choices') or [{}])[0].get('message', {})
                out.content = message.get('content') or ''
                out.reasoning = message.get('reasoning_content') or ''
                out.timings = payload.get('timings') or {}
    except asyncio.TimeoutError:
        # No completion arrived within LLAMA_TEST_TIMEOUT.
        out.gone = True
        out.error = f'no response from port {port} (timed out); is llama-serve running?'
    except aiohttp.ClientError as exc:
        # Connection refused, and also a stream that died mid-read: in both
        # cases no completion arrived. A non-2xx status is handled above and
        # deliberately does not land here -- the server answered, and a 400
        # can be item-specific (a prompt past the context window), which is a
        # per-item fact worth recording rather than a serving failure.
        out.gone = True
        out.error = f'no response from port {port} ({exc}); is llama-serve running?'
    except (ValueError, TypeError) as exc:
        out.error = f'unparseable response: {exc}'
    out.wall_ms = (time.monotonic() - t0) * 1000
    return out


async def _read_sse(resp: aiohttp.ClientResponse, out: Answer) -> None:
    """Split an OpenAI SSE stream into reasoning_content (thinking) and content.

    Ported from the CLI's terminal-rendering _read_sse(): the answer/thinking
    split is kept, but there is no terminal on the other end of an HTTP
    request, so the live stdout/stderr rendering is dropped. A future
    streaming router endpoint decides its own wire format for pushing tokens
    onward; this function only decides how to classify them.
    """
    async for raw in resp.content:
        line = raw.decode('utf-8', 'replace').strip()
        if not line.startswith('data: '):
            continue
        payload = line[6:]
        if payload == '[DONE]':
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue

        if isinstance(chunk.get('timings'), dict):
            out.timings = chunk['timings']
        delta = (chunk.get('choices') or [{}])[0].get('delta') or {}

        reasoning = delta.get('reasoning_content')
        if reasoning:
            out.reasoning += reasoning

        content = delta.get('content')
        if content:
            out.content += content


# ---------------------------------------------------------------------------
# running items
# ---------------------------------------------------------------------------
async def run_item(session: aiohttp.ClientSession, row: dict, ctx: dict) -> dict:
    """Ask one item, grade it, record it. Returns the stored record.

    One transaction writes the request timings, the verdict and the answer
    (BenchmarkResults.add_result()), committed before the next item is asked.
    That is what makes an interrupted suite a partial result rather than no
    result.
    """
    adapter, item = row['adapter'], row['item']
    prompt = render_prompt(adapter, item)

    ans = await ask(
        session, prompt, ctx['model'], ctx['profile'], ctx['port'], stream=ctx['stream'], system=ctx.get('system')
    )

    # Before anything is written: a dead server is not a verdict about this
    # item, and writing one would both understate the model and, via the
    # uniqueness constraint, make the item unresumable.
    if ans.gone:
        raise ServerGone(ans.error)

    if ans.error:
        outcome, reason = 'fail_error', ans.error
    else:
        outcome, reason = grade(adapter, item, ans.content)

    record = {
        'suite_run_id': ctx['suite_run_id'],
        'run_id': ctx['run_id'],
        'at': int(time.time()),
        'model': ctx['model'],
        'config_id': ctx['config_id'],
        'profile': ctx['profile'].get('name', ''),
        'benchmark': row['benchmark'],
        'dataset_revision': ctx['revisions'].get(row['benchmark'], ''),
        'tier': ctx['tier'],
        'seed': ctx['seed'],
        'item_id': row['item_id'],
        'outcome': outcome,
        'reason': reason[:400],
        'wall_ms': round(ans.wall_ms),
        'timings': ans.timings,
        'params': ctx['params'],
        # Empty for no system prompt; add_result stores that as NULL, which
        # the schema defines as "none sent" rather than "unknown".
        'system_name': (ctx.get('system') or {}).get('name', ''),
        'system_sha': (ctx.get('system') or {}).get('sha', ''),
        # The wrapper this repo puts around the published item. dataset_revision
        # covers the item; this covers the prompt_template and grading config
        # that turn it into a request.
        'adapter_sha': adapter.get('_sha', ''),
        'reasoning_chars': len(ans.reasoning),
    }
    # The answer is stored as three fields rather than one rendered blob, so
    # "the failures whose reasoning ran past 15k characters" is a query
    # instead of a grep over rendered files.
    answer = None if ans.error else {'prompt': prompt, 'content': ans.content, 'reasoning': ans.reasoning}
    record['result_id'] = await BenchmarkResults.add_result(record, answer)
    return record


RecordCallback = Optional[Callable[[dict], Union[None, Awaitable[None]]]]
StopCallback = Optional[Callable[[], Union[bool, Awaitable[bool]]]]


async def _maybe_await(value):
    return await value if asyncio.iscoroutine(value) else value


async def run_items(
    session: aiohttp.ClientSession,
    rows: list[dict],
    ctx: dict,
    *,
    results: list[dict] | None = None,
    on_record: RecordCallback = None,
    should_stop: StopCallback = None,
) -> list[dict]:
    """Run graded items against the server, one committed transaction each.

    The shared execution loop: a future suite-run router endpoint and a
    future tuning sweep both drive the same todo list through this, rather
    than each re-implementing what "run a graded item" means.

    Exceptions (ServerGone in particular) propagate rather than being handled
    here, because different callers say different things about them -- a
    suite-run endpoint reports how far it got before the server died, a sweep
    closes the visit and moves to the next candidate. `results` is passed in
    and appended to for the same reason: on ServerGone the caller still needs
    what was measured before it, and a return value would have been lost with
    the stack.

    on_record fires with each record as it lands (a live progress push, or a
    tuner watching for a mid-visit throughput collapse); should_stop is
    awaited/called before each item and ends the loop cleanly when it answers
    true (a cancel request, a spent budget). Both may be sync or async
    callables, since a router endpoint's callback will usually need to await
    something (writing to an SSE queue) while a synchronous caller should not
    be forced to wrap a plain function.
    """
    if results is None:
        results = []
    for row in rows:
        if should_stop is not None and await _maybe_await(should_stop()):
            break
        record = await run_item(session, row, ctx)
        results.append(record)
        if on_record is not None:
            await _maybe_await(on_record(record))
    return results


# ---------------------------------------------------------------------------
# resolving a suite run
# ---------------------------------------------------------------------------
async def context(
    session: aiohttp.ClientSession,
    *,
    profile_name: str | None,
    tier: str,
    seed: int,
    benches: list[str],
    system_name: str | None = None,
    suite_run_id: str | None = None,
) -> dict:
    """Everything a run of items is measured under, resolved once.

    Two ids, and they are not the same thing. `suite_run_id` names this suite
    run and is what a resume continues; `run_id` names the serving run the
    telemetry recorder has open, and is None for a server someone started by
    hand. A result carries both, plus a denormalised config_id so it keeps
    its configuration identity even if its run row is later pruned.
    """
    prof = await env_profile.profile(profile_name)
    port = env_profile.port(prof)
    model = await env_profile.served_model(session, port, prof)
    active = await BenchmarkRuns.get_active_run(port)
    stream = os.environ.get('LLAMA_TEST_STREAM', '1') != '0'
    system = load_system(system_name)
    body = build_body('', model, prof, stream, system)

    revisions = {}
    for b in benches:
        manifest = read_manifest(b) or {}
        revisions[b] = manifest.get('revision') or manifest.get('content_sha256', '')

    return {
        'suite_run_id': suite_run_id or f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:6]}",
        'run_id': active.run_id if active else None,
        'model': model,
        'profile': prof,
        'port': port,
        'stream': stream,
        # None, not "unrecorded": the column is a foreign key into config,
        # and the display layer is where a missing configuration gets a name.
        'config_id': active.config_id if active else None,
        'tier': tier,
        'seed': seed,
        'revisions': revisions,
        'system': system,
        'params': params_of(body),
    }


async def prepare_suite(
    session: aiohttp.ClientSession,
    *,
    suite_name: str,
    profile_name: str | None = None,
    benchmark: str | None = None,
    item_slice: str | None = None,
    system_name: str | None = None,
    resume: bool = False,
    suite_run_id: str | None = None,
) -> tuple[dict, list[dict], list[dict]]:
    """Resolve a suite into (ctx, todo, skipped). No items are run.

    Mirrors the original CLI's prepare_suite() exactly, so a router endpoint
    resolves the same todo list (interleaving, slice, missing-library
    exclusion, calibration warnings, resume diffing) that the original
    dashboard already relied on this function to produce -- a second
    implementation of this setup would eventually disagree with this one
    about what a tier actually contains.
    """
    adapters = load_adapters()
    suite = _guard(load_suite, suite_name)
    selected, skipped = _guard(build_suite, suite, adapters, only=benchmark)
    if not selected:
        raise SuiteLoadError(f"suite '{suite_name}' selected no items")

    # One canonical order for a tier: round-robin across benchmarks, each in
    # its sampled sequence. A slice cuts this list, so a partial run can be
    # reproduced later and land on the same items.
    selected = interleave(selected)
    if item_slice:
        lo, hi = _slice_bounds(item_slice, len(selected))
        selected = selected[lo:hi]
        if not selected:
            raise SuiteLoadError(f'--slice {item_slice} selected no items')

    # Checked once up front: importing pandas costs about a second, and a
    # full DS-1000 run would otherwise pay it hundreds of times to learn the
    # same fact.
    blocked = missing_libraries(selected)
    if blocked:
        selected = [r for r in selected if f"{r['benchmark']}/{r['item_id']}" not in blocked]
        for key, why in blocked.items():
            b, iid = key.split('/', 1)
            skipped.append({'benchmark': b, 'item_id': iid, 'reason': why})

    stale: dict[str, str] = {}
    for name in sorted({r['benchmark'] for r in selected}):
        why = calibration_stale(name)
        if why:
            stale[name] = why
            log.warning(
                '%s is not calibrated for this environment (%s); items its own '
                'reference solutions cannot pass here will be scored as failures.',
                name,
                why,
            )

    ctx = await context(
        session,
        profile_name=profile_name,
        tier=suite['id'],
        seed=int(suite.get('seed', 0)),
        benches=sorted({r['benchmark'] for r in selected}),
        system_name=system_name,
        suite_run_id=suite_run_id,
    )

    if resume:
        prior = await BenchmarkResults.latest_suite_run(model=ctx['model'], tier=suite['id'])
        if prior:
            ctx['suite_run_id'] = prior

    done = await BenchmarkResults.completed(ctx['suite_run_id'])
    todo = [r for r in selected if (r['benchmark'], r['item_id']) not in done]

    # Exclusions are recorded, not merely counted -- but once, against the
    # benchmark and dataset revision they belong to, rather than as one row
    # per item per run. Whether an item is outside the filter, or
    # ungradeable on this box, is a property of the benchmark and this
    # environment; nothing about a serving run changes it.
    await BenchmarkSuiteExclusions.record_exclusions(
        [
            {
                'benchmark': s['benchmark'],
                'item_id': s['item_id'],
                'dataset_revision': ctx['revisions'].get(s['benchmark'], ''),
                'kind': s.get('kind', 'unattemptable'),
                'reason': s['reason'],
            }
            for s in skipped
        ]
    )

    return ctx, todo, skipped
