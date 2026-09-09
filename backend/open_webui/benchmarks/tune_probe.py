"""benchmarks/tune_probe.py - serving one candidate, and reading the GPU with nothing served.

Ported from local-llm's scripts/llama_tune.py (outer repo). Split out of
benchmarks/tune.py (the round-elimination orchestrator) and
benchmarks/tune_schedule.py (budget/round arithmetic and the search space)
because this is the module's other clearly separable concern: the mechanics
of getting one candidate's `lllm-serve` up, watching it for a mid-visit
collapse, and reading `nvidia-smi` when nothing is served at all (which the
telemetry recorder cannot do, since it needs a served port).

`Server.start()`/`.wait()`/`.stop()` are the async port of the original's
`Server` class -- readiness is `GET /v1/models`, deliberately not a TCP
connect, because llama-server binds the port before it loads the model, and
a port check would start the clock on a candidate whose weights are still
being read.

Launching is pluggable through LLAMA_TUNE_LAUNCH so the whole algorithm can
be exercised against a stub with no GPU (see the outer repo's
llama_tune_check.py / llama_tune_stub.py). That is not a testing convenience
bolted on: fault injection -- refuse to bind, OOM, hang past the timeout, die
mid-visit, drop to 12% throughput -- is where this module's real defects
live, and none of them can be provoked on demand from real hardware.

Nothing here imports FastAPI, and nothing here writes to this app's
database -- Cooldown (in tune.py) is what turns a probe into a recorded
pause; this module only takes the readings.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shlex
import time
from collections import deque
from typing import TYPE_CHECKING

import aiohttp

from open_webui.benchmarks import stats
from open_webui.benchmarks.proc import Command
from open_webui.benchmarks.tune_schedule import median

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a real import cycle
    from open_webui.benchmarks.tune_schedule import Candidate

# ---------------------------------------------------------------------------
# serving one candidate
# ---------------------------------------------------------------------------
OOM_PATTERN = re.compile(r'out of memory|cudaMalloc|failed to allocate|ggml_backend_.*alloc.*failed', re.I)


#: Serving is not the only way to fail, and the four are not interchangeable.
#: An OOM prunes a region of the grid; a load error is a bug in the grid or
#: the build; a timeout may be a slow disk on a cold page cache. They are
#: recorded apart so a reader can tell "this configuration does not fit"
#: from "this configuration was never actually tried".
class Infeasible(RuntimeError):
    def __init__(self, kind: str, reason: str):
        super().__init__(reason)
        self.kind = kind
        self.reason = reason


def _tail_reason(text: str) -> str:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    return ' / '.join(lines[-3:])[:400] or 'no output'


class Server:
    """One `lllm-serve` under a candidate's overrides, and its readiness.

    Readiness is `GET /v1/models` over a shared aiohttp session (the same
    session a caller uses for the rest of a visit's requests), deliberately
    not a TCP connect.
    """

    def __init__(self, candidate: 'Candidate', port: int, *, load_timeout: float, session: aiohttp.ClientSession):
        self.candidate = candidate
        self.port = port
        self.load_timeout = load_timeout
        self.session = session
        self.cmd: Command | None = None
        self.load_ms: float | None = None
        self.tail: deque[str] = deque(maxlen=40)
        self._drain: asyncio.Task | None = None

    def _command(self) -> str:
        launch = os.environ.get('LLAMA_TUNE_LAUNCH')
        if launch:
            return f'{launch} {shlex.quote(self.candidate.profile)}'
        return f'lllm-serve {shlex.quote(self.candidate.profile)}'

    async def start(self) -> None:
        self.cmd = Command(self._command(), env=self.candidate.env(), plain=True)
        await self.cmd.start()
        # Drained on a background task rather than read at the end. The pipe
        # is 64 KiB and llama-server runs at -lv 4, so a server left unread
        # blocks on write partway through loading and the caller would wait
        # out the whole load timeout on a process that is alive and stuck.
        # The tail is kept for the OOM classification below.
        self._drain = asyncio.create_task(self._pump())

    async def _pump(self) -> None:
        try:
            async for line in self.cmd.lines():
                self.tail.append(line)
        except (ValueError, OSError):
            pass

    async def ready(self) -> bool:
        try:
            async with self.session.get(
                f'http://127.0.0.1:{self.port}/v1/models', timeout=aiohttp.ClientTimeout(total=3)
            ) as r:
                return r.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError, ValueError):
            return False

    async def wait(self) -> float:
        """Wait until the model answers, or raise Infeasible saying why."""
        started = time.monotonic()
        while time.monotonic() - started < self.load_timeout:
            if await self.ready():
                self.load_ms = (time.monotonic() - started) * 1000.0
                return self.load_ms
            if self.cmd is not None and not self.cmd.running:
                text = '\n'.join(self.tail)
                if OOM_PATTERN.search(text):
                    raise Infeasible('oom', _tail_reason(text))
                raise Infeasible('load_error', _tail_reason(text))
            await asyncio.sleep(1.0)
        await self.stop()
        raise Infeasible('load_timeout', f'no /v1/models within {self.load_timeout:.0f}s')

    def pgid(self) -> int | None:
        if self.cmd is None or self.cmd.proc is None:
            return None
        try:
            return os.getpgid(self.cmd.proc.pid)
        except (ProcessLookupError, OSError):
            return None

    async def stop(self) -> None:
        """SIGINT first, so the recorder closes its own run.

        `lllm-serve` backgrounds the telemetry recorder and signals it on
        the way out; killing the group outright leaves the run row open for
        a stale-run sweep to close as 'stale' instead of 'clean'. A sweep of
        eighty visits would then have the store claiming eighty servers
        crashed, and the one that genuinely did would be indistinguishable.
        """
        if self.cmd is None:
            return
        if self.cmd.running:
            self.cmd.interrupt()
            deadline = time.monotonic() + 20.0
            while self.cmd.running and time.monotonic() < deadline:
                await asyncio.sleep(0.5)
        if self.cmd.running:
            await self.cmd.stop(grace=10.0)
        if self._drain is not None:
            self._drain.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._drain


async def port_busy(port: int, session: aiohttp.ClientSession) -> bool:
    try:
        async with session.get(f'http://127.0.0.1:{port}/v1/models', timeout=aiohttp.ClientTimeout(total=3)) as r:
            return r.status == 200
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError, ValueError):
        return False


async def kill_pgid(pgid: int | None) -> bool:
    """Stop an orphan from an earlier, crashed sweep. Signals only its group."""
    if not pgid or pgid <= 1:
        return False
    import signal as _signal

    for sig in (_signal.SIGINT, _signal.SIGTERM, _signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            return sig is not _signal.SIGINT
        await asyncio.sleep(3.0)
    return True


def tps(timings: dict | None) -> float | None:
    """Generation tokens/second out of a request's own timings dict."""
    if not timings:
        return None
    n, ms = timings.get('predicted_n'), timings.get('predicted_ms')
    try:
        return (float(n) / float(ms)) * 1000.0 if n and ms else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


# ---------------------------------------------------------------------------
# cooldown primitives: reading the GPU with nothing served
# ---------------------------------------------------------------------------
class GpuProbe:
    """What nvidia-smi says with nothing served.

    The one place the tuner reads the GPU without the recorder, and it
    exists because the recorder needs a served port while the entire point
    of a pause is that nothing is served. The throttle mask is decoded
    through benchmarks.stats.throttle_reasons so a pause and a run name the
    same hardware state in the same words -- `SwPowerCap` in a pause table
    and `SwPowerCap` in a run summary have to be comparable by eye.
    """

    QUERY = 'temperature.gpu,power.draw,clocks_throttle_reasons.active'

    def __init__(self):
        self.binary = os.environ.get('LLAMA_TUNE_NVIDIA_SMI', 'nvidia-smi')

    async def read(self) -> dict | None:
        try:
            proc = await asyncio.create_subprocess_exec(
                self.binary,
                f'--query-gpu={self.QUERY}',
                '--format=csv,noheader,nounits',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError:
            return None
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=20)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return None
        if proc.returncode != 0 or not stdout.strip():
            return None
        text = stdout.decode(errors='replace')
        parts = [p.strip() for p in text.strip().splitlines()[0].split(',')]
        if len(parts) < 3:
            return None
        try:
            temp = float(parts[0])
            power = float(parts[1])
        except ValueError:
            return None
        raw = parts[2]
        try:
            mask = int(raw, 16) if raw.lower().startswith('0x') else int(raw)
        except ValueError:
            mask = 0
        return {'temp': temp, 'power': power, 'throttle': mask, 'reasons': stats.throttle_reasons([{'throttle': mask}])}


#: The bits that invalidate a measurement, as opposed to describing an idle
#: card. GpuIdle and the clock-setting bits are expected and are not faults;
#: the thermal and power ones are what a cooldown waits out. Named from
#: benchmarks.stats.THROTTLE_BITS rather than written as a hex literal, so a
#: bit renamed there does not leave a silently wrong mask here.
BAD_REASONS = {'SwPowerCap', 'HwSlowdown', 'SwThermalSlowdown', 'HwThermalSlowdown', 'HwPowerBrakeSlowdown'}
THERMAL_POWER_MASK = 0
for _bit, _name in stats.THROTTLE_BITS:
    if _name in BAD_REASONS:
        THERMAL_POWER_MASK |= _bit


def throttled(sample: dict | None) -> bool:
    return bool(sample and (sample['throttle'] & THERMAL_POWER_MASK))


class CollapseWatch:
    """Watch a visit for a mid-run fall that never lifts.

    Round-level drift control (see tune.Sweep.run_round()) compares one
    round's baseline against round 1's, which cannot see a cliff that
    happens *inside* a visit. So the running median over the last five
    requests is compared against the median of the visit's opening
    requests. A plain callable (not a coroutine): it does no I/O, and
    benchmarks.runner.run_items()'s on_record accepts a sync or async
    callback either way.
    """

    def __init__(self, ratio: float, window: int = 5, warmup: int = 4):
        self.ratio = ratio
        self.window = window
        self.warmup = warmup
        self.rates: list[float] = []
        self.opening: float | None = None
        self.collapsed = False

    def __call__(self, record: dict) -> None:
        rate = tps(record.get('timings'))
        if not rate:
            return
        self.rates.append(rate)
        if len(self.rates) == self.warmup:
            self.opening = median(self.rates)
        if self.opening and len(self.rates) >= self.warmup + self.window:
            recent = median(self.rates[-self.window :])
            if recent is not None and recent < self.opening * self.ratio:
                self.collapsed = True
