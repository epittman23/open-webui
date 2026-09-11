"""benchmarks/proc.py - one llama-env.sh invocation, owning its process group.

Async port of local-llm's scripts/llama_proc.py (outer repo), so the backend
can launch and manage llama-server (Serve) and many concurrent tuning
candidates (Tune) without blocking the event loop. The process-group and
signal semantics are unchanged from the original -- only the subprocess
plumbing is now asyncio's.

llama-env.sh itself stays outside this app: it is the outer local-llm repo's
own operational shell layer (serving profiles, GPU flags), never vendored
into the fork. LLAMA_ENV_SH must point at it on this host.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import signal
from pathlib import Path


def _env_sh() -> Path:
    configured = os.environ.get('LLAMA_ENV_SH')
    if not configured:
        raise RuntimeError(
            'LLAMA_ENV_SH is not set. It must point at the outer local-llm '
            "repo's scripts/llama-env.sh -- the serving-profile shell layer "
            'this app spawns llama-server through.'
        )
    return Path(configured)


class Command:
    """One llama-env.sh invocation, owning its process group.

    `start_new_session=True` is the whole point. The command runs through
    `bash -c "source llama-env.sh && ..."`, so llama-server (or a tuning
    candidate's server) is a child of that bash and not the process a caller
    holds a handle to -- `proc.terminate()` alone would leave it holding
    VRAM. A session of its own means the whole tree can be signalled
    together.

    The signals are chosen for what the harness does with them, not for
    politeness:

    - SIGINT is a *clean, resumable* cancel, matching the CLI original's
      cooperative-cancel behaviour for a test suite in progress.
    - SIGTERM before SIGKILL protects the store: the telemetry recorder
      calls close_run() on the way out, and killing outright leaves the run
      open for a stale-run sweep to close as 'stale' instead of 'clean'.
    """

    def __init__(self, command: str, *, env: dict | None = None, plain: bool = True):
        self.command = command
        self.proc: asyncio.subprocess.Process | None = None
        self.env = dict(env or os.environ)
        if plain:
            # Output going into an SSE stream does its own rendering; Rich's
            # escape codes would be shown raw, not applied.
            self.env['LLAMA_PLAIN'] = '1'

    async def start(self) -> None:
        self.proc = await asyncio.create_subprocess_exec(
            'bash',
            '-c',
            f'source {shlex.quote(str(_env_sh()))} && {self.command}',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=self.env,
            start_new_session=True,
        )

    async def lines(self):
        """Stream stdout line by line. One pass, safe to iterate once."""
        if self.proc is None or self.proc.stdout is None:
            return
        async for raw in self.proc.stdout:
            yield raw.decode(errors='replace').rstrip()

    async def wait(self) -> int:
        return await self.proc.wait() if self.proc else -1

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    def signal(self, sig: int) -> bool:
        """Signal the whole process group. False if there was nothing to signal."""
        if not self.running:
            return False
        try:
            os.killpg(os.getpgid(self.proc.pid), sig)
            return True
        except (ProcessLookupError, PermissionError, OSError):
            return False

    def interrupt(self) -> bool:
        """Cancel, the way Ctrl-C at a shell would: resumable and recorded."""
        return self.signal(signal.SIGINT)

    async def stop(self, grace: float = 5.0) -> bool:
        """Ask the group to stop, then insist."""
        if not self.signal(signal.SIGTERM):
            return False
        try:
            await asyncio.wait_for(self.proc.wait(), timeout=grace)
        except asyncio.TimeoutError:
            self.signal(signal.SIGKILL)
        return True
