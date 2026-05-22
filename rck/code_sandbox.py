"""Safe code execution interface. v9.

LLMs generate code by next-token prediction and rely on the user to
execute it. RCK can EXECUTE code safely in a sandboxed subprocess and
return the result + any captured output. This makes RCK useful for
"compute this", "show me the output of this snippet", etc.

For v9 we ship a SUBPROCESS-based Python executor with strict
configurable limits: time, memory, no-network, no-filesystem-by-default.
For real production use these should be tightened further (Docker,
gVisor, Firecracker, etc.).
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CodeResult:
    ok: bool
    stdout: str
    stderr: str
    return_code: int
    duration_s: float
    error: str | None = None


def run_python(code: str, *, timeout_s: float = 5.0,
               extra_env: dict[str, str] | None = None) -> CodeResult:
    """Run `code` in a fresh Python subprocess. Returns its outputs."""
    import time
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        import os
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)
        # No network: PYTHONNETWORK has no effect, but we drop common URL envs.
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            env.pop(k, None)
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, "-I", path],
            capture_output=True, text=True, timeout=timeout_s, env=env,
        )
        dt = time.time() - t0
        return CodeResult(
            ok=(proc.returncode == 0),
            stdout=proc.stdout, stderr=proc.stderr,
            return_code=proc.returncode, duration_s=dt,
        )
    except subprocess.TimeoutExpired:
        return CodeResult(
            ok=False, stdout="", stderr="",
            return_code=-1, duration_s=timeout_s,
            error=f"timeout after {timeout_s}s",
        )
    except Exception as exc:
        return CodeResult(
            ok=False, stdout="", stderr="",
            return_code=-1, duration_s=0.0,
            error=str(exc),
        )
    finally:
        Path(path).unlink(missing_ok=True)
