"""Run candidate programs in a separate Python process and score them.

This is process isolation, not a security boundary. Each run gets a fresh temporary
directory as its working directory and ``HOME``, an environment without the parent's
variables (so API keys are not visible to candidate code), ``python -I``, a wall-clock
timeout that kills the whole process group, and on POSIX CPU and memory rlimits. The
candidate can still read the filesystem and open network connections: run untrusted
or LLM-written code inside a container or VM when that matters.

``PythonEvaluator`` runs the evaluation script in the same process as the candidate,
so a candidate can tamper with its own score (reward hacking). Results are reported on
a line tagged with a per-run nonce, which stops accidental spoofing only. Inspect the
best candidates before trusting their scores.
"""

from __future__ import annotations

import json
import math
import os
import secrets
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from adalflow.optim.evolution.types import Evaluation, to_evaluation

__all__ = ["SandboxResult", "run_python", "PythonEvaluator"]

_POSIX = os.name == "posix"
_MARK_ENV = "ADALFLOW_EVAL_MARK"


@dataclass
class SandboxResult:
    returncode: Optional[int]
    stdout: str
    stderr: str
    timed_out: bool
    duration_s: float


def _clean_env(extra: Optional[Dict[str, str]], home: str) -> Dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", os.defpath),
        "HOME": home,
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if os.name == "nt" and "SYSTEMROOT" in os.environ:
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    env.update(extra or {})
    return env


def _limits(memory_mb: Optional[int], cpu_s: int):
    def apply() -> None:  # runs in the child between fork and exec
        import resource

        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
            if memory_mb:
                limit = memory_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        except (ValueError, OSError):  # e.g. RLIMIT_AS unsupported on macOS
            pass

    return apply


def _text(value) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else value


def run_python(
    source: str,
    *,
    files: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
    memory_mb: Optional[int] = 2048,
    env: Optional[Dict[str, str]] = None,
    python: str = sys.executable,
) -> SandboxResult:
    """Runs ``source`` as ``main.py`` in a fresh directory next to ``files``."""
    with tempfile.TemporaryDirectory(prefix="adalflow-sandbox-") as tmp:
        for name, text in (files or {}).items():
            path = Path(tmp) / name
            if path.resolve().parent != Path(tmp).resolve():
                raise ValueError(f"sandbox file names must be plain names: {name!r}")
            path.write_text(text, encoding="utf-8")
        (Path(tmp) / "main.py").write_text(source, encoding="utf-8")

        kwargs = {}
        if _POSIX:
            kwargs = dict(
                start_new_session=True,
                preexec_fn=_limits(memory_mb, int(math.ceil(timeout)) + 1),
            )
        started = time.monotonic()
        proc = subprocess.Popen(
            [python, "-I", "main.py"],
            cwd=tmp,
            env=_clean_env(env, tmp),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **kwargs,
        )
        timed_out = False
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            if _POSIX:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                proc.kill()
            out, err = proc.communicate()
        return SandboxResult(
            returncode=None if timed_out else proc.returncode,
            stdout=_text(out),
            stderr=_text(err),
            timed_out=timed_out,
            duration_s=time.monotonic() - started,
        )


_WRAPPER = f"""
import json, os, sys, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_mark = os.environ.pop("{_MARK_ENV}")

def emit(payload):
    sys.stdout.flush()
    sys.stdout.write("\\n" + _mark + json.dumps(payload, default=str) + "\\n")
    sys.stdout.flush()

try:
    import candidate
except BaseException:
    emit({{"stage": "import", "traceback": traceback.format_exc(limit=5)}})
    sys.exit(0)
try:
    from evaluator import evaluate
    emit({{"stage": "ok", "result": evaluate(candidate)}})
except BaseException:
    emit({{"stage": "evaluate", "traceback": traceback.format_exc(limit=5)}})
"""


def _tail(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else "... " + text[-limit:]


class PythonEvaluator:
    """Scores a candidate Python module with a user evaluation script.

    The candidate text is saved as ``candidate.py``. ``evaluator_source`` must define
    ``evaluate(candidate)``, which receives the imported module and returns a score, a
    dict (``score``, ``instance_scores``, ``feedback``, extra metrics) or raises.
    Catch per-test exceptions inside ``evaluate`` and report them as instance
    failures with feedback; an exception that escapes marks the whole candidate as
    failed. The candidate's stderr is appended to the feedback.

    Args:
        evaluator_source: Python source of the evaluation script.
        timeout, memory_mb, env: forwarded to ``run_python``.
        feedback_chars: how much stderr/traceback text to keep.
    """

    def __init__(
        self,
        evaluator_source: str,
        *,
        timeout: float = 30.0,
        memory_mb: Optional[int] = 2048,
        env: Optional[Dict[str, str]] = None,
        feedback_chars: int = 4000,
    ) -> None:
        self.evaluator_source = evaluator_source
        self.timeout = timeout
        self.memory_mb = memory_mb
        self.env = env
        self.feedback_chars = feedback_chars

    def __call__(self, text: str) -> Evaluation:
        mark = f"__ADALFLOW_EVAL_{secrets.token_hex(8)}__"
        run = run_python(
            _WRAPPER,
            files={"candidate.py": text, "evaluator.py": self.evaluator_source},
            timeout=self.timeout,
            memory_mb=self.memory_mb,
            env={**(self.env or {}), _MARK_ENV: mark},
        )
        metrics = {"duration_s": round(run.duration_s, 3)}
        stderr = _tail(run.stderr, self.feedback_chars)
        if run.timed_out:
            return Evaluation(
                error=f"timed out after {self.timeout}s",
                feedback=f"The run was killed after {self.timeout}s.\n{stderr}".strip(),
                metrics=metrics,
            )
        marked = [ln for ln in run.stdout.splitlines() if ln.startswith(mark)]
        if not marked:
            return Evaluation(
                error=f"process exited with code {run.returncode} before reporting",
                feedback=stderr or _tail(run.stdout, self.feedback_chars),
                metrics=metrics,
            )
        payload = json.loads(marked[-1][len(mark) :])
        if payload["stage"] != "ok":
            where = (
                "importing the candidate"
                if payload["stage"] == "import"
                else "evaluate()"
            )
            trace = _tail(payload.get("traceback", ""), self.feedback_chars)
            return Evaluation(
                error=f"exception while {where}",
                feedback=trace,
                metrics=metrics,
            )
        try:
            evaluation = to_evaluation(payload["result"])
        except (TypeError, ValueError) as exc:
            return Evaluation(error=f"bad evaluator result: {exc}", metrics=metrics)
        if stderr:
            evaluation.feedback = f"{evaluation.feedback}\n[stderr]\n{stderr}".strip()
        evaluation.metrics = {**metrics, **evaluation.metrics}
        return evaluation
