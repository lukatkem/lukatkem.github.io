"""Academy — the classroom where frontier teachers train Hydro-1, from the UI.

Two pieces of state, both local-only:
  * .env next to this package — TEACHER_API_KEY / TEACHER_BASE_URL, written
    from the browser, applied to os.environ live (no server restart needed).
  * one background job at a time (distill or retrain) as a subprocess with a
    parsed log tail the UI polls.

Nothing here downloads a model: distillation ships prompts to the cloud
teacher and saves the returned TEXT (a few KB per story) as the student's
new textbook. Retraining runs the same trainer that made the base student.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
CORPUS_DIR = ROOT / "corpus"
CKPT = ROOT / "checkpoints"

# everything a distilled retrain produces — "back to base student" removes these
DISTILL_ARTIFACTS = [
    CKPT / "latest-distilled.pt",
    CKPT / "best-distilled.pt",
    CKPT / "history-distilled.json",
    CKPT / "tokenizer-distilled.json",
    CORPUS_DIR / "distilled.txt",
]

ENV_KEYS = ("TEACHER_API_KEY", "OPENROUTER_API_KEY", "NVIDIA_API_KEY",
            "TEACHER_BASE_URL", "TEACHER_MODELS", "TEACHER_TIMEOUT")


def key_env_for(key: str) -> str:
    """Route a pasted key to the right env var by its prefix."""
    k = key.strip()
    if k.startswith("sk-or-"):
        return "OPENROUTER_API_KEY"
    if k.startswith("nvapi-"):
        return "NVIDIA_API_KEY"
    return "TEACHER_API_KEY"


# ---------- .env: the only place the key is stored ----------

def load_env(path: Path = ENV_FILE) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            values[k.strip()] = v.strip().strip("'\"")
    return values


def apply_env(path: Path = ENV_FILE) -> None:
    """Push saved settings into os.environ — config reads them live."""
    for k, v in load_env(path).items():
        if k in ENV_KEYS and v:
            os.environ[k] = v


def save_key(key: str, base_url: str | None = None, path: Path = ENV_FILE) -> str:
    """Save the key under the env var its prefix implies; returns the var."""
    values = load_env(path)
    var = key_env_for(key)
    if key.strip():
        values[var] = key.strip()
    if base_url and base_url.strip() and var == "TEACHER_API_KEY":
        values["TEACHER_BASE_URL"] = base_url.strip().rstrip("/")
    path.write_text("\n".join(f"{k}={v}" for k, v in values.items()) + "\n")
    apply_env(path)
    return var


def clear_key(path: Path = ENV_FILE) -> None:
    values = load_env(path)
    for var in ("TEACHER_API_KEY", "OPENROUTER_API_KEY", "NVIDIA_API_KEY"):
        values.pop(var, None)
        os.environ.pop(var, None)
    path.write_text("\n".join(f"{k}={v}" for k, v in values.items()) + "\n")


# ---------- the one-job-at-a-time runner ----------

class Job:
    """A distill/train subprocess with live progress parsed from its log."""

    def __init__(self, kind: str, cmd: list[str], chain: bool = False):
        self.kind = kind
        self.cmd = cmd
        self.chain = chain  # distill → auto-start the retrain on success
        self.started_at = time.time()
        self.exit_code: int | None = None
        self.stage = "loading data" if kind == "train" else "starting"
        self.done = 0
        self.total = 0
        if kind == "train" and "--steps" in cmd:
            self.total = int(cmd[cmd.index("--steps") + 1])
        self.log: deque[str] = deque(maxlen=500)
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        self.proc = subprocess.Popen(
            cmd, cwd=ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        threading.Thread(target=self._pump, daemon=True).start()

    def alive(self) -> bool:
        return self.exit_code is None

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.rstrip()
            if line:
                self.log.append(line)
                self._parse(line)
        self.exit_code = self.proc.wait()
        if self.stage not in ("done",):
            self.stage = "failed" if self.exit_code else "done"
        if self.chain and self.kind == "distill" and self.exit_code == 0:
            self._chain_train()

    def _chain_train(self) -> None:
        """One-click pipeline: a successful distill flows straight into the
        retrain without another button press."""
        global _active
        corpus = CORPUS_DIR / "distilled.txt"
        if not corpus.exists() or corpus.stat().st_size < 2000:
            self.log.append("chain: distilled textbook too small — retrain not started")
            return
        self.log.append("chain: distill finished — starting retrain automatically")
        cmd = [sys.executable, "-m", "hydro1.train", "--corpus", str(corpus),
               "--steps", "8000", "--out-suffix", "distilled"]
        with _lock:
            _active = Job("train", cmd)

    def _parse(self, line: str) -> None:
        if self.kind == "distill":
            m = re.search(r"generated (\d+)/(\d+)", line)
            if m:
                self.stage, self.done, self.total = "writing stories", int(m.group(1)), int(m.group(2))
                return
            m = re.search(r"grading: (\d+) → (\d+)", line)
            if m:
                self.stage, self.done, self.total = "grading", int(m.group(2)), int(m.group(1))
                return
            if line.startswith("wrote "):
                self.stage = "done"
        else:
            m = re.search(r"step\s+(\d+)", line)
            if m and "loss" in line:
                self.stage, self.done = "training", int(m.group(1))
                return
            if line.startswith("done in"):
                self.stage, self.done = "done", self.total

    def snapshot(self) -> dict:
        return {
            "active": self.alive(),
            "kind": self.kind,
            "stage": self.stage,
            "done": self.done,
            "total": self.total,
            "exit_code": self.exit_code,
            "elapsed_s": round(time.time() - self.started_at),
            "log_tail": list(self.log)[-12:],
        }


_active: Job | None = None
_lock = threading.Lock()


def start(kind: str, stories: int = 200, steps: int = 8000, model: str | None = None, grade_model: str | None = None, chain: bool = False) -> dict:
    global _active
    with _lock:
        if _active is not None and _active.alive():
            return {"started": False, "reason": f"a {_active.kind} job is already running"}
        apply_env()  # a key saved moments ago reaches the subprocess through os.environ
        if kind == "distill":
            cmd = [sys.executable, "-m", "hydro1.distill",
                   "--stories", str(stories), "--grade", "--out", "distilled.txt"]
            if model:
                cmd += ["--model", model]
            if grade_model:
                cmd += ["--grade-model", grade_model]
        elif kind == "train":
            corpus = CORPUS_DIR / "distilled.txt"
            if not corpus.exists():
                return {"started": False, "reason": "no distilled textbook yet — run the distill step first"}
            cmd = [sys.executable, "-m", "hydro1.train", "--corpus", str(corpus),
                   "--steps", str(steps), "--out-suffix", "distilled"]
        else:
            return {"started": False, "reason": f"unknown job kind: {kind}"}
        _active = Job(kind, cmd, chain=chain)
        return {"started": True, "kind": kind}


def status() -> dict:
    return _active.snapshot() if _active is not None else {
        "active": False, "kind": None, "stage": None, "done": 0, "total": 0,
        "exit_code": None, "elapsed_s": 0, "log_tail": [],
    }


def stop() -> bool:
    j = _active
    if j is None or not j.alive():
        return False
    j.proc.terminate()
    try:
        j.proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        j.proc.kill()
    return True


def reset_distilled() -> list[str]:
    """Remove every distilled artifact so the playground fields the base student."""
    if _active is not None and _active.alive():
        raise RuntimeError("a job is running — stop it first")
    removed = []
    for p in DISTILL_ARTIFACTS:
        if p.exists():
            p.unlink()
            removed.append(p.name)
    return removed
