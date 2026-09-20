"""Hydro-1 configuration — teachers live in the cloud, configured via env.

Nothing is ever downloaded: teachers are remote endpoints. Keys are read
from os.environ on every call, so a key saved from the Academy UI applies
live without a restart. Supported key prefixes (auto-detected on save):

    OPENROUTER_API_KEY   sk-or-v1-…   → https://openrouter.ai/api/v1
    NVIDIA_API_KEY       nvapi-…      → https://integrate.api.nvidia.com/v1
    TEACHER_API_KEY      anything     → TEACHER_BASE_URL (default Z.AI)

TEACHER_MODELS (JSON) overrides the whole roster if you want custom models.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"


def _load_dotenv() -> None:
    """Load .env at import so CLI runs (distill/train/probes) see keys too.
    Existing os.environ values win — the server's live-applied keys stick."""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip("'\"")
        if k in {"TEACHER_API_KEY", "OPENROUTER_API_KEY", "NVIDIA_API_KEY",
                 "TEACHER_BASE_URL", "TEACHER_MODELS", "OLLAMA_BASE",
                 "TEACHER_TIMEOUT"} and v:
            os.environ.setdefault(k, v)


_load_dotenv()

TEACHER_TIMEOUT = float(os.environ.get("TEACHER_TIMEOUT", "120"))

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
NVIDIA_BASE = "https://integrate.api.nvidia.com/v1"
DEFAULT_TEACHER_BASE = "https://api.z.ai/api/paas/v4"

# The frontier roster, built from whichever provider keys are present.
# Every entry carries its own routing so teacher_chat never has to guess.
_ROSTER = {
    "OPENROUTER_API_KEY": [
        {"id": "z-ai/glm-5.3", "label": "GLM 5.3", "tier": "frontier",
         "provider": "OpenRouter", "base_url": OPENROUTER_BASE, "key_env": "OPENROUTER_API_KEY"},
        {"id": "z-ai/glm-5.3-flash", "label": "GLM 5.3 Flash", "tier": "fast",
         "provider": "OpenRouter", "base_url": OPENROUTER_BASE, "key_env": "OPENROUTER_API_KEY"},
        {"id": "z-ai/glm-5.2:free", "label": "GLM 5.2 (free)", "tier": "free",
         "provider": "OpenRouter", "base_url": OPENROUTER_BASE, "key_env": "OPENROUTER_API_KEY"},
    ],
    "NVIDIA_API_KEY": [
        {"id": "deepseek-ai/deepseek-v4-flash-0731", "label": "DeepSeek V4 Flash", "tier": "frontier",
         "provider": "NVIDIA", "base_url": NVIDIA_BASE, "key_env": "NVIDIA_API_KEY"},
    ],
    "TEACHER_API_KEY": [
        {"id": "glm-5.3", "label": "GLM 5.3 (direct)", "tier": "frontier",
         "provider": "custom", "base_url": None, "key_env": "TEACHER_API_KEY"},  # base_url resolved at call time
    ],
}


def cloud_teachers() -> list[dict]:
    raw = os.environ.get("TEACHER_MODELS")
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    teachers: list[dict] = []
    for key_env, entries in _ROSTER.items():
        if os.environ.get(key_env):
            for t in entries:
                t = dict(t)
                if t.get("base_url") is None:
                    t["base_url"] = os.environ.get("TEACHER_BASE_URL", DEFAULT_TEACHER_BASE)
                teachers.append(t)
    return teachers


def local_teachers() -> list[dict]:
    """Models already served by the local Ollama daemon — used as the
    always-available fallback teacher so the Arena works without any key."""
    return [{"id": "qwen2.5:7b", "label": "Qwen 2.5 7B (local)", "tier": "local"}]


def all_teachers() -> list[dict]:
    return list(local_teachers()) + cloud_teachers()


def key_set() -> bool:
    return any(os.environ.get(k) for k in ("TEACHER_API_KEY", "OPENROUTER_API_KEY", "NVIDIA_API_KEY"))


OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
