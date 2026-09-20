"""Hydrogen Gateway: a resilient multi-provider LLM gateway.

One OpenAI-compatible endpoint in front of several upstream providers, with
priority failover, per-provider circuit breakers, retries with exponential
backoff, an exact-match TTL cache, per-key token-bucket rate limits, USD
budgets, and a live dashboard.
"""

from .app import app_from_env, build_app
from .config import GatewayConfig, GatewayKey, Provider

__all__ = ["GatewayConfig", "GatewayKey", "Provider", "app_from_env", "build_app"]
__version__ = "1.0.0"
