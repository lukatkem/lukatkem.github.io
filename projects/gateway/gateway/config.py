"""Configuration objects for the gateway: providers, keys, prices.

Configuration can be built two ways:

* Directly (tests / embedding): construct the dataclasses and pass them to
  :func:`gateway.app.build_app`.
* From the environment (deployment): :meth:`GatewayConfig.from_env` reads
  ``GATEWAY_PROVIDERS``, ``GATEWAY_KEYS``, ``GATEWAY_PRICES`` and
  ``GATEWAY_ALLOW_ANON``.

No real credentials ever live in this package:

* Provider API keys are referenced by *environment variable name*
  (``Provider.api_key_env``) and resolved at call time.
* Gateway client keys are stored as SHA-256 hex digests of the full token,
  matched by a distinctive prefix.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = [
    "ConfigError",
    "DEFAULT_PROVIDERS_JSON",
    "GatewayConfig",
    "GatewayKey",
    "PriceBook",
    "PriceSpec",
    "Provider",
    "parse_keys",
    "parse_prices",
    "parse_providers",
]


class ConfigError(ValueError):
    """Raised when a configuration document is malformed."""


@dataclass(frozen=True)
class Provider:
    """One upstream OpenAI-compatible provider."""

    name: str
    base_url: str
    api_key_env: str
    models: tuple[str, ...]
    priority: int = 1

    def serves(self, model: str) -> bool:
        """Whether this provider can serve ``model``."""
        return model in self.models


@dataclass(frozen=True)
class GatewayKey:
    """A gateway client key: matched by prefix, verified by SHA-256 digest."""

    sha256: str
    plan: str = "free"
    usd_budget: float | None = None  # None = unlimited


@dataclass(frozen=True)
class PriceSpec:
    """USD price per 1,000 tokens (prompt / completion)."""

    prompt_per_1k: float = 0.002
    completion_per_1k: float = 0.006


class PriceBook:
    """Provider-indexed price table with a fallback default price."""

    def __init__(
        self,
        by_provider: Mapping[str, PriceSpec] | None = None,
        default: PriceSpec | None = None,
    ) -> None:
        self._by_provider: dict[str, PriceSpec] = dict(by_provider or {})
        self._default = default or PriceSpec()

    def cost(self, provider_name: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimated USD cost of one response from ``provider_name``."""
        spec = self._by_provider.get(provider_name, self._default)
        return (prompt_tokens / 1000.0) * spec.prompt_per_1k + (completion_tokens / 1000.0) * spec.completion_per_1k


DEFAULT_PROVIDERS_JSON = """[
  {"name": "frontier", "base_url": "https://api.frontier.example/v1",
   "api_key_env": "FRONTIER_API_KEY", "models": ["m1", "m2"], "priority": 1},
  {"name": "cheap", "base_url": "https://api.cheap.example/v1",
   "api_key_env": "CHEAP_API_KEY", "models": ["m1", "m2", "m3"], "priority": 2}
]"""

_HEX_DIGITS = set("0123456789abcdef")


@dataclass(frozen=True)
class GatewayConfig:
    """Everything the gateway needs, parsed from env or built directly."""

    providers: tuple[Provider, ...]
    keys: Mapping[str, GatewayKey] = field(default_factory=dict)
    prices: PriceBook = field(default_factory=PriceBook)
    allow_anon: bool = False

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "GatewayConfig":
        """Parse configuration from ``env`` (default: ``os.environ``)."""
        env_map: Mapping[str, str] = os.environ if env is None else env
        return cls(
            providers=tuple(parse_providers(_load_json(env_map, "GATEWAY_PROVIDERS", DEFAULT_PROVIDERS_JSON))),
            keys=parse_keys(_load_json(env_map, "GATEWAY_KEYS", "{}")),
            prices=parse_prices(_load_json(env_map, "GATEWAY_PRICES", "{}")),
            allow_anon=env_map.get("GATEWAY_ALLOW_ANON", "").strip().lower() in {"1", "true", "yes"},
        )


def _load_json(env: Mapping[str, str], name: str, default: str) -> Any:
    raw = env.get(name) or default
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{name} is not valid JSON: {exc}") from None


def parse_providers(document: Any) -> list[Provider]:
    """Parse the provider roster (a JSON list) into :class:`Provider` objects."""
    if not isinstance(document, list) or not document:
        raise ConfigError("provider roster must be a non-empty JSON list")
    providers: list[Provider] = []
    seen: set[str] = set()
    for index, raw in enumerate(document):
        if not isinstance(raw, dict):
            raise ConfigError(f"provider #{index} must be a JSON object")
        missing = {"name", "base_url", "api_key_env", "models"} - set(raw)
        if missing:
            raise ConfigError(f"provider #{index} is missing fields: {sorted(missing)}")
        name = str(raw["name"])
        if not name or name in seen:
            raise ConfigError(f"provider #{index} has an empty or duplicate name: {name!r}")
        base_url = str(raw["base_url"])
        if not base_url.startswith(("http://", "https://")):
            raise ConfigError(f"provider {name!r} base_url must start with http(s)://")
        models = tuple(str(model) for model in raw["models"])
        if not models:
            raise ConfigError(f"provider {name!r} must list at least one model")
        providers.append(
            Provider(
                name=name,
                base_url=base_url,
                api_key_env=str(raw["api_key_env"]),
                models=models,
                priority=int(raw.get("priority", 1)),
            )
        )
        seen.add(name)
    return providers


def parse_keys(document: Any) -> dict[str, GatewayKey]:
    """Parse gateway keys: ``{"<prefix>": {"sha256": "...", "plan": ..., "usd_budget": ...}}``."""
    if document is None or document == {}:
        return {}
    if not isinstance(document, dict):
        raise ConfigError("gateway keys must be a JSON object of prefix -> key record")
    keys: dict[str, GatewayKey] = {}
    for prefix, raw in document.items():
        if not isinstance(prefix, str) or not prefix:
            raise ConfigError("gateway key prefixes must be non-empty strings")
        if not isinstance(raw, dict) or "sha256" not in raw:
            raise ConfigError(f"gateway key {prefix!r} must be an object with a 'sha256' field")
        digest = str(raw["sha256"]).strip().lower()
        if len(digest) != 64 or not set(digest) <= _HEX_DIGITS:
            raise ConfigError(f"gateway key {prefix!r} sha256 must be 64 hex characters")
        budget = raw.get("usd_budget")
        try:
            parsed_budget = None if budget is None else float(budget)
        except (TypeError, ValueError):
            raise ConfigError(f"gateway key {prefix!r} usd_budget must be a number") from None
        keys[prefix] = GatewayKey(sha256=digest, plan=str(raw.get("plan", "free")), usd_budget=parsed_budget)
    return keys


def parse_prices(document: Any) -> PriceBook:
    """Parse prices: ``{"<provider|default>": {"prompt_per_1k": x, "completion_per_1k": y}}``."""
    if not document:
        return PriceBook()
    if not isinstance(document, dict):
        raise ConfigError("prices must be a JSON object of provider -> price spec")
    default = PriceSpec()
    by_provider: dict[str, PriceSpec] = {}
    for name, raw in document.items():
        if not isinstance(raw, dict) or "prompt_per_1k" not in raw or "completion_per_1k" not in raw:
            raise ConfigError(f"price entry {name!r} needs 'prompt_per_1k' and 'completion_per_1k'")
        try:
            spec = PriceSpec(prompt_per_1k=float(raw["prompt_per_1k"]), completion_per_1k=float(raw["completion_per_1k"]))
        except (TypeError, ValueError):
            raise ConfigError(f"price entry {name!r} values must be numbers") from None
        if name == "default":
            default = spec
        else:
            by_provider[str(name)] = spec
    return PriceBook(by_provider, default)
