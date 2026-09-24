"""Prompt: a template with typed slots and strict rendering."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .errors import PromptError

_SLOT = re.compile(r"\{(\w+)\}")


@dataclass
class Prompt:
    name: str
    template: str
    version: str = "1.0.0"
    slots: dict = field(default_factory=dict)   # slot -> type name ("str" | "int" | "float")

    def __post_init__(self) -> None:
        found = set(_SLOT.findall(self.template))
        declared = set(self.slots)
        if found != declared:
            missing = sorted(found - declared)
            extra = sorted(declared - found)
            raise PromptError(f"template/slot mismatch in {self.name}: "
                              f"undeclared {missing}, unused {extra}")

    def render(self, **kwargs) -> str:
        for name, typ in self.slots.items():
            if name not in kwargs:
                raise PromptError(f"{self.name}: missing slot {name!r}")
            value = kwargs[name]
            if typ == "int" and not isinstance(value, int):
                raise PromptError(f"{self.name}: slot {name!r} must be int")
            if typ == "float" and not isinstance(value, (int, float)):
                raise PromptError(f"{self.name}: slot {name!r} must be float")
            if typ == "str" and not isinstance(value, str):
                raise PromptError(f"{self.name}: slot {name!r} must be str")
        return _SLOT.sub(lambda m: str(kwargs[m.group(1)]), self.template)
