from collections.abc import Mapping
from typing import Any

from .base import BaseFunctionSpec


class FamilyRegistry:
    def __init__(self):
        self._families: dict[str, type[BaseFunctionSpec]] = {}

    def register(self, spec_cls: type[BaseFunctionSpec]) -> None:
        if not issubclass(spec_cls, BaseFunctionSpec):
            raise TypeError(f"{spec_cls!r} must inherit from BaseFunctionSpec")

        family = spec_cls.model_fields["family"].default
        if not isinstance(family, str) or not family:
            raise ValueError(f"{spec_cls.__name__}.family must be a non-empty string literal")

        existing = self._families.get(family)
        if existing is not None and existing is not spec_cls:
            raise ValueError(f"family '{family}' already registered for {existing.__name__}")

        self._families[family] = spec_cls

    def get(self, family: str) -> type[BaseFunctionSpec]:
        try:
            return self._families[family]
        except KeyError as exc:
            available = ", ".join(sorted(self._families)) or "<none>"
            raise KeyError(f"unknown family '{family}'. available families: {available}") from exc

    def parse(self, data: Mapping[str, Any]) -> BaseFunctionSpec:
        family_value = data.get("family")
        if not isinstance(family_value, str):
            raise ValueError("spec payload must include a string 'family' field")
        spec_cls = self.get(family_value)
        return spec_cls.model_validate(data)

    def list_families(self) -> tuple[str, ...]:
        return tuple(sorted(self._families))


FAMILY_REGISTRY = FamilyRegistry()
