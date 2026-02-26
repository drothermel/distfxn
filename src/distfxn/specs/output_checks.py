from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from .base import BaseFunctionSpec

FiniteStrictFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class CheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    passed: bool
    message: str | None = None

    def to_line(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        details = f" ({self.message})" if self.message else ""
        return f"[{status}] {self.name}{details}"

    def to_dict(self) -> dict:
        return self.model_dump()


class OutputVerificationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    family: str
    passed: bool
    results: tuple[CheckResult, ...]

    def failed_results(self) -> tuple[CheckResult, ...]:
        return tuple(result for result in self.results if not result.passed)

    def to_lines(self) -> tuple[str, ...]:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"[{status}] output checks for family '{self.family}'"]
        lines.extend(f"  {result.to_line()}" for result in self.results)
        return tuple(lines)

    def to_markdown(self) -> str:
        return "\n".join(self.to_lines())

    def to_dict(self) -> dict:
        return self.model_dump()


class OutputCheckBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str
    name: str

    def run(self, values: np.ndarray, *, spec: BaseFunctionSpec, count: int) -> CheckResult:
        raise NotImplementedError("output checks must implement run()")


class OneDimensionalCheck(OutputCheckBase):
    kind: Literal["one_dimensional"] = "one_dimensional"
    name: str = "one_dimensional"

    def run(self, values: np.ndarray, *, spec: BaseFunctionSpec, count: int) -> CheckResult:
        passed = values.ndim == 1
        message = None if passed else f"expected 1D output but got ndim={values.ndim}"
        return CheckResult(name=self.name, passed=passed, message=message)


class LengthCheck(OutputCheckBase):
    kind: Literal["length"] = "length"
    name: str = "length"

    def run(self, values: np.ndarray, *, spec: BaseFunctionSpec, count: int) -> CheckResult:
        if count <= 0:
            return CheckResult(name=self.name, passed=False, message="count must be greater than 0")
        if values.ndim != 1:
            return CheckResult(name=self.name, passed=False, message="length check requires 1D output")
        passed = values.shape[0] == count
        message = None if passed else f"expected length {count} but got {values.shape[0]}"
        return CheckResult(name=self.name, passed=passed, message=message)


class NumericDtypeCheck(OutputCheckBase):
    kind: Literal["numeric_dtype"] = "numeric_dtype"
    name: str = "numeric_dtype"

    def run(self, values: np.ndarray, *, spec: BaseFunctionSpec, count: int) -> CheckResult:
        passed = np.issubdtype(values.dtype, np.number)
        message = None if passed else f"expected numeric dtype but got {values.dtype}"
        return CheckResult(name=self.name, passed=passed, message=message)


class FiniteValuesCheck(OutputCheckBase):
    kind: Literal["finite_values"] = "finite_values"
    name: str = "finite_values"

    def run(self, values: np.ndarray, *, spec: BaseFunctionSpec, count: int) -> CheckResult:
        if not np.issubdtype(values.dtype, np.number):
            return CheckResult(name=self.name, passed=False, message="finite check requires numeric dtype")
        passed = bool(np.isfinite(values).all())
        message = None if passed else "output contains NaN or infinite values"
        return CheckResult(name=self.name, passed=passed, message=message)


class InSetCheck(OutputCheckBase):
    kind: Literal["in_set"] = "in_set"
    name: str = "in_set"
    allowed: tuple[FiniteStrictFloat, ...] = Field(min_length=1)

    def run(self, values: np.ndarray, *, spec: BaseFunctionSpec, count: int) -> CheckResult:
        if not np.issubdtype(values.dtype, np.number):
            return CheckResult(name=self.name, passed=False, message="set check requires numeric dtype")

        mask = np.isin(values, np.array(self.allowed))
        passed = bool(mask.all())
        if passed:
            return CheckResult(name=self.name, passed=True)

        first_invalid = values[np.logical_not(mask)][0]
        message = f"value {first_invalid!r} is not in allowed set {self.allowed!r}"
        return CheckResult(name=self.name, passed=False, message=message)


class InRangeCheck(OutputCheckBase):
    kind: Literal["in_range"] = "in_range"
    name: str = "in_range"
    min_value: FiniteStrictFloat | None = None
    max_value: FiniteStrictFloat | None = None
    min_field: str | None = None
    max_field: str | None = None
    include_min: bool = True
    include_max: bool = True

    @model_validator(mode="after")
    def validate_range_sources(self) -> "InRangeCheck":
        has_min_value = self.min_value is not None
        has_min_field = self.min_field is not None
        has_max_value = self.max_value is not None
        has_max_field = self.max_field is not None

        if has_min_value == has_min_field:
            raise ValueError("provide exactly one of min_value or min_field")
        if has_max_value == has_max_field:
            raise ValueError("provide exactly one of max_value or max_field")

        return self

    def _resolve_min(self, spec: BaseFunctionSpec) -> float:
        if self.min_field is not None:
            value = getattr(spec, self.min_field)
            return float(value)
        return float(self.min_value)

    def _resolve_max(self, spec: BaseFunctionSpec) -> float:
        if self.max_field is not None:
            value = getattr(spec, self.max_field)
            return float(value)
        return float(self.max_value)

    def run(self, values: np.ndarray, *, spec: BaseFunctionSpec, count: int) -> CheckResult:
        if not np.issubdtype(values.dtype, np.number):
            return CheckResult(name=self.name, passed=False, message="range check requires numeric dtype")

        try:
            lower = self._resolve_min(spec)
            upper = self._resolve_max(spec)
        except (AttributeError, TypeError, ValueError) as exc:
            return CheckResult(name=self.name, passed=False, message=str(exc))

        if lower > upper:
            return CheckResult(
                name=self.name,
                passed=False,
                message=f"resolved invalid range [{lower}, {upper}]",
            )

        lower_mask = values >= lower if self.include_min else values > lower
        upper_mask = values <= upper if self.include_max else values < upper
        mask = np.logical_and(lower_mask, upper_mask)

        passed = bool(mask.all())
        if passed:
            return CheckResult(name=self.name, passed=True)

        first_invalid = values[np.logical_not(mask)][0]
        left_bracket = "[" if self.include_min else "("
        right_bracket = "]" if self.include_max else ")"
        message = (
            f"value {first_invalid!r} is outside allowed range "
            f"{left_bracket}{lower}, {upper}{right_bracket}"
        )
        return CheckResult(name=self.name, passed=False, message=message)


OutputCheck = Annotated[
    OneDimensionalCheck
    | LengthCheck
    | NumericDtypeCheck
    | FiniteValuesCheck
    | InSetCheck
    | InRangeCheck,
    Field(discriminator="kind"),
]


def default_output_checks() -> tuple[OutputCheck, ...]:
    return (
        OneDimensionalCheck(),
        LengthCheck(),
        NumericDtypeCheck(),
        FiniteValuesCheck(),
    )
