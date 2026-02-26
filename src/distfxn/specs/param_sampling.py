from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .types import FiniteStrictFloat


class SamplingSpecError(ValueError):
    """Raised when a sampling specification cannot produce valid values."""


class UniformFloatParamSampler(BaseModel):
    """Samples a finite float uniformly from [min_value, max_value]."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    min_value: FiniteStrictFloat
    max_value: FiniteStrictFloat

    @model_validator(mode="after")
    def validate_bounds(self) -> "UniformFloatParamSampler":
        if self.min_value >= self.max_value:
            raise ValueError("min_value must be less than max_value")
        return self

    def sample(
        self,
        rng: np.random.Generator,
        *,
        context: dict[str, float] | None = None,
    ) -> float:
        del context
        value = float(rng.uniform(self.min_value, self.max_value))
        if not np.isfinite(value):
            raise SamplingSpecError(f"sampler '{self.name}' produced a non-finite value")
        return value


class LogUniformPositiveFloatParamSampler(BaseModel):
    """Samples a finite positive float log-uniformly from [min_value, max_value]."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    min_value: FiniteStrictFloat = Field(gt=0.0)
    max_value: FiniteStrictFloat = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_bounds(self) -> "LogUniformPositiveFloatParamSampler":
        if self.min_value >= self.max_value:
            raise ValueError("min_value must be less than max_value")
        return self

    def sample(
        self,
        rng: np.random.Generator,
        *,
        context: dict[str, float] | None = None,
    ) -> float:
        del context
        log_min = float(np.log(self.min_value))
        log_max = float(np.log(self.max_value))
        if not np.isfinite(log_min) or not np.isfinite(log_max):
            raise SamplingSpecError(f"sampler '{self.name}' has invalid log-space bounds")

        value = float(np.exp(rng.uniform(log_min, log_max)))
        if not np.isfinite(value):
            raise SamplingSpecError(f"sampler '{self.name}' produced a non-finite value")
        if value <= 0.0:
            raise SamplingSpecError(f"sampler '{self.name}' produced a non-positive value")
        return value
