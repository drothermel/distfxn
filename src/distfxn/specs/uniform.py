from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .base import BaseFunctionSpec
from .output_checks import InRangeCheck, OutputCheck, default_output_checks
from .param_sampling import (
    LogUniformPositiveFloatParamSampler,
    SamplingSpecError,
    UniformFloatParamSampler,
)
from .types import FiniteStrictFloat


class UniformSamplingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start_sampler: UniformFloatParamSampler = Field(
        default_factory=lambda: UniformFloatParamSampler(
            name="start",
            min_value=-100.0,
            max_value=100.0,
        )
    )
    width_sampler: LogUniformPositiveFloatParamSampler = Field(
        default_factory=lambda: LogUniformPositiveFloatParamSampler(
            name="width",
            min_value=1e-6,
            max_value=100.0,
        )
    )


class UniformSpec(BaseFunctionSpec):
    family: Literal["uniform"] = "uniform"
    start: FiniteStrictFloat
    end: FiniteStrictFloat
    output_checks: tuple[OutputCheck, ...] = Field(
        default_factory=lambda: default_output_checks()
        + (InRangeCheck(min_field="start", max_field="end", include_max=False),)
    )

    @model_validator(mode="after")
    def validate_bounds(self) -> "UniformSpec":
        if self.start >= self.end:
            raise ValueError("start must be less than end")
        return self

    def sample_dist(self, rng, count: int):
        return rng.uniform(self.start, self.end, size=count)

    def render(self) -> str:
        return (
            "def sample_dist(rng, count):\n"
            f"    return rng.uniform({self.start!r}, {self.end!r}, size=count)\n"
        )

    @classmethod
    def edge_specs(cls) -> tuple["UniformSpec", ...]:
        return (
            cls(start=-1.0, end=1.0),
            cls(start=0.0, end=1e-9),
            cls(start=10.0, end=1000.0),
        )

    @classmethod
    def sample_spec(
        cls,
        rng: np.random.Generator,
        *,
        sampling_spec: UniformSamplingSpec | None = None,
    ) -> "UniformSpec":
        resolved_sampling_spec = sampling_spec or UniformSamplingSpec()
        start_value = resolved_sampling_spec.start_sampler.sample(rng)
        width_value = resolved_sampling_spec.width_sampler.sample(
            rng,
            context={"start": start_value},
        )
        end_value = start_value + width_value
        if not np.isfinite(end_value):
            raise SamplingSpecError("sampled uniform bounds produced a non-finite end value")
        return cls(start=start_value, end=end_value)

    @classmethod
    def sample_specs(
        cls,
        rng: np.random.Generator,
        *,
        count: int,
        sampling_spec: UniformSamplingSpec | None = None,
    ) -> tuple["UniformSpec", ...]:
        if count <= 0:
            raise ValueError("count must be greater than 0")
        resolved_sampling_spec = sampling_spec or UniformSamplingSpec()
        return tuple(
            cls.sample_spec(rng, sampling_spec=resolved_sampling_spec) for _ in range(count)
        )
