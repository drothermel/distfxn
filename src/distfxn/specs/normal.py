from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .base import BaseFunctionSpec
from .param_sampling import LogUniformPositiveFloatParamSampler, UniformFloatParamSampler
from .types import FiniteStrictFloat

PositiveFiniteStrictFloat = Annotated[
    float,
    Field(strict=True, allow_inf_nan=False, gt=0.0),
]


class NormalSamplingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mean_sampler: UniformFloatParamSampler = Field(
        default_factory=lambda: UniformFloatParamSampler(
            name="mean",
            min_value=-100.0,
            max_value=100.0,
        )
    )
    stddev_sampler: LogUniformPositiveFloatParamSampler = Field(
        default_factory=lambda: LogUniformPositiveFloatParamSampler(
            name="stddev",
            min_value=1e-6,
            max_value=100.0,
        )
    )


class NormalSpec(BaseFunctionSpec):
    family: Literal["normal"] = "normal"
    mean: FiniteStrictFloat
    stddev: PositiveFiniteStrictFloat

    def sample_dist(self, rng, count: int):
        return rng.normal(self.mean, self.stddev, size=count)

    def render(self) -> str:
        return (
            "def sample_dist(rng, count):\n"
            f"    return rng.normal({self.mean!r}, {self.stddev!r}, size=count)\n"
        )

    @classmethod
    def edge_specs(cls) -> tuple["NormalSpec", ...]:
        return (
            cls(mean=0.0, stddev=1e-9),
            cls(mean=-10.0, stddev=1.0),
            cls(mean=10.0, stddev=5.0),
        )

    @classmethod
    def sample_spec(
        cls,
        rng: np.random.Generator,
        *,
        sampling_spec: NormalSamplingSpec | None = None,
    ) -> "NormalSpec":
        resolved_sampling_spec = sampling_spec or NormalSamplingSpec()
        mean_value = resolved_sampling_spec.mean_sampler.sample(rng)
        stddev_value = resolved_sampling_spec.stddev_sampler.sample(
            rng,
            context={"mean": mean_value},
        )
        return cls(mean=mean_value, stddev=stddev_value)

    @classmethod
    def sample_specs(
        cls,
        rng: np.random.Generator,
        *,
        count: int,
        sampling_spec: NormalSamplingSpec | None = None,
    ) -> tuple["NormalSpec", ...]:
        if count <= 0:
            raise ValueError("count must be greater than 0")
        resolved_sampling_spec = sampling_spec or NormalSamplingSpec()
        return tuple(
            cls.sample_spec(rng, sampling_spec=resolved_sampling_spec) for _ in range(count)
        )
