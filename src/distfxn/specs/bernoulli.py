from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .base import BaseFunctionSpec
from .output_checks import InSetCheck, OutputCheck, default_output_checks
from .param_sampling import UniformFloatParamSampler

Probability = Annotated[
    float,
    Field(strict=True, allow_inf_nan=False, ge=0.0, le=1.0),
]


class BernoulliSamplingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    p_sampler: UniformFloatParamSampler = Field(
        default_factory=lambda: UniformFloatParamSampler(
            name="p",
            min_value=0.0,
            max_value=1.0,
        )
    )


class BernoulliSpec(BaseFunctionSpec):
    family: Literal["bernoulli"] = "bernoulli"
    p: Probability
    output_checks: tuple[OutputCheck, ...] = Field(
        default_factory=lambda: default_output_checks()
        + (InSetCheck(allowed=(0.0, 1.0)),)
    )

    def sample_dist(self, rng, count: int):
        return rng.binomial(n=1, p=self.p, size=count)

    def render(self) -> str:
        return (
            "def sample_dist(rng, count):\n"
            f"    return rng.binomial(n=1, p={self.p!r}, size=count)\n"
        )

    @classmethod
    def edge_specs(cls) -> tuple["BernoulliSpec", ...]:
        return (
            cls(p=0.0),
            cls(p=1.0),
            cls(p=0.5),
        )

    @classmethod
    def sample_spec(
        cls,
        rng: np.random.Generator,
        *,
        sampling_spec: BernoulliSamplingSpec | None = None,
    ) -> "BernoulliSpec":
        resolved_sampling_spec = sampling_spec or BernoulliSamplingSpec()
        p_value = resolved_sampling_spec.p_sampler.sample(rng)
        return cls(p=p_value)

    @classmethod
    def sample_specs(
        cls,
        rng: np.random.Generator,
        *,
        count: int,
        sampling_spec: BernoulliSamplingSpec | None = None,
    ) -> tuple["BernoulliSpec", ...]:
        if count <= 0:
            raise ValueError("count must be greater than 0")
        resolved_sampling_spec = sampling_spec or BernoulliSamplingSpec()
        return tuple(
            cls.sample_spec(rng, sampling_spec=resolved_sampling_spec) for _ in range(count)
        )
