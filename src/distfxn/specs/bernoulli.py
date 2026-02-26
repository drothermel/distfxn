from typing import Annotated, Literal

from pydantic import Field

from .base import BaseFunctionSpec
from .output_checks import InSetCheck, OutputCheck, default_output_checks

Probability = Annotated[
    float,
    Field(strict=True, allow_inf_nan=False, ge=0.0, le=1.0),
]


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
