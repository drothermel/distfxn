from typing import Annotated, Literal

from pydantic import Field

from .base import BaseFunctionSpec

Probability = Annotated[
    float,
    Field(strict=True, allow_inf_nan=False, ge=0.0, le=1.0),
]


class BernoulliSpec(BaseFunctionSpec):
    family: Literal["bernoulli"] = "bernoulli"
    p: Probability

    def sample_dist(self, rng, count: int):
        return rng.binomial(n=1, p=self.p, size=count)

    def render(self) -> str:
        return (
            "def sample_dist(rng, count):\n"
            f"    return rng.binomial(n=1, p={self.p!r}, size=count)\n"
        )
