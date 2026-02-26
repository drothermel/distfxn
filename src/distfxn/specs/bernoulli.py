from typing import Annotated, Literal

from pydantic import Field

from .base import BaseFunctionSpec

Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class BernoulliSpec(BaseFunctionSpec):
    family: Literal["bernoulli"] = "bernoulli"
    p: Probability

    def render(self) -> str:
        return (
            "def sample_dist(rng, count):\n"
            f"    return rng.binomial(n=1, p={self.p!r}, size=count)\n"
        )
