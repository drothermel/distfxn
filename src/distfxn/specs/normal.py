from typing import Annotated, Literal

from pydantic import Field

from .base import BaseFunctionSpec

FiniteStrictFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]
PositiveFiniteStrictFloat = Annotated[
    float,
    Field(strict=True, allow_inf_nan=False, gt=0.0),
]


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
