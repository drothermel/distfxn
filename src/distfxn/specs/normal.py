from typing import Annotated, Literal

from pydantic import Field

from .base import BaseFunctionSpec

PositiveFloat = Annotated[float, Field(gt=0.0)]


class NormalSpec(BaseFunctionSpec):
    family: Literal["normal"] = "normal"
    mean: float
    stddev: PositiveFloat

    def render(self) -> str:
        return (
            "def sample_dist(rng, count):\n"
            f"    return rng.normal({self.mean!r}, {self.stddev!r}, size=count)\n"
        )
