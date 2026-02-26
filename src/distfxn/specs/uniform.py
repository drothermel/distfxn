from typing import Annotated, Literal

from pydantic import Field, model_validator

from .base import BaseFunctionSpec
from .output_checks import InRangeCheck, OutputCheck, default_output_checks

FiniteStrictFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class UniformSpec(BaseFunctionSpec):
    family: Literal["uniform"] = "uniform"
    start: FiniteStrictFloat
    end: FiniteStrictFloat
    output_checks: tuple[OutputCheck, ...] = Field(
        default_factory=lambda: default_output_checks()
        + (InRangeCheck(min_field="start", max_field="end"),)
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
