from typing import Literal

from pydantic import model_validator

from .base import BaseFunctionSpec


class UniformSpec(BaseFunctionSpec):
    family: Literal["uniform"] = "uniform"
    start: float
    end: float

    @model_validator(mode="after")
    def validate_bounds(self) -> "UniformSpec":
        if self.start >= self.end:
            raise ValueError("start must be less than end")
        return self
