from typing import Annotated, Literal

from pydantic import Field

from .base import BaseFunctionSpec

PositiveFloat = Annotated[float, Field(gt=0.0)]


class NormalSpec(BaseFunctionSpec):
    family: Literal["normal"] = "normal"
    mean: float
    stddev: PositiveFloat
