from typing import Annotated, Literal

from pydantic import Field

from .base import BaseFunctionSpec

Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class BernoulliSpec(BaseFunctionSpec):
    family: Literal["bernoulli"] = "bernoulli"
    p: Probability
