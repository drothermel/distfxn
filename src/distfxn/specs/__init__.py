from typing import Annotated

from pydantic import Field

from .base import BaseFunctionSpec
from .bernoulli import BernoulliSpec
from .normal import NormalSpec
from .uniform import UniformSpec

FunctionSpec = Annotated[
    BernoulliSpec | UniformSpec | NormalSpec,
    Field(discriminator="family"),
]

__all__ = [
    "BaseFunctionSpec",
    "BernoulliSpec",
    "UniformSpec",
    "NormalSpec",
    "FunctionSpec",
]
