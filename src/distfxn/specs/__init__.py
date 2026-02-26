from typing import Annotated

from pydantic import Field

from .base import BaseFunctionSpec
from .bernoulli import BernoulliSpec
from .normal import NormalSpec
from .registry import FAMILY_REGISTRY, FamilyRegistry
from .uniform import UniformSpec
from .verification import check_spec_equivalence, render_to_callable

FunctionSpec = Annotated[
    BernoulliSpec | UniformSpec | NormalSpec,
    Field(discriminator="family"),
]

for _spec_cls in (BernoulliSpec, UniformSpec, NormalSpec):
    FAMILY_REGISTRY.register(_spec_cls)
del _spec_cls

__all__ = [
    "BaseFunctionSpec",
    "BernoulliSpec",
    "UniformSpec",
    "NormalSpec",
    "FunctionSpec",
    "FamilyRegistry",
    "FAMILY_REGISTRY",
    "render_to_callable",
    "check_spec_equivalence",
]
