from typing import Annotated

from pydantic import Field

from .base import BaseFunctionSpec
from .bernoulli import BernoulliSpec
from .normal import NormalSpec
from .output_checks import (
    CheckResult,
    FiniteValuesCheck,
    InRangeCheck,
    InSetCheck,
    LengthCheck,
    NumericDtypeCheck,
    OneDimensionalCheck,
    OutputCheck,
    OutputCheckBase,
    OutputVerificationReport,
    default_output_checks,
)
from .registry import FAMILY_REGISTRY, FamilyRegistry
from .uniform import UniformSpec
from .verification import (
    assert_valid_output,
    check_spec_equivalence,
    render_to_callable,
    verify_output,
)

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
    "OutputCheckBase",
    "OutputCheck",
    "CheckResult",
    "OutputVerificationReport",
    "OneDimensionalCheck",
    "LengthCheck",
    "NumericDtypeCheck",
    "FiniteValuesCheck",
    "InSetCheck",
    "InRangeCheck",
    "default_output_checks",
    "render_to_callable",
    "check_spec_equivalence",
    "verify_output",
    "assert_valid_output",
]
