from typing import Annotated

from pydantic import Field

from .base import BaseFunctionSpec
from .bernoulli import BernoulliSamplingSpec, BernoulliSpec
from .equivalence_cases import EquivalenceCase, default_equivalence_cases
from .normal import NormalSamplingSpec, NormalSpec
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
from .param_sampling import (
    LogUniformPositiveFloatParamSampler,
    SamplingSpecError,
    UniformFloatParamSampler,
)
from .registry import FAMILY_REGISTRY, FamilyRegistry
from .uniform import UniformSamplingSpec, UniformSpec
from .verification import (
    CaseVerificationReport,
    CaseEquivalenceResult,
    SpecVerificationReport,
    SpecEquivalenceReport,
    assert_valid_output,
    check_spec_equivalence,
    render_to_callable,
    run_equivalence_cases,
    run_render_equivalence_cases,
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
    "BernoulliSamplingSpec",
    "UniformSpec",
    "UniformSamplingSpec",
    "NormalSpec",
    "NormalSamplingSpec",
    "FunctionSpec",
    "FamilyRegistry",
    "FAMILY_REGISTRY",
    "EquivalenceCase",
    "default_equivalence_cases",
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
    "UniformFloatParamSampler",
    "LogUniformPositiveFloatParamSampler",
    "SamplingSpecError",
    "render_to_callable",
    "check_spec_equivalence",
    "CaseVerificationReport",
    "CaseEquivalenceResult",
    "SpecVerificationReport",
    "SpecEquivalenceReport",
    "run_equivalence_cases",
    "run_render_equivalence_cases",
    "verify_output",
    "assert_valid_output",
]
