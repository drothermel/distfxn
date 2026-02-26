from collections.abc import Callable
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from .base import BaseFunctionSpec
from .equivalence_cases import EquivalenceCase
from .output_checks import CheckResult, OutputVerificationReport

CandidateSampler = Callable[[BaseFunctionSpec, np.random.Generator, int], Any]


class CaseVerificationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case: EquivalenceCase
    canonical_output_report: OutputVerificationReport
    candidate_output_report: OutputVerificationReport
    exact_output_match: bool
    passed: bool
    failure_reasons: tuple[str, ...] = ()

    def to_lines(self) -> tuple[str, ...]:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"[{status}] case '{self.case.name}' seed={self.case.seed} count={self.case.count}",
            f"  exact_output_match: {self.exact_output_match}",
            "  canonical:",
        ]
        lines.extend(f"    {line}" for line in self.canonical_output_report.to_lines())
        lines.append("  candidate:")
        lines.extend(f"    {line}" for line in self.candidate_output_report.to_lines())
        if self.failure_reasons:
            lines.append("  failure_reasons:")
            lines.extend(f"    - {reason}" for reason in self.failure_reasons)
        return tuple(lines)

    def to_markdown(self) -> str:
        return "\n".join(self.to_lines())

    def to_dict(self) -> dict:
        return self.model_dump()


class SpecVerificationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    family: str
    passed: bool
    case_reports: tuple[CaseVerificationReport, ...]

    def to_lines(self) -> tuple[str, ...]:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"[{status}] verification report for family '{self.family}'"]
        for case_report in self.case_reports:
            lines.extend(f"  {line}" for line in case_report.to_lines())
        return tuple(lines)

    def to_markdown(self) -> str:
        return "\n".join(self.to_lines())

    def to_dict(self) -> dict:
        return self.model_dump()


CaseEquivalenceResult = CaseVerificationReport
SpecEquivalenceReport = SpecVerificationReport


def render_to_callable(spec: BaseFunctionSpec):
    namespace = {}
    exec(spec.render(), {}, namespace)
    sample_dist = namespace.get("sample_dist")
    if not callable(sample_dist):
        raise ValueError("render() must define a callable named 'sample_dist'")
    return sample_dist


def _sampler_error_report(spec: BaseFunctionSpec, message: str) -> OutputVerificationReport:
    return OutputVerificationReport(
        family=spec.family,
        passed=False,
        results=(CheckResult(name="sampler_error", passed=False, message=message),),
    )


def _failure_reasons(
    canonical_output_report: OutputVerificationReport,
    candidate_output_report: OutputVerificationReport,
    exact_output_match: bool,
) -> tuple[str, ...]:
    reasons = [
        f"canonical.{result.name}: {result.message or 'failed'}"
        for result in canonical_output_report.failed_results()
    ]
    reasons.extend(
        f"candidate.{result.name}: {result.message or 'failed'}"
        for result in candidate_output_report.failed_results()
    )
    if not exact_output_match:
        reasons.append("exact_output_match: canonical and candidate outputs differ")
    return tuple(reasons)


def run_equivalence_cases(
    spec: BaseFunctionSpec,
    candidate_sampler: CandidateSampler,
    *,
    cases: tuple[EquivalenceCase, ...] | None = None,
) -> SpecVerificationReport:
    resolved_cases = cases if cases is not None else spec.all_equivalence_cases()
    if len(resolved_cases) == 0:
        raise ValueError("at least one equivalence case is required")

    case_reports: list[CaseVerificationReport] = []

    for case in resolved_cases:
        canonical_output = None
        try:
            canonical_output = spec.sample_dist(np.random.default_rng(case.seed), case.count)
            canonical_report = verify_output(spec, canonical_output, count=case.count)
        except Exception as exc:
            canonical_report = _sampler_error_report(spec, f"canonical sampler failed: {exc!r}")

        candidate_output = None
        try:
            candidate_output = candidate_sampler(spec, np.random.default_rng(case.seed), case.count)
            candidate_report = verify_output(spec, candidate_output, count=case.count)
        except Exception as exc:
            candidate_report = _sampler_error_report(spec, f"candidate sampler failed: {exc!r}")

        exact_output_match = (
            canonical_output is not None
            and candidate_output is not None
            and bool(np.array_equal(canonical_output, candidate_output))
        )
        passed = canonical_report.passed and candidate_report.passed and exact_output_match
        case_reports.append(
            CaseVerificationReport(
                case=case,
                canonical_output_report=canonical_report,
                candidate_output_report=candidate_report,
                exact_output_match=exact_output_match,
                passed=passed,
                failure_reasons=_failure_reasons(
                    canonical_output_report=canonical_report,
                    candidate_output_report=candidate_report,
                    exact_output_match=exact_output_match,
                ),
            )
        )

    return SpecVerificationReport(
        family=spec.family,
        passed=all(case_report.passed for case_report in case_reports),
        case_reports=tuple(case_reports),
    )


def run_render_equivalence_cases(
    spec: BaseFunctionSpec,
    *,
    cases: tuple[EquivalenceCase, ...] | None = None,
) -> SpecVerificationReport:
    rendered_sample_dist = render_to_callable(spec)
    return run_equivalence_cases(
        spec,
        lambda _spec, rng, count: rendered_sample_dist(rng, count),
        cases=cases,
    )


def check_spec_equivalence(spec: BaseFunctionSpec, *, seed: int, count: int) -> bool:
    case = EquivalenceCase(name="single_case", seed=seed, count=count)
    report = run_render_equivalence_cases(spec, cases=(case,))
    return report.passed


def verify_output(
    spec: BaseFunctionSpec,
    output,
    *,
    count: int,
) -> OutputVerificationReport:
    return spec.validate_output(output, count=count)


def assert_valid_output(spec: BaseFunctionSpec, output, *, count: int) -> None:
    spec.assert_valid_output(output, count=count)
