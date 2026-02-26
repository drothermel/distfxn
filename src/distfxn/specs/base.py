from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .output_checks import OutputCheck, OutputVerificationReport, default_output_checks


class BaseFunctionSpec(BaseModel):
    """Base schema for synthetic function-family specifications."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    family: str
    output_checks: tuple[OutputCheck, ...] = Field(default_factory=default_output_checks)

    def sample_dist(self, rng, count: int):
        raise NotImplementedError("spec families must implement sample_dist()")

    def render(self) -> str:
        raise NotImplementedError("spec families must implement render()")

    def validate_output(self, output: Any, *, count: int) -> OutputVerificationReport:
        values = np.asarray(output)
        results = tuple(
            check.run(values, spec=self, count=count)
            for check in self.output_checks
        )
        return OutputVerificationReport(
            family=self.family,
            passed=all(result.passed for result in results),
            results=results,
        )

    def assert_valid_output(self, output: Any, *, count: int) -> None:
        report = self.validate_output(output, count=count)
        if report.passed:
            return

        failures = [
            f"{result.name}: {result.message or 'failed'}"
            for result in report.results
            if not result.passed
        ]
        raise ValueError(
            f"output validation failed for family '{self.family}': {'; '.join(failures)}"
        )
