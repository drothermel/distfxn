import marimo

__generated_with = "0.20.2"
app = marimo.App(width="columns")

with app.setup:
    import numpy as np
    from distfxn.specs import (
        FAMILY_REGISTRY,
        check_spec_equivalence,
        render_to_callable,
        verify_output,
    )

    seed = 1023


@app.cell
def _():
    count = 5
    raw_specs = [
        {"family": "bernoulli", "p": 0.3},
        {"family": "uniform", "start": 0.0, "end": 50.0},
        {"family": "normal", "mean": 0.2, "stddev": 1.0},
    ]
    specs = [FAMILY_REGISTRY.parse(raw_spec) for raw_spec in raw_specs]

    checks = []
    for spec in specs:
        canonical_output = spec.sample_dist(np.random.default_rng(seed), count)
        rendered_output = render_to_callable(spec)(
            np.random.default_rng(seed), count
        )
        canonical_report = verify_output(spec, canonical_output, count=count)
        rendered_report = verify_output(spec, rendered_output, count=count)

        checks.append(
            (
                spec.family,
                check_spec_equivalence(spec, seed=seed, count=count),
                canonical_report.passed,
                rendered_report.passed,
            )
        )

    assert all(
        equivalent and canonical_ok and rendered_ok
        for _, equivalent, canonical_ok, rendered_ok in checks
    )
    checks
    return


if __name__ == "__main__":
    app.run()
