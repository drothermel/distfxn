import marimo

__generated_with = "0.20.2"
app = marimo.App(width="columns")

with app.setup:
    from distfxn.specs import (
        FAMILY_REGISTRY,
        run_render_equivalence_cases,
    )


@app.cell
def _():
    raw_specs = [
        {"family": "bernoulli", "p": 0.3},
        {"family": "uniform", "start": 0.0, "end": 50.0},
        {"family": "normal", "mean": 0.2, "stddev": 1.0},
    ]
    specs = [FAMILY_REGISTRY.parse(raw_spec) for raw_spec in raw_specs]

    reports = [run_render_equivalence_cases(spec) for spec in specs]
    assert all(report.passed for report in reports)
    [report.to_dict() for report in reports]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
