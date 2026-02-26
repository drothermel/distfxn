import marimo

__generated_with = "0.20.2"
app = marimo.App(width="columns")

with app.setup:
    from distfxn.specs import FAMILY_REGISTRY, check_spec_equivalence

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

    checks = [
        (spec.family, check_spec_equivalence(spec, seed=seed, count=count))
        for spec in specs
    ]

    assert all(match for _, match in checks)
    checks
    return


if __name__ == "__main__":
    app.run()
