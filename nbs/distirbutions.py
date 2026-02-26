import marimo

__generated_with = "0.20.2"
app = marimo.App(width="columns")

with app.setup:
    import marimo as mo
    import numpy as np
    from distfxn.specs import FAMILY_REGISTRY

    seed = 1023


@app.function
def render_to_callable(spec):
    namespace = {}
    exec(spec.render(), {}, namespace)
    return namespace["sample_dist"]


@app.cell
def _(render_to_callable):
    count = 5
    raw_specs = [
        {"family": "bernoulli", "p": 0.3},
        {"family": "uniform", "start": 0, "end": 50},
        {"family": "normal", "mean": 0.2, "stddev": 1.0},
    ]
    specs = [FAMILY_REGISTRY.parse(raw_spec) for raw_spec in raw_specs]

    checks = []
    for spec in specs:
        rng_canonical = np.random.default_rng(seed)
        rng_rendered = np.random.default_rng(seed)

        canonical_values = spec.sample_dist(rng_canonical, count)
        rendered_sample_dist = render_to_callable(spec)
        rendered_values = rendered_sample_dist(rng_rendered, count)
        checks.append((spec.family, np.array_equal(canonical_values, rendered_values)))

    assert all(match for _, match in checks)
    checks
    return


if __name__ == "__main__":
    app.run()
