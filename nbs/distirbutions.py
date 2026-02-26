import marimo

__generated_with = "0.20.2"
app = marimo.App(width="columns")

with app.setup:
    import marimo as mo
    import numpy as np
    from distfxn.specs import BernoulliSpec, FunctionSpec, NormalSpec, UniformSpec

    seed = 1023


@app.function
def sample_bernoulli(spec: BernoulliSpec, rng: np.random.Generator, count: int):
    return rng.binomial(n=1, p=spec.p, size=count)


@app.function
def sample_uniform(spec: UniformSpec, rng: np.random.Generator, count: int):
    return rng.uniform(spec.start, spec.end, size=count)


@app.function
def sample_normal(spec: NormalSpec, rng: np.random.Generator, count: int):
    return rng.normal(spec.mean, spec.stddev, size=count)


@app.function
def sample_dist(spec: FunctionSpec, rng: np.random.Generator, count: int):
    if isinstance(spec, BernoulliSpec):
        return sample_bernoulli(spec, rng, count)
    if isinstance(spec, UniformSpec):
        return sample_uniform(spec, rng, count)
    if isinstance(spec, NormalSpec):
        return sample_normal(spec, rng, count)
    raise TypeError(f"unsupported spec type: {type(spec)!r}")


@app.function
def render_to_callable(spec: FunctionSpec):
    namespace = {}
    exec(spec.render(), {}, namespace)
    return namespace["sample_dist"]


@app.cell
def _(render_to_callable, sample_dist):
    count = 5
    specs = [
        BernoulliSpec(p=0.3),
        UniformSpec(start=0, end=50),
        NormalSpec(mean=0.2, stddev=1.0),
    ]

    checks = []
    for spec in specs:
        rng_canonical = np.random.default_rng(seed)
        rng_rendered = np.random.default_rng(seed)

        canonical_values = sample_dist(spec, rng_canonical, count)
        rendered_sample_dist = render_to_callable(spec)
        rendered_values = rendered_sample_dist(rng_rendered, count)
        checks.append((spec.family, np.array_equal(canonical_values, rendered_values)))

    assert all(match for _, match in checks)
    checks
    return


if __name__ == "__main__":
    app.run()
