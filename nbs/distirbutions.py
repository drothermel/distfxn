import marimo

__generated_with = "0.20.2"
app = marimo.App(width="columns")

with app.setup:
    import marimo as mo
    import numpy as np

    seed = 1023


@app.cell
def _():
    RNG = np.random.default_rng(seed)
    return (RNG,)


@app.function
def sample_bernoulli(spec, rng, count):
    return rng.binomial(n=1, p=spec["p"], size=count)


@app.function
def sample_uniform(spec, rng, count):
    return rng.uniform(spec["start"], spec["end"], size=count)


@app.function
def sample_normal(spec, rng, count):
    return rng.normal(spec["mean"], spec["stddev"], size=count)


@app.cell
def _(RNG):
    (
        sample_bernoulli({"p": 0.3}, RNG, 5),
        sample_uniform({"start": 0, "end": 50}, RNG, 5),
        sample_normal({"mean": 0.2, "stddev": 1.0}, RNG, 5),
    )
    return


if __name__ == "__main__":
    app.run()
