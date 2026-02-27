import marimo

__generated_with = "0.20.2"
app = marimo.App(width="columns")

@app.cell
def _():
    import numpy as np

    from distfxn.specs import (
        BernoulliSamplingSpec,
        BernoulliSpec,
        FAMILY_REGISTRY,
        LogUniformPositiveFloatParamSampler,
        NormalSamplingSpec,
        NormalSpec,
        UniformFloatParamSampler,
        UniformSamplingSpec,
        UniformSpec,
        run_render_equivalence_cases,
    )
    return (
        BernoulliSamplingSpec,
        BernoulliSpec,
        FAMILY_REGISTRY,
        LogUniformPositiveFloatParamSampler,
        NormalSamplingSpec,
        NormalSpec,
        UniformFloatParamSampler,
        UniformSamplingSpec,
        UniformSpec,
        np,
        run_render_equivalence_cases,
    )


@app.cell
def _(FAMILY_REGISTRY, run_render_equivalence_cases):
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
def _(BernoulliSpec, NormalSpec, UniformSpec, np):
    rng = np.random.default_rng(123)
    default_samples = {
        "bernoulli": BernoulliSpec.sample_specs(rng, count=3),
        "uniform": UniformSpec.sample_specs(rng, count=3),
        "normal": NormalSpec.sample_specs(rng, count=3),
    }
    {key: [sample.model_dump() for sample in value] for key, value in default_samples.items()}
    return default_samples, rng


@app.cell
def _(
    BernoulliSamplingSpec,
    BernoulliSpec,
    LogUniformPositiveFloatParamSampler,
    NormalSamplingSpec,
    NormalSpec,
    UniformFloatParamSampler,
    UniformSamplingSpec,
    UniformSpec,
    np,
):
    custom_rng = np.random.default_rng(999)

    bernoulli_sampling_spec = BernoulliSamplingSpec(
        p_sampler=UniformFloatParamSampler(name="p", min_value=0.2, max_value=0.8)
    )
    normal_sampling_spec = NormalSamplingSpec(
        mean_sampler=UniformFloatParamSampler(name="mean", min_value=-5.0, max_value=5.0),
        stddev_sampler=LogUniformPositiveFloatParamSampler(
            name="stddev",
            min_value=1e-3,
            max_value=2.0,
        ),
    )
    uniform_sampling_spec = UniformSamplingSpec(
        start_sampler=UniformFloatParamSampler(name="start", min_value=-10.0, max_value=10.0),
        width_sampler=LogUniformPositiveFloatParamSampler(
            name="width",
            min_value=1e-2,
            max_value=5.0,
        ),
    )

    custom_samples = {
        "bernoulli": BernoulliSpec.sample_specs(
            custom_rng,
            count=3,
            sampling_spec=bernoulli_sampling_spec,
        ),
        "normal": NormalSpec.sample_specs(
            custom_rng,
            count=3,
            sampling_spec=normal_sampling_spec,
        ),
        "uniform": UniformSpec.sample_specs(
            custom_rng,
            count=3,
            sampling_spec=uniform_sampling_spec,
        ),
    }
    {key: [sample.model_dump() for sample in value] for key, value in custom_samples.items()}
    return custom_samples


@app.cell
def _(custom_samples, run_render_equivalence_cases):
    sampled_reports = [
        run_render_equivalence_cases(spec)
        for samples in custom_samples.values()
        for spec in samples
    ]
    assert all(report.passed for report in sampled_reports)
    [report.to_dict() for report in sampled_reports]
    return


@app.cell
def _(default_samples):
    # quick invariant checks for sampled specs
    assert all(0.0 <= sample.p <= 1.0 for sample in default_samples["bernoulli"])
    assert all(sample.start < sample.end for sample in default_samples["uniform"])
    assert all(sample.stddev > 0.0 for sample in default_samples["normal"])
    return


if __name__ == "__main__":
    app.run()
