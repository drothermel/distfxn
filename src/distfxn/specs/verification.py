import numpy as np

from .base import BaseFunctionSpec


def render_to_callable(spec: BaseFunctionSpec):
    namespace = {}
    exec(spec.render(), {}, namespace)
    sample_dist = namespace.get("sample_dist")
    if not callable(sample_dist):
        raise ValueError("render() must define a callable named 'sample_dist'")
    return sample_dist


def check_spec_equivalence(spec: BaseFunctionSpec, *, seed: int, count: int) -> bool:
    if count <= 0:
        raise ValueError("count must be greater than 0")

    rng_canonical = np.random.default_rng(seed)
    rng_rendered = np.random.default_rng(seed)

    canonical_values = spec.sample_dist(rng_canonical, count)
    rendered_values = render_to_callable(spec)(rng_rendered, count)
    return np.array_equal(canonical_values, rendered_values)
