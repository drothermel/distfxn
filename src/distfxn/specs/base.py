from pydantic import BaseModel, ConfigDict


class BaseFunctionSpec(BaseModel):
    """Base schema for synthetic function-family specifications."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    family: str

    def sample_dist(self, rng, count: int):
        raise NotImplementedError("spec families must implement sample_dist()")

    def render(self) -> str:
        raise NotImplementedError("spec families must implement render()")
