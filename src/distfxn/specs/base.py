from pydantic import BaseModel, ConfigDict


class BaseFunctionSpec(BaseModel):
    """Base schema for synthetic function-family specifications."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    family: str
