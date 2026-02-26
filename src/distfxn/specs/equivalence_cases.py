from pydantic import BaseModel, ConfigDict, Field


class EquivalenceCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    seed: int = Field(strict=True, ge=0)
    count: int = Field(strict=True, gt=0)


def default_equivalence_cases() -> tuple[EquivalenceCase, ...]:
    return (
        EquivalenceCase(name="seed_0_count_1", seed=0, count=1),
        EquivalenceCase(name="seed_1_count_5", seed=1, count=5),
        EquivalenceCase(name="seed_1023_count_32", seed=1023, count=32),
    )
