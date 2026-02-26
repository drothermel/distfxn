from typing import Annotated

from pydantic import Field

FiniteStrictFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]

