from typing import Any

import pydantic
import pydantic.generics as pg
from pydantic_numpy.ndarray import NDArray

import mm.generics.simple as gens
from mm.baseitems import Item


class MModel(pydantic.BaseModel):
    """
    Base class from pydantic-mmodel
    """

    def __init__(self, **data: Any):
        super().__init__(**data)


class MmodelDict(gens.D3[Item, dict, str, MModel]):
    ...


class GenericMModel(pg.GenericModel):
    """
    Base class from pydantic-mmodel
    """

    def __init__(self, **data: Any):
        super().__init__(**data)


class CircleHole(MModel):
    center: NDArray
    radius: float


class Panel(pydantic.BaseModel):
    tag: str
    pins: NDArray
    marker_pins: NDArray
    holes: list[CircleHole]
