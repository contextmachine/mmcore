from __future__ import annotations
from typing import Protocol, TypeVar, SupportsIndex, Sequence, Union

from numpy._typing import ArrayLike

ShapeLike = Union[SupportsIndex, Sequence[SupportsIndex]]

D = TypeVar("D", bound=SupportsIndex)


class CurveProtocol(Protocol):
    def interval(self) -> tuple[float, float]:
        ...

    def __call__(self, t: Union[ArrayLike, float]) -> D:
        ...
