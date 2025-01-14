from __future__ import annotations
from typing import Protocol,runtime_checkable
from numpy.typing import ArrayLike,NDArray
@runtime_checkable
class CurveProtocol(Protocol):
    def evaluate(self, t:float)->ArrayLike[float]:...

    def evaluate_multi(self, t:NDArray[float]) -> ArrayLike[float]: ...

    def derivative(self, t: float) -> ArrayLike[float]: ...

    def second_derivative(self, t: float) -> ArrayLike[float]: ...

    def tangent(self, t: float) -> ArrayLike[float]: ...

    def normal(self, t: float) -> ArrayLike[float]: ...

@runtime_checkable
class SurfaceProtocol(Protocol):
    def evaluate(self, uv:NDArray[float])->ArrayLike[float]:...

    def evaluate_multi(self, uv:NDArray[float]) -> ArrayLike[float]: ...

    def derivative_u(self, uv:NDArray[float]) -> ArrayLike[float]: ...

    def second_derivative_uu(self, uv:NDArray[float]) -> ArrayLike[float]: ...
    def derivative_v(self, uv:NDArray[float]) -> ArrayLike[float]: ...

    def second_derivative_vv(self, uv:NDArray[float]) -> ArrayLike[float]: ...
    def second_derivative_uv(self, uv:NDArray[float]) -> ArrayLike[float]: ...

    def normal(self, uv:NDArray[float]) -> ArrayLike[float]: ...

