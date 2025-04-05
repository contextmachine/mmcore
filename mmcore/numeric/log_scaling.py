from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

def to_log(a:NDArray)->NDArray:
    return np.sign(a) * np.log1p(np.abs(a))

def from_log(a:NDArray)->NDArray:
    return  np.sign(a) * (np.expm1(np.abs(a)))
