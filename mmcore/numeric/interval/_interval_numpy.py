"""
Structured-dtype helpers for interval arrays (portable, fast, no custom DType).

- Storage: two float64 fields named 'low' and 'upp'.
- Safe to use with `np.zeros`, `np.ones`, `np.full`, etc.
- Provides helpers to construct arrays and convert to/from Python Interval objects.

This does not make `dtype=Interval` work directly. It provides a concrete
NumPy dtype (`interval_dtype`) and simple utilities around it.
"""
from __future__ import annotations

from typing import Iterable, Tuple, Sequence

import numpy as np

try:
    # Local import to avoid circulars
    from ._interval import Interval
except Exception:  # pragma: no cover - during early build stages
    Interval = None  # type: ignore


# Structured dtype: 16 bytes per element (two float64)
interval_dtype = np.dtype([("low", "<f8"), ("upp", "<f8")])


def _pack_one(x) -> Tuple[float, float]:
    """Return (low, upp) for a value that can denote an interval.

    Accepts:
      - Interval
      - 2-tuple/list of numbers
      - scalar number (degenerate interval)
    """
    if Interval is not None and isinstance(x, Interval):
        return float(x.low), float(x.upp)
    if isinstance(x, (tuple, list)) and len(x) == 2:
        lo, hi = x
        lo = float(lo); hi = float(hi)
        if lo > hi:
            lo, hi = hi, lo
        return lo, hi
    # scalar → degenerate interval
    v = float(x)
    return v, v


def interval_zeros(shape) -> np.ndarray:
    """Zeros array with structured interval dtype.

    Example: arr = interval_zeros(10)
    """
    return np.zeros(shape, dtype=interval_dtype)


def interval_full(shape, value) -> np.ndarray:
    """Full array with the same interval value in every slot.

    Note: this is safe (distinct elements) because it fills structured scalars,
    not object references.
    """
    lo, hi = _pack_one(value)
    arr = np.empty(shape, dtype=interval_dtype)
    arr["low"].fill(lo)
    arr["upp"].fill(hi)
    return arr


def from_intervals(items: Sequence) -> np.ndarray:
    """Create a structured array from a sequence of Interval-like items.

    Each element can be Interval, (low, upp), or a scalar.
    """
    data = np.asarray([_pack_one(x) for x in items], dtype="<f8")
    return data.view(interval_dtype).reshape(data.shape[:-1])


def to_intervals(arr: np.ndarray) -> np.ndarray:
    """Convert a structured interval array to an object array of Interval.

    Returns a new array of dtype object with freshly constructed Interval
    instances. Requires `Interval` to be importable.
    """
    if arr.dtype != interval_dtype:
        raise TypeError("Expected array with interval_dtype")
    if Interval is None:
        raise RuntimeError("Interval class not available for conversion")
    out = np.empty(arr.shape, dtype=object)
    low = arr["low"].ravel()
    upp = arr["upp"].ravel()
    flat = out.ravel()
    for i in range(flat.size):
        flat[i] = Interval(low[i], upp[i])
    return out


def view2(arr: np.ndarray) -> np.ndarray:
    """View a structured interval array as a float64 array with last dim 2.

    This is useful for vectorized math in Cython or NumPy when you wish to
    operate on the two components directly.
    """
    if arr.dtype != interval_dtype:
        raise TypeError("Expected array with interval_dtype")
    v = arr.view("<f8").reshape(arr.shape + (2,))
    return v


__all__ = [
    "interval_dtype",
    "interval_zeros",
    "interval_full",
    "from_intervals",
    "to_intervals",
    "view2",
]

