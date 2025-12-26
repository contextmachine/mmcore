import operator
from enum import Enum
from functools import total_ordering
import math


import numpy as np

class Comparison(int,Enum):
    TRUE = 1
    FALSE = 0
    MAYBE = -1
@total_ordering
class Interval:
    """
    Interval class for representing and operating on intervals.
    """
    # Tell NumPy to give Interval methods precedence in mixed expressions
    __array_priority__ = 1_000.0

    __slots__ = ("low", "upp")

    def __init__(self, low, upp=None):

        if isinstance(low,tuple) and upp is None:
            low,upp=low
        elif upp is None:
            upp=low
        if low>upp: low,upp=upp,low
        self.low=float(low); self.upp=float(upp)
    def width(self): return self.upp-self.low
    def mid(self): return (self.low+self.upp)/2
    def _subdivide_step(self):
        m = self.mid()
        return Interval(self.low, m), Interval(m, self.upp)

    # ---------------------------------------------------- string / repr / hash
    def __repr__(self):
        return f"Interval({self.low:g}, {self.upp:g})"

    def __str__(self):
        return f"[{self.low:g}, {self.upp:g}]"

    def __hash__(self):
        return hash((self.low, self.upp))

    # -------------------------------------------------------- arithmetic ops
    def __add__(self, other):
        if isinstance(other, Interval):
            return Interval(self.low + other.low, self.upp + other.upp)
        return Interval(self.low + other, self.upp + other)

    __radd__=__add__

    def __sub__(self, other):
        if isinstance(other, Interval):
            return Interval(self.low - other.upp, self.upp - other.low)
        return Interval(self.low - other, self.upp - other)

    def __rsub__(self, other):
        if isinstance(other, Interval):
            return Interval(other.low - self.upp, other.upp - self.low)
        return Interval(other - self.upp, other - self.low)

    def __mul__(self, other):
        if isinstance(other, Interval):
            a, b, c, d = self.low, self.upp, other.low, other.upp
            products = (a * c, a * d, b * c, b * d)
            return Interval(min(products), max(products))
        return Interval(self.low * other, self.upp * other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Interval):
            if other.low <= 0 <= other.upp:
                raise ZeroDivisionError("Division by interval spanning zero.")
            return self * Interval(1.0 / other.upp, 1.0 / other.low)
        return Interval(self.low / other, self.upp / other)

    def __rtruediv__(self, other):
        if self.low <= 0 <= self.upp:
            raise ZeroDivisionError("Division by interval spanning zero.")
        if isinstance(other, Interval):
            return other.__truediv__(self)
        return Interval(other / self.upp, other / self.low)

    def __pow__(self, n: int):
        if not isinstance(n, int):
            raise TypeError("Exponent must be an integer.")
        if n == 0:
            return Interval(1.0)
        if n % 2 == 0:
            lo = min(self.low ** n, self.upp ** n)
            hi = max(self.low ** n, self.upp ** n)
            if self.low <= 0 <= self.upp:
                lo = 0.0
            return Interval(lo, hi)
        return Interval(self.low ** n, self.upp ** n)

    def __rpow__(self, base):
        return Interval(base ** self.low, base ** self.upp)

    # in-place variants
    def __iadd__(self, other):
        r = self + other
        self.low, self.upp = r.low, r.upp
        return self

    def __isub__(self, other):
        r = self - other
        self.low, self.upp = r.low, r.upp
        return self

    def __imul__(self, other):
        r = self * other
        self.low, self.upp = r.low, r.upp
        return self

    def __itruediv__(self, other):
        r = self / other
        self.low, self.upp = r.low, r.upp
        return self

    # unary
    def __neg__(self):
        return Interval(-self.upp, -self.low)

    def __pos__(self):
        return self

    # ------------------------------------------------------------ comparisons
    def compare(self, other):
        if isinstance(other, Interval):
            if self.upp < other.low:
                return Comparison.TRUE
            if self.low > other.upp:
                return Comparison.FALSE
            return Comparison.MAYBE
        return Comparison.TRUE if self.upp < other else Comparison.FALSE

    def __lt__(self, other):
        return self.compare(other) == Comparison.TRUE

    def __le__(self, other):
        return self.compare(other) != Comparison.FALSE

    def __gt__(self, other):
        return self.compare(other) == Comparison.FALSE

    def __ge__(self, other):
        return self.compare(other) != Comparison.TRUE

    def __eq__(self, other):
        if isinstance(other, Interval):
            return (self.low, self.upp) == (other.low, other.upp)
        return self.low == other and self.upp == other

    # --------------------------------------------------------- set-like ops
    def __contains__(self,x):
        if isinstance(x,Interval):
            return self.low<=x.low and self.upp>=x.upp
        return self.low<=x<=self.upp
    def contains(self, other, low_inclusive = True, up_inclusive = True):
        if up_inclusive:
            comp_upp=operator.ge
        else:
            comp_upp=operator.gt
        if low_inclusive:
            comp_low=operator.le
        else:
            comp_low=operator.lt
        if isinstance(other, Interval):

            return comp_low(self.low,other.low) and comp_upp(self.upp,other.upp)
        return comp_low(self.low,other) and comp_upp(self.upp,other)


    def intersect(self,other):

        lo = max(self.low, other.low)
        hi = min(self.upp, other.upp)
        return None if lo>hi else Interval(lo,hi)

    def intersects(self,other)->bool:
        if isinstance(other, Interval):
            lo = max(self.low, other.low)
            hi = min(self.upp, other.upp)
            return False if lo>hi else True
        lo = max(self.low, other)
        hi = min(self.upp, other)
        return False if lo > hi else True
        # merging
    def hull(self,other):
        return Interval(min(self.low,other.low), max(self.upp,other.upp))
    def __and__(self, other):
        return self.intersect(other)

    def evaluate(self, t):
        return self.low + (self.upp - self.low) * t
    def expand(self,other):
        if isinstance(other, Interval):
            self.low,self.upp=min(self.low,other.low), max(self.upp,other.upp)
        else:
            self.low,self.upp = min(self.low,other), max(self.upp,other)
    def split(self, t: float) -> tuple["Interval", "Interval"]:
        """
        Split the interval by a normalized parameter t in [0, 1].

        Returns a pair (left, right) such that:
        - left = [low, low + t*(upp-low)]
        - right = [low + t*(upp-low), upp]

        The split is clamped to the [0, 1] range and endpoints are
        computed in a way that preserves containment under floating
        point round-off (clamped into [low, upp]).
        """
        try:
            tt = float(t)
        except Exception as exc:
            raise TypeError("t must be a real number in [0, 1]") from exc

        if not math.isfinite(tt):
            raise ValueError("t must be finite and in [0, 1]")

        # Clamp t into [0, 1] to be robust to minor numeric noise
        if tt <= 0.0:
            m = self.low
        elif tt >= 1.0:
            m = self.upp
        else:
            m = self.evaluate(tt)
            # Clamp midpoint into the interval to avoid any drift
            if m < self.low:
                m = self.low
            elif m > self.upp:
                m = self.upp

        # Outward rounding to maintain enclosure under FP arithmetic
        if 0.0 < tt < 1.0 and self.low < m < self.upp:
            left_upp = np.nextafter(m, float("inf"))
            right_low = np.nextafter(m, float("-inf"))
        else:
            left_upp = right_low = m

        return Interval(self.low, left_upp), Interval(right_low, self.upp)

    def __or__(self, other):
        return self.hull(other)

    def __invert__(self):
        if self.low == 0 or self.upp == 0:
            raise ZeroDivisionError("Division by zero is undefined.")

        if self.low < 0 < self.upp:
            # Return an interval that represents the reciprocal extending to infinity
            return Interval(float('-inf'), float('inf'))

        # Swap bounds for the reciprocal
        return Interval(1 / self.upp, 1 / self.low)

    def merge(self, other):
        return Interval(min(self.low, other.low), max(self.upp, other.upp))

    def ulp(self):
        return math.ulp(self.low)

    @staticmethod
    def max(*args):
        return max(args)

    @staticmethod
    def min(*args):
        return min(args)

    @staticmethod
    def pow(base, exp):
        if exp % 2 == 0:
            return Interval(min(base.low ** exp, base.upp ** exp), max(base.low ** exp, base.upp ** exp))
        else:
            return Interval(base.low ** exp, base.upp ** exp)

    def __float__(self):
        return (self.low + self.upp) / 2

    @classmethod
    def from_tuple(cls, t):

        return cls(cls.from_tuple(t[0]), cls.from_tuple(t[1]))

    def to_tuple(self):

        return (self.low, self.upp)
    def __iter__(self):
        return iter(self.to_tuple())

    # def __array__(self, dtype=None):
    #    return np.array(self.to_tuple(), dtype=dtype)

    # def __pow__(self, exp):
    #    if exp % 2 == 0:
    #        return Interval(min(self.low ** exp, self.upp ** exp), max(self.low ** exp, self.upp ** exp))
    #    else:
    #        return Interval(self.low ** exp, self.upp ** exp)
    # ordering
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Minimal ufunc override so that **scalar** Interval objects participate
        in NumPy expressions.  Object-dtype arrays already work without this.
        """
        if method != "__call__":
            return NotImplemented

        # Extract operands as Interval objects
        args = [x if isinstance(x, Interval) else Interval(x) for x in inputs]

        # Helpers for unary monotone functions
        def inc(fn, a: Interval):
            return Interval(fn(a.low), fn(a.upp))

        def dec(fn, a: Interval):
            return Interval(fn(a.upp), fn(a.low))

        def abs_iv(a: Interval):
            if a.low >= 0:
                return Interval(a.low, a.upp)
            if a.upp <= 0:
                return Interval(-a.upp, -a.low)
            return Interval(0.0, max(-a.low, a.upp))

        def sign_iv(a: Interval):
            if a.low > 0:
                return Interval(1.0, 1.0)
            if a.upp < 0:
                return Interval(-1.0, -1.0)
            if a.low == 0 and a.upp == 0:
                return Interval(0.0, 0.0)
            # crosses or touches zero → could be -1, 0, or 1
            return Interval(-1.0, 1.0)

        def sqrt_iv(a: Interval):
            if a.upp < 0:
                raise ValueError("sqrt domain requires x >= 0")
            lo = max(0.0, a.low)
            return Interval(math.sqrt(lo), math.sqrt(a.upp))

        def log_iv(a: Interval, base=None):
            if a.upp <= 0:
                raise ValueError("log domain requires x > 0")
            lo = a.low
            if lo <= 0:
                lo = np.nextafter(0.0, 1.0)
            if base is None:
                f = math.log
            else:
                f = lambda x: math.log(x, base)
            return Interval(f(lo), f(a.upp))

        def sin_iv(a: Interval):
            a0, b0 = a.low, a.upp
            if b0 - a0 >= 2 * math.pi:
                return Interval(-1.0, 1.0)
            cands = [math.sin(a0), math.sin(b0)]
            # critical points: pi/2 + k*pi
            kmin = math.ceil((a0 - math.pi / 2) / math.pi)
            kmax = math.floor((b0 - math.pi / 2) / math.pi)
            for k in range(int(kmin), int(kmax) + 1):
                # sin at these points is ±1
                cands.append(1.0 if (k % 2 == 0) else -1.0)
            return Interval(min(cands), max(cands))

        def cos_iv(a: Interval):
            a0, b0 = a.low, a.upp
            if b0 - a0 >= 2 * math.pi:
                return Interval(-1.0, 1.0)
            cands = [math.cos(a0), math.cos(b0)]
            # critical points: k*pi
            kmin = math.ceil(a0 / math.pi)
            kmax = math.floor(b0 / math.pi)
            for k in range(int(kmin), int(kmax) + 1):
                cands.append(1.0 if (k % 2 == 0) else -1.0)
            return Interval(min(cands), max(cands))

        def tan_iv(a: Interval):
            a0, b0 = a.low, a.upp
            # check asymptote crossings at pi/2 + k*pi
            kmin = math.ceil((a0 - math.pi / 2) / math.pi)
            kmax = math.floor((b0 - math.pi / 2) / math.pi)
            if kmin <= kmax:
                return Interval(float('-inf'), float('inf'))
            va, vb = math.tan(a0), math.tan(b0)
            return Interval(min(va, vb), max(va, vb))

        def asin_iv(a: Interval):
            lo = max(-1.0, a.low)
            hi = min(1.0, a.upp)
            if lo > hi:
                raise ValueError("arcsin domain requires x in [-1, 1]")
            return Interval(math.asin(lo), math.asin(hi))

        def acos_iv(a: Interval):
            lo = max(-1.0, a.low)
            hi = min(1.0, a.upp)
            if lo > hi:
                raise ValueError("arccos domain requires x in [-1, 1]")
            # decreasing on [-1,1]
            return Interval(math.acos(hi), math.acos(lo))

        def atan_iv(a: Interval):
            return Interval(math.atan(a.low), math.atan(a.upp))

        def sinh_iv(a: Interval):
            return Interval(math.sinh(a.low), math.sinh(a.upp))

        def cosh_iv(a: Interval):
            # cosh is even, minimum at 0
            lo = 1.0 if (a.low <= 0.0 <= a.upp) else min(math.cosh(a.low), math.cosh(a.upp))
            hi = max(math.cosh(a.low), math.cosh(a.upp))
            return Interval(lo, hi)

        def tanh_iv(a: Interval):
            return Interval(math.tanh(a.low), math.tanh(a.upp))

        if ufunc is np.add:
            result = args[0] + args[1]
        elif ufunc is np.subtract:
            result = args[0] - args[1]
        elif ufunc is np.multiply:
            result = args[0] * args[1]
        elif ufunc in (np.divide, np.true_divide):
            result = args[0] / args[1]
        elif ufunc is np.power:
            # exponent may be scalar
            exp = inputs[1]
            if isinstance(exp, Interval):
                return NotImplemented  # interval**interval not defined
            result = args[0] ** int(exp)
        elif ufunc is np.square:
            result = args[0] ** 2
        elif ufunc is np.sqrt:
            result = sqrt_iv(args[0])
        elif ufunc in (np.log, np.log2, np.log10):
            if ufunc is np.log:
                result = log_iv(args[0])
            elif ufunc is np.log2:
                result = log_iv(args[0], base=2)
            else:
                result = log_iv(args[0], base=10)
        elif ufunc is np.expm1:
            result = inc(math.expm1, args[0])
        elif ufunc is np.log1p:
            # domain x > -1
            a = args[0]
            if a.upp <= -1.0:
                raise ValueError("log1p domain requires x > -1")
            lo = a.low
            if lo <= -1.0:
                lo = np.nextafter(-1.0, 1.0)
            result = Interval(math.log1p(lo), math.log1p(a.upp))
        elif ufunc is np.exp:
            result = inc(math.exp, args[0])
        elif ufunc is np.abs or ufunc is np.absolute or ufunc is np.fabs:
            result = abs_iv(args[0])
        elif ufunc is np.negative:
            result = -args[0]
        elif ufunc is np.positive:
            result = args[0]
        elif ufunc is np.floor:
            result = Interval(math.floor(args[0].low), math.floor(args[0].upp))
        elif ufunc is np.ceil:
            result = Interval(math.ceil(args[0].low), math.ceil(args[0].upp))
        elif ufunc is np.trunc:
            result = Interval(math.trunc(args[0].low), math.trunc(args[0].upp))
        elif ufunc is np.reciprocal:
            result = ~args[0]
        elif ufunc is np.minimum or ufunc is np.fmin:
            a, b = args
            result = Interval(min(a.low, b.low), min(a.upp, b.upp))
        elif ufunc is np.maximum or ufunc is np.fmax:
            a, b = args
            result = Interval(max(a.low, b.low), max(a.upp, b.upp))
        elif ufunc is np.sin:
            result = sin_iv(args[0])
        elif ufunc is np.cos:
            result = cos_iv(args[0])
        elif ufunc is np.tan:
            result = tan_iv(args[0])
        elif ufunc is np.arcsin:
            result = asin_iv(args[0])
        elif ufunc is np.arccos:
            result = acos_iv(args[0])
        elif ufunc is np.arctan:
            result = atan_iv(args[0])
        elif ufunc is np.sinh:
            result = sinh_iv(args[0])
        elif ufunc is np.cosh:
            result = cosh_iv(args[0])
        elif ufunc is np.tanh:
            result = tanh_iv(args[0])
        else:
            return NotImplemented

        # Handle `out` keyword (only single-output ufuncs here)
        if kwargs.get("out") is not None:
            (out_arr,) = kwargs["out"]
            out_arr[...] = result
            return out_arr
        return result


import math

# ───────────────────────────────────────────────────────────────
# 1.  A light “box” wrapper for ℝⁿ intervals
# ───────────────────────────────────────────────────────────────
class IntervalND:
    """
    Axis-aligned box in ℝⁿ:  [x0_low,x0_up] × … × [xn_low,xn_up]
    Implemented as a list of Interval objects.
    """
    def __init__(self, intervals):
        self.iv = list(intervals)               # List[Interval]
        self._low=np.array([i.low for i in self.iv  ])
        self._upp=np.array([i.upp for i in self.iv  ])
    @property
    def low(self):
        return  self._low

    @property
    def upp(self):
        return self._upp
    # helpers ---------------------------------------------------
    def mid(self):
        return [r.mid() for r in self.iv]       # centre point (float list)
    def width(self):
        return max(r.width() for r in self.iv)  # max edge length
    def copy(self):
        return IntervalND([Interval(r.low, r.upp) for r in self.iv])

    # intersection "&" -----------------------------------------
    def __and__(self, other):
        new = [a & b for a, b in zip(self.iv, other.iv)]
        if any(i is None for i in new):         # disjoint → empty
            return None
        return IntervalND(new)

    # preferred splitting:  bisect the *widest* edge -----------
    def bisect(self):
        k = max(range(len(self.iv)), key=lambda i: self.iv[i].width())
        left_i, right_i = self.iv[k]._subdivide_step()
        L = self.copy(); L.iv[k] = left_i
        R = self.copy(); R.iv[k] = right_i
        return L, R
    def __iter__(self):
        return iter(self.iv)
    # cosmetics ------------------------------------------------
    def as_tuple(self):
        return tuple((i.low, i.upp) for i in self.iv)
    def __repr__(self):
        return "IntervalND(" + ", ".join(repr(i) for i in self.iv) + ")"

    def __array__(self, dtype=None,*args,**kwargs):
        return np.array(self.as_tuple(),dtype, *args,**kwargs)


    def subdivide(self):
        widths = [iv.width() for iv in self.iv]
        k = max(range(len(widths)), key=widths.__getitem__)
        l, r = self.iv[k]._subdivide_step()
        left = self.iv.copy();
        left[k] = l
        right = self.iv.copy();
        right[k] = r
        return IntervalND(left), IntervalND(right)


# ------------------------------------------------------------
# Interval list utilities
# ------------------------------------------------------------
from bisect import bisect_left
from typing import List, Optional, Tuple


def insert_interval_sorted(intervals: List[Interval], new_iv: Interval, *, tol: float = 0.0) -> int:
    """
    Insert an interval into a sorted, non-overlapping list of Interval objects.

    The function mutates the input list in-place, preserving the following invariants:
    - List remains sorted by `low`.
    - Overlapping or touching intervals (within `tol`) are merged into a single interval.
    - Any fully covered intervals are replaced by the enveloping interval.

    Parameters
    ----------
    intervals : list[Interval]
        Existing list, sorted by `low`, with no overlapping elements (within `tol`).
    new_iv : Interval
        Interval to insert.
    tol : float, default 0.0
        Tolerance for considering two intervals as touching/overlapping. If
        `A.upp + tol >= B.low` they are merged.

    Returns
    -------
    int
        The index at which the resulting merged interval resides after insertion.

    Notes
    -----
    - Time complexity is O(n) in the worst case (single pass left/right).
    - Works even if `new_iv` is inverted (low > upp) — it is normalized.
    """
    if not intervals:
        intervals.append(Interval(new_iv.low, new_iv.upp))
        return 0

    low = float(new_iv.low)
    upp = float(new_iv.upp)
    if low > upp:
        low, upp = upp, low
    new_low, new_upp = low, upp  # remember original bounds before expansion
    new_low, new_upp = low, upp  # remember original for absorption test

    # Find initial insertion point by lower bound.
    lows = [iv.low for iv in intervals]
    i = bisect_left(lows, low)

    # Expand to include any overlapping/touching intervals on the left.
    start = i
    while start > 0 and intervals[start - 1].upp + tol >= low:
        start -= 1
        low = min(low, intervals[start].low)
        upp = max(upp, intervals[start].upp)

    # Expand to include any overlapping/touching intervals on the right.
    end = i
    while end < len(intervals) and intervals[end].low <= upp + tol:
        low = min(low, intervals[end].low)
        upp = max(upp, intervals[end].upp)
        end += 1

    # Replace the covered span with the merged interval.
    intervals[start:end] = [Interval(low, upp)]
    return start
