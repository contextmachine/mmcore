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

    def intersect(self,other):
        lo = max(self.low, other.low)
        hi = min(self.upp, other.upp)
        return None if lo>hi else Interval(lo,hi)
    # merging
    def hull(self,other):
        return Interval(min(self.low,other.low), max(self.upp,other.upp))
    def __and__(self, other):
        return self.intersect(other)

    def evaluate(self, t):
        return self.low + (self.upp - self.low) * t

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

    #def __array__(self, dtype=None):
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
        elif ufunc is np.negative:
            result = -args[0]
        else:
            return NotImplemented

        # Handle `out` keyword (only single-output ufuncs here)
        if kwargs.get("out") is not None:
            (out_arr,) = kwargs["out"]
            out_arr[...] = result
            return out_arr
        return result


# ───────────────────────────────────────────────────────────────
#  0.  One–dimensional interval type  (your class + small fixes)
# ───────────────────────────────────────────────────────────────
from enum import Enum
from functools import total_ordering
import math, itertools
# … ⟨-- paste your original Interval class here, but add three tiny things
#      * a correct even-power rule (handles the “crosses 0” case)
#      * unary minus  __neg__
#      * _subdivide_step  (mid-point bisection) ⟩
#
#  All other operators stay exactly as you wrote them
# ───────────────────────────────────────────────────────────────


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
