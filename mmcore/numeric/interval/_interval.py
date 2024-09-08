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

    __slots__ = ("low", "upp")

    def __init__(self, low, upp=None):
        if isinstance(low, tuple) and upp is None:
            self.low, self.upp = low
        elif upp is None:
            self.low = self.upp = low
        else:
            self.low = low
            self.upp = upp


    def __repr__(self):
        return f"Interval({self.low}, {self.upp})"

    def __add__(self, other):
        if isinstance(other, self.__class__):
            return Interval(self.low + other.low, self.upp + other.upp)
        return Interval(self.low + other, self.upp + other)

    def __radd__(self, other):
        if isinstance(other, self.__class__):
            return Interval(self.low + other.low, self.upp + other.upp)
        return Interval(self.low + other, self.upp + other)

    def __sub__(self, other):
        if isinstance(other, Interval):
            return Interval(self.low - other.low, self.upp - other.upp)
        return Interval(self.low - other, self.upp - other)

    def __mul__(self, other):
        if isinstance(other, Interval):
            products = [self.low * other.low, self.low * other.upp, self.upp * other.low, self.upp * other.upp]
            return Interval(min(products), max(products))
        return Interval(self.low * other, self.upp * other)

    def __truediv__(self, other):
        if isinstance(other, Interval):

            reciprocals = [1 / other.low, 1 / other.upp]
            return self * Interval(min(reciprocals), max(reciprocals))
        return Interval(self.low / other, self.upp / other)

    def __contains__(self, item):
        if isinstance(item, Interval):
            return self.low <= item.low and self.upp >= item.upp
        return self.low <= item <= self.upp

    def compare(self, other):
        if isinstance(other, Interval):
            if self.upp < other.low:
                return Comparison.TRUE  # Definitely less than
            elif self.low > other.upp:
                return Comparison.FALSE  # Definitely greater than
            else:
                return Comparison.MAYBE  # Overlapping intervals, uncertain comparison
        return Comparison.TRUE if self.upp < other else Comparison.FALSE

    def __lt__(self, other):
        comp = self.compare(other)
        return comp == Comparison.TRUE  # Only return True if it's definitively less

    def __le__(self, other):
        comp = self.compare(other)
        return comp in (Comparison.TRUE, Comparison.MAYBE)  # True if less or uncertain

    def __gt__(self, other):
        comp = self.compare(other)
        return comp == Comparison.FALSE  # Only return True if it's definitively greater

    def __ge__(self, other):
        comp = self.compare(other)
        return comp in (Comparison.FALSE, Comparison.MAYBE)  # True if greater or uncertain



    def lower(self):
        return Interval(self.low, self.low)

    def upper(self):
        return Interval(self.upp, self.upp)

    def _subdivide_step(self):
        mid = (self.low + self.upp) * 0.5

        return Interval(self.low, mid), Interval(
            mid, self.upp
        )

    def subdivide(self, steps=1):
        res = [self]
        for _ in range(steps):
            subd = []
            for j in res:
                subd.extend(j._subdivide_step())
            res = subd
        return res

    def evaluate(self, t):
        return self.low + (self.upp - self.low) * t

    def __and__(self, other):
        if isinstance(other, Interval):
            return Interval(max(self.low, other.low), min(self.upp, other.upp))
        return Interval(max(self.low, other), min(self.upp, other))

    def __or__(self, other):
        if isinstance(other, Interval):
            return Interval(min(self.low, other.low), max(self.upp, other.upp))
        return Interval(min(self.low, other), max(self.upp, other))

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


    def __iadd__(self, other):
        result = self + other
        self.low, self.upp = result.low, result.upp
        return self

    def __isub__(self, other):
        result = self - other
        self.low, self.upp = result.low, result.upp
        return self

    def __imul__(self, other):
        result = self * other
        self.low, self.upp = result.low, result.upp
        return self

    def __itruediv__(self, other):
        result = self / other
        self.low, self.upp = result.low, result.upp
        return self

    def __float__(self):
        return (self.low + self.upp) / 2

    @classmethod
    def from_tuple(cls, t):

        return cls(cls.from_tuple(t[0]), cls.from_tuple(t[1]))

    def to_tuple(self):


        return (self.low, self.upp)
    def __iter__(self):
        return iter(self.to_tuple())

    def __array__(self, dtype=None):
        return np.array(self.to_tuple(), dtype=dtype)

    def __pow__(self, exp):
        if exp % 2 == 0:
            return Interval(min(self.low ** exp, self.upp ** exp), max(self.low ** exp, self.upp ** exp))
        else:
            return Interval(self.low ** exp, self.upp ** exp)