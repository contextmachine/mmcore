from functools import total_ordering
import math


import numpy as np


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
            return Interval(self.low * other.low, self.upp * other.upp)
        return Interval(self.low * other, self.upp * other)

    def __truediv__(self, other):
        if isinstance(other, Interval):
            return Interval(self.low / other.low, self.upp / other.upp)
        return Interval(self.low / other, self.upp / other)

    def __contains__(self, item):
        if isinstance(item, Interval):
            return self.low <= item.low and self.upp >= item.upp
        return self.low <= item <= self.upp

    def __le__(self, other):
        if isinstance(other, Interval):
            return self.low <= other.low and self.upp <= other.upp
        return self.upp <= other

    def __lt__(self, other):
        if isinstance(other, Interval):
            return self.low < other.low and self.upp < other.upp
        return self.upp < other

    def __eq__(self, other):
        if isinstance(other, Interval):
            return self.low == other.low and self.upp == other.upp
        return self.low == other and self.upp == other

    def __ne__(self, other):
        return not self == other

    def __ge__(self, other):
        if isinstance(other, Interval):
            return self.low >= other.low and self.upp >= other.upp
        return self.low >= other

    def __gt__(self, other):
        if isinstance(other, Interval):
            return self.low > other.low and self.upp > other.upp
        return self.low > other

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
        return [Interval(float("-inf"), self.low), Interval(self.upp, float("inf"))]

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
        if isinstance(base, Interval):
            if isinstance(exp, int):
                return Interval(base.low**exp, base.upp**exp)
            else:
                return Interval(base.low**exp, base.upp**exp)
        else:
            return base**exp

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

