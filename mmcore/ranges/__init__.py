import operator
from enum import Enum
from functools import total_ordering

import numpy as np


def remap(x, source, target):
    return np.interp(x, source, target)


remap = np.vectorize(remap, signature='(),(i),(i)->()')


@total_ordering
class Range(tuple):
    def __new__(cls, start, stop):
        dat = tuple(sorted([start, stop]))
        if dat not in RANGES:
            RANGES[dat] = super().__new__(cls, dat)

        return RANGES[dat]

    def __str__(self):
        return f'Range({self[0]} ... {self[1]})'

    def __repr__(self):
        return f'Range({self[0]}, {self[1]}) at {hex(id(self))}'

    def __contains__(self, item):
        if isinstance(item, Range):
            return all((self[0] <= item[0], item[0] <= self[1]))
        return self[0] <= item <= self[1]

    def __lt__(self, other):
        if isinstance(other, Range):
            return self[0] < other[0] and self[1] < other[1]

        else:
            return self[1] < other

    def __le__(self, other):
        if isinstance(other, Range):
            return self[0] <= other[0] and self[1] <= other[1]

        else:
            return any((self[0] <= other, self[1] < other))

    def divide(self, num: int = 10):
        return np.linspace(*self, num)

    def remap(self, target: 'Range'):
        return RangeMap(self, target)

    def __sub__(self, other):
        if isinstance(other, Range):
            return Range(self[0] - other[0], self[1] - other[1])
        return Range(self[0] - other, self[1] - other)

    def __mul__(self, other: float):
        if isinstance(other, Range):

            return Range(self[0] * other[0], self[1] * other[1])
        else:
            return Range(self[0] * other, self[1] * other)

    def __eq__(self, other):
        return self[0] == other[0] and self[1] == other[1]

    def __hash__(self):
        return hash(tuple(self))

    def __array__(self):
        return np.array([self[0], self[1]])

    @classmethod
    def from_data(cls, data):
        return cls(np.min(data), np.max(data))


class RangeMap:
    __slots__ = ('source', 'target')

    def __new__(cls, source: Range, target: Range):
        if (source, target) not in RANGE_MAPS.keys():
            self = super().__new__(cls)
            self.source = source
            self.target = target
            RANGE_MAPS[(source, target)] = self
            return self
        else:
            return RANGE_MAPS[(source, target)]

    def __call__(self, t):
        return remap(t, np.array(self.source), np.array(self.target))

    def __eq__(self, other):
        return all([other.source == self.source, other.target == self.target])

    def __hash__(self):
        return hash((self.source, self.target))


@total_ordering
class Range2D(Range):
    def __new__(cls, start, stop):
        if not isinstance(start, Range):
            start = Range(*start)
        if not isinstance(stop, Range):
            stop = Range(*stop)

        return super().__new__(cls, start, stop)

    def __contains__(self, item):
        u, v = item
        return u in self[0], v in self[1]

    def __lt__(self, other):
        u, v = other
        return operator.lt(self[0], u) and operator.lt(self[1], v)

    def __le__(self, other):
        u, v = other
        return operator.le(self[0], u) and operator.le(self[1], v)

    def __eq__(self, other):
        u, v = other
        return operator.eq(self[0], u) and operator.eq(self[1], v)

    def __gt__(self, other):
        u, v = other
        return operator.gt(self[0], u) and operator.gt(self[1], v)

    @classmethod
    def from_data(cls, data):
        raise NotImplementedError

    def divide(self, num_u: int = 10, num_v: int = 10):
        return np.meshgrid(self[0].divide(num_u), self[1].divide(num_v))

    def remap(self, target: 'Range2D'):

        return RangeMap(self, target)

    def __hash__(self):
        return hash((self[0], self[1]))


RANGES = dict()
RANGE_MAPS = dict()


class RangeEnum(Range, Enum):
    POLAR = Range(0.0, np.pi * 2)
    UNIT = Range(0.0, 1.0)
    EMPTY = Range(0.0, 0.0)


polar_to_unit = RangeEnum.POLAR.remap(RangeEnum.UNIT)
unit_to_polar = RangeEnum.UNIT.remap(RangeEnum.POLAR)
