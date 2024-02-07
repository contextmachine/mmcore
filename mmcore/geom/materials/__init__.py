import functools
import json
import typing
from collections import namedtuple
from typing import Any, SupportsIndex

import numpy as np

# ColorRGBA = namedtuple("ColorRGBA", ["r", "g", "b", "a"])
# _ColorRGB = namedtuple("ColorRGB", ["r", "g", "b"])

cmap = dict()


class ColorRGB(tuple):
    """
    >>> from mmcore.geom.materials import ColorRGB
    >>> c=ColorRGB(0.5,0.5,0.5)
    >>> str(c)
    Out[4]: '#808080'
    >>> int(c)
    Out[5]: 8421504
    >>> np.array(c,float)
    Out[6]: array([0.50196078, 0.50196078, 0.50196078])
    >>> np.array(c,int)
    Out[7]: array([128, 128, 128])
    """

    def __new__(cls, *iterable, **kwargs):

        if len(iterable) == 1:
            iterable, = iterable
        if all(map(lambda x: isinstance(x, int), iterable)):
            inst = tuple.__new__(cls, iterable)
            inst.r, inst.g, inst.b = iterable[:3]
            return inst
        else:
            ss = np.array(list(iterable))
            dd = np.array(np.round(ss * 255), dtype=int).tolist()

            inst = tuple.__new__(cls, tuple(dd))
            inst.r, inst.g, inst.b = tuple(dd)
            return inst

    @property
    def rgb_0to1(self):
        return np.array([self.r / 255, self.g / 255, self.b / 255], float)

    @property
    def rgb_0to255(self):
        return np.array([self.r, self.g, self.b], int)

    @property
    def decimal(self):
        return int(self.hex(), 16)

    def __str__(self):
        return "#" + self.hex()

    def __int__(self):
        return int(self.decimal)

    def __hash__(self):
        return hash((self.r, self.g, self.b))

    def hex(self):
        return "%02x%02x%02x" % (self.r, self.g, self.b)

    def index(
        self, __value: Any, __start: SupportsIndex = ..., __stop: SupportsIndex = ...
    ) -> int:
        return tuple.index(self, __value, __start, __stop)

    def __getitem__(self, item):
        return tuple.__getitem__(self, item)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.r}, {self.g}, {self.b})"

    def to_rhino(self):
        return (self.r, self.g, self.b, 255)

    def todict(self):
        return {"r": self.r, "g": self.g, "b": self.b, "a": 255}

    def ToJSON(self):
        return json.dumps(self.todict())

    @classmethod
    def random(cls, size=1):
        if size == 1:
            return ColorRGB(*np.random.randint(0, 255, 3) / 255)
        else:
            return [
                ColorRGB(*col) for col in (np.random.randint(0, 255, (size, 3)) / 255)
            ]

    def __array__(self, dtype=float):
        if np.dtype(dtype).kind in ["i", "u"]:
            return np.array(self.rgb_0to255, dtype=dtype)
        elif np.dtype(dtype).kind in ["f", "d"]:
            return np.array(self.rgb_0to1, dtype=dtype)
        elif np.dtype(dtype).kind in ["U"]:
            return np.array(self.hex(), dtype=np.str_)

    @classmethod
    def from_hex(cls, h: str):
        h = h.lstrip("#")
        return cls(*tuple(int(h[i : i + 2], 16) for i in (0, 2, 4)))


class ColorRGBA(ColorRGB):
    def __new__(cls, r=0, g=0, b=0, a=1.0, **kwargs):
        obj = super().__new__(cls, r, g, b, **kwargs)
        obj.a = a
        return obj

    def todict(self):
        self.s
        return {"r": self.r, "g": self.g, "b": self.b, "a": 255}
def rgb_to_three_decimal(color: typing.Union[ColorRGB, ColorRGBA]) -> int:
    return int("%02x%02x%02x" % (color.r, color.g, color.b), 16)
