import json
import typing
from collections import namedtuple
from typing import Any, SupportsIndex

import numpy as np

ColorRGBA = namedtuple("ColorRGBA", ["r", "g", "b", "a"])
_ColorRGB = namedtuple("ColorRGB", ["r", "g", "b"])


class ColorRGB(tuple):

    def __new__(cls, *iterable, **kwargs):
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
    def decimal(self):
        return int('%02x%02x%02x' % (self.r, self.g, self.b), 16)

    def __int__(self):
        return int(self.hex(), 16)

    def hex(self):
        return '%02x%02x%02x' % (self.r, self.g, self.b)

    def index(self, __value: Any, __start: SupportsIndex = ..., __stop: SupportsIndex = ...) -> int:
        return tuple.index(self, __value, __start, __stop)

    def __getitem__(self, item):
        return tuple.__getitem__(self, item)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.r}, {self.g}, {self.b})"

    def to_rhino(self):
        return (self.r, self.g, self.b, 255)

    def to_dict(self):
        return {"r": self.r, "g": self.g, "b": self.b, "a": 255}

    def ToJSON(self):
        return json.dumps(self.to_dict())
    @classmethod
    def random(cls, size=1):
        if size==1:

            return ColorRGB(*np.random.randint(0,255,3)/255)
        else:
            return [ColorRGB(*col) for col in (np.random.randint(0, 255, (size,3)) / 255)]


def rgb_to_three_decimal(color: typing.Union[ColorRGB, ColorRGBA]) -> int:
    return int('%02x%02x%02x' % (color.r, color.g, color.b), 16)

