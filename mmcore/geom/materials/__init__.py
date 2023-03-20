import json
from collections import namedtuple
from typing import Any, SupportsIndex

from mmcore.baseitems import Matchable
from mmcore.baseitems.descriptors import DataView

ColorRGBA = namedtuple("ColorRGBA", ["r", "g", "b", "a"])
_ColorRGB = namedtuple("ColorRGB", ["r", "g", "b"])

import numpy as np


class ColorRGB(tuple):

    def __new__(cls, *iterable, **kwargs):
        if all(map(lambda x: isinstance(x, int), iterable)):
            inst = tuple.__new__(cls, iterable)
            inst.r, inst.g, inst.b = iterable[:3]
            return inst
        else:

            ss = np.array(iterable)
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


def rgb_to_three_decimal(color: ColorRGB | ColorRGBA) -> int:
    return int('%02x%02x%02x' % (color.r, color.g, color.b), 16)


class MaterialData(DataView):
    def item_model(self, name, value):
        return name, value

    def data_model(self, instance, value):
        return dict(value)


class Material(Matchable):
    properties = "uuid", "type", "color"
    __match_args__ = "color",
    data: dict = MaterialData(*properties)
    _color: ColorRGB | ColorRGBA = ColorRGB(50, 50, 50)

    def __new__(cls, color, **kwargs):

        if len(color) == 4:
            cls.properties.extend(['opacity', 'transparent'])
        indt = super().__new__(cls)
        indt.__init__(*(color,), **kwargs)
        return indt

    @property
    def opacity(self) -> float | None:
        try:
            return (1 / 255) * self._color.a
        except AttributeError:
            return None

    @property
    def transparent(self) -> bool | None:
        try:
            return self.opacity is None
        except AttributeError:
            return None

    @property
    def type(self):
        return self.__class__._type

    @property
    def color(self):
        return rgb_to_three_decimal(self._color)

    @color.setter
    def color(self, v: ColorRGB | ColorRGBA | tuple):
        if isinstance(v, tuple):
            if len(v) == 3:
                self._color = ColorRGB(*v)
            elif len(v) == 4:
                self._color = ColorRGBA(*v)
            else:
                pass

        else:
            self._color = v

    def ToJSON(self, *args, **kwargs):
        return json.dumps(self.data, *args, **kwargs)


class MaterialType(type):
    @classmethod
    def __prepare__(metacls, name, bases=(Material,), templates=f"{__file__.replace('/__init__.py', '')}/templates",
                    **kwargs):
        ns = dict(super().__prepare__(name, bases))

        with open(f"{templates}/{name}.json") as f:
            data = json.load(f)

        ns.update(data)
        ns["_type"] = data["type"]
        ns['properties'] = tuple(set(("uuid",
                                      "type",
                                      "color") + tuple(data.keys())))
        ns['data'] = MaterialData(*ns['properties'])

        return ns

    def __new__(mcs, name, bases, dct, **kwargs):
        dct |= kwargs
        cls = super().__new__(mcs, name, bases, dct)

        return cls


class MeshPhongBasic(Material, metaclass=MaterialType):
    ...


class MeshPhongFlatShading(Material, metaclass=MaterialType):
    ...


class MeshPhysicalBasic(Material, metaclass=MaterialType):
    ...


class MeshPhysicalMetallic(Material, metaclass=MaterialType):
    ...


class PointsBase(Material, metaclass=MaterialType):
    ...
