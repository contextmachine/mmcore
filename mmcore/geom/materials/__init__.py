import json
from collections import namedtuple

from mmcore.baseitems import Mmodel
from mmcore.baseitems.descriptors import DataView

ColorRGBA = namedtuple("ColorRGBA", ["r", "g", "b", "a"])
ColorRGB = namedtuple("ColorRGB", ["r", "g", "b"])


def rgb_to_three_decimal(color: ColorRGB | ColorRGBA) -> int:
    return int('%02x%02x%02x' % (color.r, color.g, color.b), 16)


class MaterialData(DataView):
    def item_model(self, name, value):
        return name, value

    def data_model(self, instance, value):
        return dict(value)


class Material(Mmodel):
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
        except:
            pass

    @property
    def transparent(self) -> bool:
        try:
            return self.opacity < 1.0
        except:
            pass

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
