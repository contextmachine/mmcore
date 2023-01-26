#  Copyright (c) 2022. Computational Geometry, Digital Engineering and Optimizing your construction processe"
from abc import ABC, abstractmethod
from typing import Any, Iterator, Mapping


class AbstractDescriptor(ABC):
    def __set_name__(self, owner, name):
        self.name = name

    @abstractmethod
    def __get__(self, instance, owner):
        ...

    @abstractmethod
    def __set__(self, instance, val):
        ...


class Descriptor(AbstractDescriptor):
    """
        Basiс Descriptor.
        Простейшая общая реализация дескриптора,
        реализующая __get__, __set_name__ и абстрактный __set__.
        Наследуйтесь от этого класса если хотите приготовить что то кастомное.
        Что-то для чего не подходят DataDescriptor, NoDataDescriptor (см. ниже).
    """

    def __set_name__(self, owner, name):
        super().__set_name__(owner, name)

    def __get__(self, instance, owner):

        if instance is None:
            return owner.__getattribute__(self.name)
        else:
            return getattr(instance, self.name)

    @abstractmethod
    def __set__(self, instance, val):
        ...


class NoDataDescriptor(Descriptor):
    """
        Basic No Data Descriptor.
        Простейшая реализация дескриптора НЕ данных
    """

    def __get__(self, instance, owner):
        return super().__get__(instance, owner)

    def __set__(self, instance, val):
        raise AttributeError("NoDataDescriptor does not support attributes __set__")


class DataDescriptor(Descriptor):
    """
    Basic Data Descriptor
    Простейшая реализация дескриптора данных
    """

    def __set__(self, instance, val):
        instance.__setattr__(self.name, val)

    def __get__(self, instance, owner):
        return super().__get__(instance, owner)


class BaseClientDescriptor(Descriptor):

    def __set_name__(self, owner, name):
        super().__set_name__(owner, name)
        self.bucket = owner.bucket
        self.prefix = owner.prefix

    def __get__(self, instance, owner=None):
        return instance.__client__.get_object(Bucket=instance.bucket, Key=f"{instance.prefix}{self.name}")

    def __set__(self, instance, value):
        instance.__client__.put_object(Bucket=instance.bucket, Key=f"{instance.prefix}{self.name}", Body=value)


class HookDescriptor(BaseClientDescriptor):

    def __set__(self, instance, value):
        super(HookDescriptor, self).__set__(instance, instance.__sethook__(value))

    def __get__(self, instance, owner=None):
        return instance.__gethook__(super(HookDescriptor, self).__get__(instance, owner))


class ClientDescriptor(Descriptor):
    def __set_name__(self, owner, name):
        self.bucket = owner.bucket
        self.prefix = owner.prefix
        self.name = name

    def __get__(self, instance, owner=None):
        print(instance, owner)
        return instance.__gethook__(instance.client.get_object(Bucket=self.bucket, Key=f"{self.prefix}{self.name}"))

    def __set__(self, instance, value):
        print(instance, value)
        instance.client.put_object(Bucket=self.bucket, Key=f"{self.prefix}{self.name}{instance.suffix}",
                                   Body=instance.__sethook__(value))


def safe_getattribute(cls, self, item) -> None | Any:
    try:
        return cls.__getattribute__(self, item)
    except AttributeError as err:
        return None


class DataView(NoDataDescriptor):
    """
    >>> class DataOO(DataView):
    ...     def item_model(self, name, value):
    ...         return {"id": name, "value": value}
    ...     def data_model(self, value):
    ...         return {"type":"DataOO","data":value}
    >>> from mmcore.baseitems import Descriptor, NoDataDescriptor, Matchable

    >>> class AA('Matchable'):
    ...     __match_args__="a","b","c"
    ...     compute_data_params=["a","b"]
    ...
    ...     compute_data=DataOO("compute_data_params")
    >>> a = AA(1,2.,"tt")
    >>> a.compute_data
    {'type': 'DataOO',
     'data': [{'id': 'a', 'value': 1}, {'id': 'b', 'value': 2.0}]}

    """

    @abstractmethod
    def item_model(self, name: str, value: Any):
        ...

    @abstractmethod
    def data_model(self, instance, value: list[tuple[str, Any]]):
        ...

    def __init__(self, *targets):
        super().__init__()
        self.targets = targets

    def __set_name__(self, owner, name):
        super().__set_name__(owner, name)

        setattr(owner, "_" + name, self)

    def __generate__(self, instance, owner):
        for name in self.targets:
            yield self.item_model(name=name, value=safe_getattribute(owner, instance, name))

    def __get__(self, instance, owner):
        if instance is None:
            return self
        else:
            return self.data_model(instance, value=list(self.__generate__(instance, owner)))


class UserDataField(DataView):
    @abstractmethod
    def item_model(self, name: str, value: Any):
        ...

    def data_model(self, instance, value: list[None | tuple[str, Any]]):
        if all(list(map(lambda x: x is None, dict(value).values()))):
            return None
        else:
            return dict(value)


class UserDataProperties(UserDataField):
    def item_model(self, name, value):
        return name, value

    def data_model(self, instance, value: list[None | tuple[str, Any]]):
        return super().data_model(instance, value)


class UserData(DataView):
    deps = "properties", "gui"

    def __init__(self, *targets):
        object.__init__(self)
        super().__init__(*(self.deps + targets))

    def item_model(self, name: str, value: Any):

        return name, value

    def data_model(self, instance, value: list[tuple[str, Any]]) -> dict:
        d = {}
        for k, v in value:
            if v is not None:
                d[k] = v
        return d


from enum import Enum


class Template(str):
    type: str

    def __repr__(self):
        return f"{self.type}-template'{super().__repr__()}'"

    def __str__(self):
        return super().__str__()


class ChartTemplate(Template):
    type: str = "chart"


class GuiTemplates(Template, Enum):
    line_chart = ChartTemplate("linechart")
    pie_chart = ChartTemplate("piechart")


class GuiColors(str, Enum):
    default = "default"


class UserDataGuiItem(Mapping, DataDescriptor):
    """
    {
        "id": "color-linechart",
        "name": "график по цветам",
        "type": "chart",
        "key": "color",
        "colors": "default",
        "require": [
          "piechart"
       ]
    }
    """

    def __set_name__(self, owner, name):
        owner.gui.targets.append(name)
        self.name = name

    def __iter__(self) -> Iterator:
        return self._dct().__iter__()

    def __len__(self) -> int:
        return self._dct().__len__()

    id: str
    common: str
    colors: GuiColors = GuiColors.default

    def __init__(self, templates, common="График", **kwargs):
        super().__init__()
        self.common = common
        self.templates = templates
        self.common = common
        self.__dict__ |= kwargs

    def __getitem__(self, item):
        return self._dct().__getitem__(item)

    def _dct(self):
        return {
            "id": f"{self.name}-{'-'.join(self.templates)}",
            "name": self.common,
            "type": self.templates[0].type,
            "key": self.name,
            "colors": self.colors,
            "require": list(self.templates)
            }


class UserDataGui(DataView):
    targets = []

    def __init__(self, *targets):
        super().__init__(*targets)
        self.targets = list(self.targets)

    def item_model(self, name: str, value: UserDataGuiItem):
        return value

    def data_model(self, instance, value: list[UserDataGuiItem] | None = None):
        return None if (value is None) or (value == []) else value


class GroupUserData(DataView):
    def __init__(self, *targets):
        super().__init__(*(("gui",) + targets))

    def item_model(self, name, value):
        return {name: value}

    def data_model(self, instance, value):
        return dict(value)
