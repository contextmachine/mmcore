#  Copyright (c) 2022. Computational Geometry, Digital Engineering and Optimizing your construction processe"
from abc import ABC, abstractmethod
from typing import Any


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
        super().__get__(instance, owner)
        return getattr(instance, self.name)

    @abstractmethod
    def __set__(self, instance, val): ...


class NoDataDescriptor(Descriptor):
    """
        Basic No Data Descriptor.
        Простейшая реализация дескриптора НЕ данных
    """

    def __set__(self, instance, val):
        raise AttributeError("NoDataDescriptor does not support attributes __set__")


class DataDescriptor(Descriptor):
    """
    Basic Data Descriptor
    Простейшая реализация дескриптора данных
    """

    def __set__(self, instance, val):
        instance.__setattr__(self.name, val)


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

    def __init__(self, targets):
        super().__init__()
        self.targets = targets

    def __set_name__(self, owner, name):
        super().__set_name__(owner, name)

        setattr(owner, "_" + name, self)

    def __generate__(self, instance, owner):
        for name in getattr(instance, self.targets):
            yield self.item_model(name=name, value=owner.__getattribute__(instance, name))

    def __get__(self, instance, owner):
        if instance is None:
            return self
        else:
            return self.data_model(instance, value=list(self.__generate__(instance, owner)))
