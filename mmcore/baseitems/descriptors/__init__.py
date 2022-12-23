#  Copyright (c) 2022. Computational Geometry, Digital Engineering and Optimizing your construction processe"
from abc import ABC, abstractmethod


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
        super().__set_name__(wner, name)
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
