#  Copyright (c) 2022. Computational Geometry, Digital Engineering and Optimizing your construction processe"
from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence


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
        ##print(instance, owner)
        return instance.__gethook__(instance.client.get_object(Bucket=self.bucket, Key=f"{self.prefix}{self.name}"))

    def __set__(self, instance, value):
        print(instance, value)
        instance.client.put_object(Bucket=self.bucket, Key=f"{self.prefix}{self.name}{instance.suffix}",
                                   Body=instance.__sethook__(value))


def safe_getattribute(cls, self, item):
    try:
        return cls.__getattribute__(self, item)
    except AttributeError as err:
        return None


class DataView(NoDataDescriptor, dict):
    """
    >>> class DataOO(DataView):
    ...     def item_model(self, name, value):
    ...         return {"id": name, "value": value}
    ...     def data_model(self, instance, value):
    ...         return {"type":instance.__clas__.__name__,"data":value}
    >>> from mmcore.base.descriptors import Descriptor, NoDataDescriptor

    >>> class AA('Matchable'):
    ...     __match_args__="a","b","c"
    ...     compute_data=DataOO("a","b")
    >>> a = AA(1,2.,"tt")
    >>> a.compute_data
    {'type': 'DataOO',
     'data': [{'id': 'a', 'value': 1}, {'id': 'b', 'value': 2.0}]}

    """

    @abstractmethod
    def item_model(self, name: str, value: Any):
        ...

    @abstractmethod
    def data_model(self, instance, value):
        ...

    def __init__(self, *targets):
        super().__init__()
        if targets is not None:
            self.targets = targets
        super(dict, self).__init__()
        for target in self.targets:
            self.setdefault(target, None)

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

    def data_model(self, instance, value):
        if all(list(map(lambda x: x is None, dict(value).values()))):
            return None
        else:
            return dict(value)


class UserDataProperties(UserDataField):
    def item_model(self, name, value):
        return name, value

    def data_model(self, instance, value):
        return super().data_model(instance, value)


class UserData(DataView):
    deps = "properties", "gui"

    def __init__(self, *targets):
        object.__init__(self)
        super().__init__(*(self.deps + targets))

    def item_model(self, name: str, value: Any):

        return name, value

    def data_model(self, instance, value) -> dict:
        d = {}
        for k, v in value:
            if v is not None:
                d[k] = v
        return d



class JsonView(DataView):
    deps = "uuid", "type"

    def __init__(self, *targets):
        object.__init__(self)
        super().__init__(*(self.deps + targets))


class Template(str):
    type: str

    def __repr__(self):
        return f"{self.type}-template'{super().__repr__()}'"

    def __str__(self):
        return super().__str__()


class ChartTemplate(Template):
    type: str = "chart"


class GroupUserData(DataView):
    def __init__(self, *targets):
        super().__init__(*(("gui",) + targets))

    def item_model(self, name, value):
        return {name: value}

    def data_model(self, instance, value):
        return dict(value)


class BackendProxyDescriptor:
    def __init__(self):
        self.name = '__getitem__'

    def __get__(self, instance, owner):

        def __getitem__(item):
            i, j = tuple(item)
            if instance is None:

                return lambda x: getattr(x._backend, f"M{i}{j}")
            else:
                return getattr(instance._backend, f"M{i}{j}")

        return __getitem__



class DumpData(DataView):

    def __set_name__(self, owner, name):
        super().__set_name__(owner, name)
        targets = set()
        for key in dir(owner):
            if not key.startswith("_"):
                targets.add(key)
        self.targets = list(targets)

    def item_model(self, name: str, value: Any):

        if hasattr(value, "todict"):
            v = value.todict()
        elif hasattr(value, "data"):
            v = value.data
        elif hasattr(value, "dumpdata"):
            v = value.dumpdata
        elif isinstance(value, Sequence) and not isinstance(value, str):
            return [self.item_model(None, val) for val in value]
        elif isinstance(value, Mapping):
            return [self.item_model(key, val) for key, val in value]
        else:
            v = value
        return v if name is None else (name, v)

    def data_model(self, instance, value) -> dict:

        return dict(value)
