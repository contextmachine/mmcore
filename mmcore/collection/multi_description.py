# Copyright (c) CONTEXTMACHINE
# Andrew Astkhov (sth-v) aa@contextmachine.ru
import collections
import itertools
from abc import ABC
from typing import Any, Callable, Generic, Iterable, Iterator, Mapping, Sequence, Type, TypeVar

import numpy as np

from ..baseitems import IdentifiableMatchable, Matchable


def _(): pass


FunctionType = type(_)
MapType = type(map)


# Multi Getter concept.
# Simple functional and objective implementation for a generic collections getter.

# See more in original gist: https://gist.github.com/sth-v/7898cb37b9c56d11ca004936a823e366

# Functional Implementation
# -----------------------------------------------------------------------------------------------------------------------

def multi_getter(z): return lambda y: map(lambda x: getattr(x, y), z)


def multi_getitem(sequence): return lambda y: map(lambda x: x[y], sequence)


def get_with_applicate(function) -> Callable[[Iterable, ...], Callable[[str], MapType]]:
    """
    Извлечь по ключу, применить функцию к каждому элементу последовательности перед и вернуть значение.
    Пример:
    >>> from dataclasses import dataclass

    >>> @dataclass
    ... class ExampleNamespace:
    ...     foo: str
    ...     some: dict

    >>> ex1 = ExampleNamespace(foo="bar", some={"message":"hello"})
    >>> ex2 = ExampleNamespace(foo="nothing", some={"message":"github"})
    >>> exs = ex1, ex2

    >>> def make_upper(x):
    ...     if isinstance(x, str): return x.upper()
    ...     elif isinstance(x, dict): return dict([(k.upper(), make_upper(v)) for k, v in x.items()])
    ...     else: return x

    >>> getter=get_with_applicate(make_upper)(exs)
    >>> list(getter("foo"))
    ['BAR', 'NOTHING']
    >>> list(getter("some")
    [{'MESSAGE': 'HELLO'}, {'MESSAGE': 'GITHUB'}]

    :param function:
    :return: multi_getitem/multi_getattr

    """

    def wrp(sequence):
        curried = multi_getitem(sequence) if isinstance(sequence[0], dict) else multi_getter(sequence)
        return lambda key: map(function, curried(key))

    return wrp


# Уместен ли здесь сеттер -- спорное утверждение. Не факт что этот метод будет пользоваться популярностью.
# Тем не менее мне хотелось бы предоставить возможность и инструмент
# для простого назначения атрибутивной строки всем элементам сразу.
# Это чем то похоже на работу с простой таблицей SQL или Excel

def multi_setter(y) -> Callable[[str, Any], None]:
    def wrap(k: str, v: Any) -> None:
        list(itertools.starmap(lambda xz, zz: setattr(xz, k, zz), zip(y, v)))

    return wrap


def multi_setitem(y) -> Callable[[str, Any], None]:
    def wrap(k: str, v: Any) -> None:
        list(itertools.starmap(lambda xz, zz: xz.__setitem__(k, zz), zip(y, v)))

    return wrap


# Class Implementation
# -----------------------------------------------------------------------------------------------------------------------
KTo = TypeVar("KTo", covariant=True)
VTco = TypeVar("VTco", contravariant=True)
T = TypeVar("T")
Seq = TypeVar("Seq", bound=Sequence)


class traverse(Callable):
    """
    Полностью проходит секвенцию любой степени вложенности.
    В момент обладания каждым из объектов может применить `placeholder`.
    По умолчанию не делает ничего (вернет генератор той-же самой секвенции, что и получил).
    Генератор всегда предсказуем. самый простой способ распаковки -- один раз передать его в `next`.
    -----------------------------------------------------------------------------------------------------
    Example:
    >>> import numpy as np
    >>> a = np.random.random((4,2,3,7))
    >>> b = next(traverse(a))
    >>> b.shape, a.shape
    ((4, 2, 3, 7), (4, 2, 3, 7))
    >>> np.allclose(a,b)
    True
    """
    __slots__ = ("callback",)

    def __init__(self, callback: Callable):
        if callback is None:
            self.callback = lambda x: x
        else:
            self.callback = callback

    def __call__(self, seq: Sequence | Any) -> collections.Generator[Sequence | Any]:

        if not isinstance(seq, str) and isinstance(seq, Sequence):
            for l in seq: yield next(self(l))
        else:
            yield self.callback(seq)


class _MultiDescriptor(Mapping[KTo, Seq], ABC):
    ...


class CollectionItemGetter(_MultiDescriptor[[Sequence, ...], str, Any]):
    """
    # Multi Getter
    Simple functional and objectiv implementation for a generic collection getter.
    Example using python doctest:

    >>> from dataclasses import dataclass

    >>> @dataclass
    ... class ExampleNamespace:
    ...     foo: str
    ...     some: dict

    >>> ex1 = ExampleNamespace(foo="bar", some={"message":"hello"})
    >>> ex2 = ExampleNamespace(foo="nothing", some={"message":"github"})
    >>> exs = ex1, ex2
    >>> getter = multi_getter(exs)
    >>> list(getter("foo"))
    ['bar', 'nothing']
    >>> mg = CollectionItemGetter(exs)
    >>> mg["foo"]
    ['bar', 'nothing']
    >>> ex1.foo = "something else"
    >>> mg["foo"]
    ['something else', 'nothing']
    """

    def __len__(self) -> int:
        pass

    def __iter__(self) -> Iterator[_T_co]:
        pass

    def __init__(self, seq: Generic[Seq, T]):
        super().__init__()
        self._seq = seq
        if isinstance(seq[0], dict | self.__class__):
            self._getter = multi_getitem(self._seq)
        else:
            self._getter = multi_getter(self._seq)

    def __getitem__(self, k) -> Seq:
        return list(self._getter(k))


class CollectionItemGetSetter(CollectionItemGetter):
    """
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class ExampleNamespace:
    ...     foo: str
    ...     some: dict

    >>> ex1 = ExampleNamespace(foo="bar", some={"message":"hello"})
    >>> ex2 = ExampleNamespace(foo="nothing", some={"message":"github"})
    >>> exs = ex1, ex2
    >>> mg = CollectionItemGetSetter(exs)
    >>> mg["foo"]
    ['bar', 'nothing']
    >>> mg["foo"] = 'something else', 'nothing'
    >>> ex1.foo
    'something else'
    """

    def __init__(self, seq: Generic[Seq, T]):
        self._inst: Type[T] = type(seq[0])

        super().__init__(seq)
        if isinstance(seq[0], dict | self.__class__):
            self._setter = multi_setitem(self._seq)
        else:
            self._setter = multi_setter(self._seq)

    def __setitem__(self, key: str, value):
        # print("v")
        self._setter(key, value)


from ..collection.masks import Mask
import hashlib


class MultiDescriptorMask(Mask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def wrapper(mask, instance, owner, constrains,
                **kwargs):  # Strong, very strong, functional programming ... 💪🏼💪🏼💪🏼
        return lambda key: list(filter(constrains(instance[key], **kwargs), instance))  # 💄💋 By, baby


class MaskedGetSetter(CollectionItemGetSetter, ABC):
    masks = {}

    def set_mask(self, name: str, mask: Mask):
        self.__dict__[name] = mask
        mask.__set_name__(self, name)
        self.masks[name] = mask

    def get_mask(self, name):
        return self.masks[name]


class MultiDescriptor(CollectionItemGetSetter):
    """
    Common class
    """

    def __len__(self):
        return len(self._seq)

    def __hash__(self):
        self.sha = hashlib.sha256(f"{self['__dict__']}".encode())
        return int(self.sha.hexdigest(), 36)


class SequenceBinder(MultiDescriptor):
    ignored = int, float, str, Callable, FunctionType

    def __getitem__(self, item):
        r = super().__getitem__(item)

        placeholder = lambda x: type(x) not in self.ignored
        typechecker = traverse(callback=placeholder)
        if all(next(typechecker(r))):
            return SequenceBinder(r)
        else:
            return r


class _MultiGetitem:

    def __get__(self, instance, owner):
        self.instance = instance
        self.owner = owner
        return self

    def __call__(self, item):

        print(item)
        if isinstance(item, tuple | list | set | slice):
            return [self(i) for i in item]
        else:
            return dict(self.instance)[item]


class MultiGetitem2:
    def __get__(self, instance, owner):
        self.instance = instance
        self.owner = owner

        def wrap(item):
            if isinstance(item, Mapping | dict):
                for k, v in item.items():
                    try:
                        return self.__get__(instance[k], owner)(v)
                    except ValueError as err:
                        print(f'{err}\n\n\t{instance[k]}, {v}')

            elif isinstance(item, tuple | list | set | slice):
                return [self.__get__(instance, owner)(i) for i in item]
            else:
                # Я осознаю что это страшный бул-шит

                try:
                    return dict.__getitem__(instance, item)

                except:
                    return list.__getitem__(instance, item)

        return wrap


class MultiSetitem2:
    def __get__(self, instance, o):
        self.instance = instance

        def wrap(item, val):
            if isinstance(item, Mapping | dict):
                for k, v in item.items():
                    try:
                        return wrap(v, val)
                    except ValueError as err:
                        print(f'{err}\n\n\t{instance[k]}, {v}')

            elif isinstance(item, tuple | list | set | slice):
                return [wrap(i, val) for i in item]
            else:

                try:
                    print(item, val)
                    return dict.__setitem__(instance, item, val)

                except:
                    print(item, val)
                    return list.__setitem__(instance, item, val)

        return wrap


class UserData:

    def __get__(self, instance, owner):
        dd = []
        # print(instance, owner)
        for k in self.userdata_names:
            dt = self.udd[k]
            # print(dt)

            dd.append({
                "name": dt.doc,
                "value": dt.value(instance),
                "id": dt.name
            })
        return dd


    class UserDataProperty(Matchable):
        """
        User Data Property
        """
        __include__ = "id", "name", "value"

        def __init__(self, f):
            super().__init__()
            self.f = f
            self.name = f.__name__
            self.id = self.name

        def __get__(self, instance, owner):
            return self.f(instance)

        @property
        def value(self):
            return lambda instance: self.f(instance)


    def __init__(self):
        super().__init__()

        self.userdata_names = []
        self.udd = {}

    def property(self, common_name="UserData Property"):
        def werp(ck):
            inst = self.UserDataProperty(ck)
            inst.common_name = common_name
            self.userdata_names.append(inst.name)
            self.udd[inst.name] = inst

            def wrp(slf):
                return inst.f(slf)

            wrp.__name__ = wrp.name = inst.name

            return wrp

        return werp


class ObjectWithUserData(IdentifiableMatchable):
    __match_args__ = ("bar",)

    userdata = UserData()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @userdata.property("Foo")
    def foo(self):
        return self.bar


class UserDataExample(IdentifiableMatchable):
    """
    Data Views Management.
    ----------------------------------------------------------------------------------------------

    ☎️ По итогу обычный класс должен выглядеть как то так: Основная логика + Менеджмент разных представлений для разных целей
    (Это в том числе импорт/экспорт в разные комплексы но не только ).
    Разница представлений `панель -> 3d | развертка` не сильно отличается от `панель -> three.js | tekla | rhino | ...`
    Представления в виде декораторов в свою очередь удобно менеджерить.

    Example:

    >>> class MultiDataExample(IdentifiableMatchable):
    ...     __match_args__ = ...
    ...     userdata = UserData()
    ...     websocket = WsData()
    ...     production = ProdData()
    ...
    ...
    ...     @websocket.property("api/route/myobj")
    ...     @userdata.property("X")
    ...     def x(self): return self.x
    ...
    ...
    ...     @production.property("test")
    ...     @websocket.property("api/route/myobj")
    ...     @userdata.property("Y")
    ...     def y(self): return self.y
    ...
    ...
    ...     @websocket.property("api/route/myobj")
    ...     @userdata.property("Z")
    ...     def z(self): return self.z
    ...
    ...
    ...     @production.property("tag")
    ...     @userdata.property("UUID")
    ...     def uuid(self): return super(IdentifiableMatchable, self).uuid.__str__()
    ...
    """
    __match_args__ = "x", "y", "z"
    userdata = UserData()

    @userdata.property("X")
    def x(self): return self.x

    @userdata.property("Y")
    def y(self): return self.y

    @userdata.property("Z")
    def z(self): return self.z

    @userdata.property("UUID")
    def uuid(self): return super(IdentifiableMatchable, self).uuid.__str__()
