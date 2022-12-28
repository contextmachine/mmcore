# Copyright (c) CONTEXTMACHINE
# Andrew Astkhov (sth-v) aa@contextmachine.ru
import collections
import itertools
from abc import ABC
from collections import Counter
from typing import Any, Callable, Generic, Iterable, Iterator, Mapping, Protocol, Sequence, Type, TypeVar, \
    Union

import numpy as np


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

    def __call__(self, seq: Sequence | Any) -> collections.abc.Generator[Sequence | Any]:

        if not isinstance(seq, str) and isinstance(seq, Sequence):
            for l in seq: yield next(self(l))
        else:
            yield self.callback(seq)


# Проходит по всем элементам секвенции, возвращая секвенцию типов элементов.
type_extractor = traverse(lambda x: x.__class__)


def sequence_type(seq: Sequence) -> Type:
    """
    Extract types for sequence.

    >>> ints = [2, 3, 4, 9]
    >>> sequence_type(ints)
    <class 'int'>
    >>> some = [2, 3, 4, "9", {"a": 6}]
    >>> sequence_type(some)
    typing.Union[str, dict, int]

    @param seq: input sequence
    @return: single or Union type
    """

    assert isinstance(seq, Sequence)
    tps = set(type_extractor(seq))

    return Union[tuple(tps)] if len(tps) != 0 else tuple(tps)[0]


def ismonotype(seq: Sequence) -> bool:
    tp = sequence_type(seq),
    try:
        return not (tp.__origin__ == Union)

    except Exception as err:
        print(err)
        return True


def sequence_type_counter(seq: Sequence) -> Counter[Type]:
    """
    ```type_extractor``` based function.

    @param seq: Any sequence
    @return: Count of unique types

    >>> some = [2, 3, 4, "9", {"a": 6}]
    >>> sequence_type_counter(some)
    Counter({<class 'int'>: 3, <class 'str'>: 1, <class 'dict'>: 1})
    """
    assert isinstance(seq, Sequence)
    return Counter(type_extractor(seq))


class _MultiDescriptor(Mapping[KTo, Seq], ABC):
    ...


class CollectionItemGetter(_MultiDescriptor[str, Sequence]):
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
    element_type: Type = property(fget=lambda self: sequence_type(self._seq).pop())

    # element_type: Generic[Seq, T] = property(fget=lambda self: sequence_type(self._seq, return_type=True))

    def __len__(self) -> int:
        return self._seq.__len__()

    def __iter__(self) -> Iterator:
        return iter(self[k] for k in self.keys())

    def __init__(self, seq: Generic[Seq, T]):
        super().__init__()
        assert len(sequence_type(seq)) == 1
        self._seq = seq
        if isinstance(seq[0], Mapping):

            self._getter = multi_getitem(self._seq)
        else:
            self._getter = multi_getter(self._seq)

    def __getitem__(self, k) -> Seq:
        return list(self._getter(k))

    def __repr__(self):
        return self.__class__.__name__ + f"[{self._seq.__class__.__name__}, {sequence_type(self._seq).pop()}]"


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

        super().__init__(seq)
        if isinstance(seq[0], dict | self.__class__):
            self._setter = multi_setitem(self._seq)
        else:
            self._setter = multi_setter(self._seq)

    def __setitem__(self, key: str, value):
        # print("v")
        self._setter(key, value)


from mmcore.collection.masks import Mask
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
        return super().__len__()

    def __hash__(self):
        self.sha = hashlib.sha256(f"{self['__dict__']}".encode())
        return int(self.sha.hexdigest(), 36)


class ElementSequence(MultiDescriptor):
    ignored = int, float, str, Callable, FunctionType

    def __getitem__(self, item):

        r = super().__getitem__(item)
        typechecker = traverse(callback=lambda x: type(x) not in self.ignored)
        if all(typechecker(r)):
            return ElementSequence(r)
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


class SeqProto(Protocol):

    @property
    def _getter(self):
        return multi_getitem(self._seq) if list(sequence_type(self._seq))[0] == dict else multi_getter(self._seq)

    def __getitem__(self, i):
        return list(self._getter(i))


class E(SeqProto):
    def __init__(self, seq):
        super().__init__()
        self._seq = seq


c = E([{"a": 1, "b": 2}, {"a": 5, "b": 3}, {"a": 9, "b": 12}])
#
# Explicit passing of an element type to a __init_subclass__
# --------------------------------------------------------
