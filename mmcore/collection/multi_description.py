# Copyright (c) CONTEXTMACHINE
# Andrew Astkhov (sth-v) aa@contextmachine.ru
import itertools
from typing import Any, Callable, Generic, Iterable, Mapping, Type, TypeVar

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

T = TypeVar("T")
Seq = TypeVar("Seq", bound=Iterable)


class CollectionItemGetter(Generic[Seq, T]):
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

    def __init__(self, seq: Generic[Seq, T]):
        super().__init__()
        self._seq = seq
        if isinstance(seq[0], dict | self.__class__):
            self._getter = multi_getitem(self._seq)
        else:
            self._getter = multi_getter(self._seq)

    def __getitem__(self, k) -> Seq:
        return list(self._getter(k))


class CollectionItemGetSetter(CollectionItemGetter[Seq, T]):
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


class MaskedGetSetter(CollectionItemGetSetter):
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

    def __hash__(self):
        self.sha = hashlib.sha256(f"{self['__dict__']}".encode())
        return int(self.sha.hexdigest(), 36)


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


class _MultiGetitem2:
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
                    return dict(instance)[item]
                except:
                    return list(instance)[item]
        return wrap
