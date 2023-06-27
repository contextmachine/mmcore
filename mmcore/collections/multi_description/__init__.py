# Copyright (c) CONTEXTMACHINE
# Andrew Astkhov (sth-v) aa@contextmachine.ru
# multi_description.py
import abc
import functools
import itertools
from abc import ABC
from operator import attrgetter, itemgetter, methodcaller
from typing import Any, Callable, Generic, Iterable, Iterator, KeysView, Mapping, Protocol, Sequence, Type, TypeVar

import more_itertools
import numpy as np

from mmcore.collections.traversal import dict_type_extractor, item_type_extractor, sequence_type


def _(): pass


FunctionType = type(_)
MapType = type(map)


# Multi Getter concept.
# Simple functional and objective implementation for a generic collections getter.

# See more in original gist: https://gist.github.com/sth-v/7898cb37b9c56d11ca004936a823e366

# Functional Implementation
# -----------------------------------------------------------------------------------------------------------------------


def multi_getter(z):
    return lambda y: map(lambda x: getattr(x, y), z)


def multi_getitem(sequence): return lambda y: map(lambda x: x.get(y), sequence)


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


class _MultiDescriptor(Mapping[KTo, Seq], ABC):
    ...


from mmcore.collections.traversal import ttrv


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
    sequence_type: Type = property(fget=lambda self: sequence_type(self._seq))
    element_type = d = property(fget=lambda self: ttrv(self._seq))
    format_spec = lambda self: [self.__class__.__name__, self.element_type.__name__, list(self.keys()), id(self)]

    # element_type: Generic[Seq, T] = property(fget=lambda self: sequence_type(self._seq, return_type=True))
    def keys(self) -> KeysView:
        try:

            d = np.array(self._seq, dtype=object).flatten()[0]
            keys = {}
            if isinstance(d, dict):
                return d.keys()
            else:
                for key in dir(d):
                    if not key.startswith("_"):
                        keys.setdefault(key)
                return keys.keys()
        except IndexError:
            return []

    def __len__(self) -> int:
        return self._seq.__len__()

    def __iter__(self) -> Iterator:
        return iter(self[k] for k in self.keys())

    def __init__(self, seq: Generic[Seq, T]):
        super().__init__()

        self._seq = seq

    def __getitem__(self, k) -> Seq:

        try:
            if isinstance(sequence_type(self._seq), dict):

                # _getter = multi_getitem(self._seq)

                _getter = multi_getitem(self._seq)
                return tuple(_getter(k))
            elif isinstance(self._seq[0], str):
                _getter = multi_getter(self._seq)
                return tuple(_getter(k))
            elif isinstance(self._seq[0], Sequence) and not isinstance(self._seq[0], str):
                return [CollectionItemGetter(i).__getitem__(k) for i in self._seq]
            elif isinstance(self._seq[0], dict):
                _getter = multi_getitem(self._seq)
                return list(_getter(k))
            elif isinstance(self._seq[0], CollectionItemGetter):
                return [i.__getitem__(k) for i in self._seq]

            else:
                _getter = multi_getter(self._seq)
                # multi_getter(self._seq)
                return list(_getter(k))
        except IndexError:
            return []

    def __str__(self):
        return f"{self.__class__.__name__}[{self.element_type.__name__}({list(self.keys())})]"

    def __repr__(self):
        return f"<{self.__class__.__name__}[{self.element_type.__name__}({list(self.keys())})] object at {id(self)}>"

    def __format__(self, format_spec=None):
        spec = self.format_spec()
        if format_spec is None:
            return spec
        else:
            newargs = format_spec(self)
            newargs.reverse()
            for val in range(len(newargs)):
                spec.insert(len(spec) - val, newargs[val])
            return spec


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

    def __setitem__(self, key: str, value):
        # #print("v")
        if isinstance(self._seq[0], dict | self.__class__):
            _setter = multi_setitem(self._seq)
        else:
            _setter = multi_setter(self._seq)
        return _setter(key, value)


import hashlib


class MaskedGetSetter(CollectionItemGetSetter, ABC):
    masks = {}

    def set_mask(self, name: str, mask):
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


class MethodDescriptor:
    def __init__(self, name):

        self.name = name

    def __get__(self, instance: MultiDescriptor, owner) -> Callable:
        return self(instance, owner if owner is not None else instance.element_type)

    def __call__(self, instance, owner) -> Callable:
        """
        Может принять одну или несколько секвенций в качестве аргументов
        @param instance:
        @return:
        """

        @functools.wraps(getattr(owner, self.name))
        def wrp(seq: ElementSequence | tuple[ElementSequence] | Any | None = None) -> MapType:
            if isinstance(seq, MultiDescriptor):
                arg = zip(instance._seq, seq._seq)
                return itertools.starmap(getattr(owner, self.name), arg)
            elif isinstance(seq, tuple):
                arg = zip(*((instance._seq) + tuple([s._seq for s in seq])))
                return itertools.starmap(getattr(owner, self.name), arg)

            else:
                arg = instance._seq

                return map(getattr(owner, self.name), arg)

        return wrp


from types import MethodType
import types



class ElementSequence(MultiDescriptor):
    ignored = int, float, str, bytes

    def schema(self):
        if self.element_type == dict:
            return dict_type_extractor(list(more_itertools.collapse(self._seq))[0])
        else:
            return item_type_extractor(list(more_itertools.collapse(self._seq))[0])

    def __list__(self):
        return list(self._seq)

    def __array__(self):
        return np.asarray(list(self._seq))

    def __getitem__(self, item):

        val = CollectionItemGetter.__getitem__(self, item)
        seq_type = sequence_type(val)
        if sequence_type(val) == property:
            return val
        elif sequence_type(val) == MethodType:
            return MethodDescriptor(item).__get__(self, None)
        else:
            return val

    def get_from_index(self, index):
        return self._seq[index]

    def multi_search_from_key_value(self, key, value) -> list[int]:
        i = -1
        ixs = []
        while True:
            try:
                i = list(self[key]).index(value, i + 1)
                ixs.append(i)
            except ValueError as err:
                break
            except Exception as err:
                raise err
        return ixs

    def where_with_rule(self, a, b, rule):
        l = []
        print(a,b,rule)
        for i, item in enumerate(self[a]):
            ans = rule(item, b)
            #print(ans)
            if ans:
                l.append(self.get_from_index(i))
        print(l)
        return l

    def iwhere(self, **rules):
        s = None

        for i, rule in enumerate(rules.items()):
            if i == 0:
                s = set(self.multi_search_from_key_value(*rule))
            else:
                s.intersection_update(set(self.multi_search_from_key_value(*rule)))
        return s

    def where(self, **rules):
        return [self._seq[i] for i in self.iwhere(**rules)]

    def search_from_key_value(self, key, value) -> int:
        """

        @param key:
        @param value:
        @return: Return index value. Item can be assessed by
        """
        try:
            return list(self[key]).index(value)
        except KeyError as err:
            raise KeyError(f"Lost key: {key}\n", err)
        except ValueError as err:
            message = f"Lost key: {key}\nLost value: {value} in sequence: \n\t{self[key]}"
            raise ValueError(message, err)

    # Convert to pandas.DataFrame
    # Presets any pandas.DataFrame conversions
    # All in this method can be call self.to_pandas().to_<target_format>().
    # Use it if you want to use ALL custom export properties .

    def to_pandas(self):
        import pandas as pd
        return pd.DataFrame(self._seq)

    def to_csv(self, **kwargs):
        """

        @param kwargs:
        @return:
        We have predefined some reasonable (in our opinion) export parameters.
        You can change it by passing **kwargs.

        If you want completely custom parameters,
         override the method or just call the original function with:
            ```
            <ElementSequence object>.to_pandas().to_<target_format>(**your_custom_kwargs)
            ```
        """
        return self.to_pandas().to_csv(**kwargs)

    def to_excel(self, **kwargs):
        return self.to_pandas().to_excel(**kwargs)

    def to_sql(self, **kwargs):
        return self.to_pandas().to_sql(**kwargs)

    def to_html(self, classes='table table-stripped', **kwargs):
        return self.to_pandas().to_html(classes=classes, **kwargs)

    def todict(self, **kwargs):
        """
        return not a dict but Array[dict]
        @param kwargs:
        @type kwargs:
        @return:
        @rtype:
        """

        return [s.todict() for s in self._seq]


class NextIndex(Iterator):
    def __init__(self, iterable, target):
        self.iterable = iterable
        self.target = target
        self.i = -1

    def __next__(self):
        try:
            self.i = self.iterable.index(self.target, self.i + 1, -1)
            return self.i
        except ValueError as err:
            raise StopIteration

    def __iter__(self):
        return self


"""
def ff(data, i):
    
    if isinstance(data, Sequence | Mapping | ) and not isinstance(data, str):
        return data[i]
    elif isinstance(data, list)
"""


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


class ES(ElementSequence):
    def __getitem__(self, item):
        k = super().__getitem__(item)

        if not isinstance(k[0], str):
            if isinstance(k[0], Sequence) or isinstance(k[0], dict):
                return ES(k)
            else:
                return k
        else:
            return k


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
                        raise ValueError(f'{err}\n\n\t{instance[k]}, {v}')

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


def fnmap(funcs: Sequence[Callable[..., map]], objects: Sequence[Any]):
    """

    :param funcs:
    :param objects:
    :return: map

    >>> import operator
    >>> fun=fnmap([operator.add,operator.mul], [1,2])
    >>> res = fun(3)
    >>> type(res)
    map
    >>> list(res)
    [4, 6]
    """
    return lambda *args, **kwargs: map(lambda f, o: f(o, *args, **kwargs), funcs, objects)


from mmcore.collections.traversal import ismonotype


class EntityCollection(ElementSequence):

    def ismonotype(self):
        return ismonotype(self._seq)

    @property
    def element_type(self):

        return sequence_type(self._seq).__args__[0]

    def __getitem__(self, item):
        try:
            val = CollectionItemGetter.__getitem__(self, item)
            seq_type = sequence_type(val).__args__[0]

            if seq_type == MethodType:

                return fnmap(val, self._seq)
            else:

                return super().__getitem__(item)
        except AttributeError as err:
            return itertools.repeat()

    def __setitem__(self, item, v):
        super().__setitem__(item, v)


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


# aa
# Explicit passing of an element type to a __init_subclass__
# --------------------------------------------------------


def init_seq(cb):
    def wrp(self, *args, **kwargs):
        if len(self) == 0:
            val = cb(self, *args, **kwargs)
            if not (len(self) == 0):
                self.orig._sequence = ElementSequence(self)
            return val
        else:
            return cb(self, *args, **kwargs)

    return wrp


def stash_seq(cb):
    def wrp(self, *args, **kwargs):
        if not (len(self) == 0):
            val = cb(self, *args, **kwargs)
            if len(self) == 0:
                del self.orig._sequence
            return val
        else:
            return cb(self, *args, **kwargs)

    return wrp


class CallbackList(list):
    """
    >>> class Orig:
    ...
    ...     def __init__(self):                          
    ...         self._names = CallbackList(orig=self) 
    ...            
    ...     @property
    ...     def names(self):
    ...         return self._sequence
    ...    
    ...     def append_name(self, name):
    ...         self._names.append(name)
    >>> oo=Orig()
    >>> oo.append_name({"instance":1, "name":"foo"})
    >>> oo.append_name({"instance":1, "name":"bar"})
    >>> oo.names.where(instance=1,name="foo")
    [{'instance': 1, 'name': 'foo'}]
    >>> oo.names.where(name="foo")
    [{'instance': 1, 'name': 'foo', 'bar': 'baz'},
     {'instance': 2, 'name': 'foo', 'bar': 'other'}]
    >>> oo.names.where(instance=1,bar="baz")
    [{'instance': 1, 'name': 'foo', 'bar': 'baz'},
     {'instance': 1, 'name': 'bar', 'bar': 'baz'}]
    
    """

    def __init__(self, *args, orig=None, **kwargs):
        super().__init__()
        self.orig = orig

    @init_seq
    def append(self, *args, **kwargs):
        list.append(self, *args, **kwargs)

    @init_seq
    def extend(self, *args, **kwargs):
        list.extend(self, *args, **kwargs)

    @stash_seq
    def remove(self, val) -> None:
        list.remove(self, val)

    @stash_seq
    def __delitem__(self, key):
        list.__delitem__(self, key)


class AttrGetter:
    def __init__(self, *query):
        super().__init__()
        self.query = query

    @property
    def methodcaller(self):
        return methodcaller(*self.query)

    @property
    def itemgetter(self):
        return itemgetter(*self.query)

    @property
    def attrgetter(self):
        return attrgetter(*self.query)

    def __call__(self, obj):

        if type(attrgetter(self.query[0])(obj)) == types.MethodType:
            getter = self.methodcaller
        elif issubclass(type(obj), Mapping):
            getter = self.itemgetter
        else:
            getter = self.attrgetter
        try:
            return self.__post_getitem__(obj, getter)
        except AttributeError as err:
            return None

    @abc.abstractmethod
    def __post_getitem__(self, obj, getter):
        return getter(obj)


class MultiAttrGetSetter:
    def __init__(self, *query):
        super().__init__()
        self.query = query

    @property
    def methodcaller(self):
        return methodcaller(*self.query)

    @property
    def itemgetter(self):
        return itemgetter(*self.query)

    @property
    def attrgetter(self):
        return attrgetter(*self.query)

    def __call__(self, objects, seq_type=None):
        if seq_type is None:
            seq_type = sequence_type(objects).__args__[0]
        print(seq_type)
        print(self.query)
        if type(attrgetter(self.query[0])(objects[0])) == types.MethodType:
            getter = self.methodcaller
        elif issubclass(seq_type, Mapping):
            getter = self.itemgetter
        else:
            getter = self.attrgetter
        return self.__post_getitem__(objects, getter)

    @abc.abstractmethod
    def __post_getitem__(self, objects, getter):
        lst = []

        for obj in objects:
            try:
                lst.append(getter(obj))
            except AttributeError as err:
                lst.append(None)
        return lst


class Paginate(EntityCollection):
    """
    - Глобально переработан гетитем Можно запрашивать любое количество ключей
    - При вызове метода для коллекции доп аргументы следует передавать также в гетитем
    """

    def __init__(self, seq):
        super().__init__(seq)

    def __getitem__(self, item):
        return MultiAttrGetSetter(*item)(self._seq)


import strawberry

strawberry.scalar


class GQLAttrGetter(AttrGetter):

    @abc.abstractmethod
    def __post_getitem__(self, obj, getter):
        return dict(zip(self.query, super().__post_getitem__(obj, getter)))


class GQLMultiAttrGetSetter(MultiAttrGetSetter):
    def __post_getitem__(self, objects, getter):

        lst = []
        for obj in objects:
            try:
                val = getter(obj)
                if not isinstance(val, (tuple, list)):
                    val = [val]

                lst.append(dict(zip(self.query, val)))
            except AttributeError as err:
                lst.append(None)
        return lst


class GQLPaginate(Paginate):
    """
    >>> pg = GQLPaginate(objs)
    >>> pg["x"]
    [{'x': 12}, {'x': 13}]
    >>> pg["x", "negx"]
    [{'x': 12, 'negx': -12}, {'x': 13, 'negx': -13}]
    """

    def __getitem__(self, item):
        return GQLMultiAttrGetSetter(*item)(self._seq)
