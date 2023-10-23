import functools
from uuid import uuid4

import numpy as np
from itertools import count

labels = dict()
components = []


def selector(*names):
    s = set()
    [s.add(components[i]) for i in labels.get(names[0], ())]
    for name in names[1:]:
        s.intersection_update(set(components[i] for i in labels.get(name, [])))

    return s


def label(*names):
    def wrap(cls):
        if not hasattr(cls, '__labels__'):
            cls.__labels__ = (cls.__qualname__,)
        cls.__labels__ = tuple(set(cls.__labels__ + (cls.__qualname__,) + names))
        return cls

    return wrap


__cmp_modules__ = dict()


class ComponentMeta(type):
    @classmethod
    def __prepare__(metacls, name, bases, key=None):
        dct = dict(type.__prepare__(name, bases))
        dct |= dict(__name__=name, __bases__=tuple(
            base.__wrapped_component__ if hasattr(base, '__wrapped_component__') else base for base in bases))
        return dct


class ComponentModule:

    def __init__(self, key):
        __cmp_modules__[key] = self
        self.key = key
        self.nodes = dict()
        self.counters = dict()
        self.links = dict()

    def component(self, key=None, derive=None):
        def wrapper(cls):
            nonlocal key, derive
            if key is None:
                key = cls.__name__.lower()
            if derive:
                if hasattr(derive, '__wrapped_component__'):
                    base = derive.__wrapped_component__
                    key = derive.__wrapped_component__.__component_key__
                else:
                    base = derive
                    key = derive.__component_key__
                cls = ComponentMeta(cls.__name__, (cls, base), {"__component_key__": key})

            else:

                cls = ComponentMeta(cls.__name__, (cls,), {"__component_key__": key})

            self.nodes[key] = []

            self.counters[key] = count()
            cls.global_registry = property(fget=lambda slf: self.nodes[slf.__class__.__component_key__],
                                           fset=lambda slf, v: self.nodes.__setitem__(
                                               slf.__class__.__component_key__,
                                               v))

            def _del(slf):
                self.nodes[key][slf._ixs] = None
                del slf

            cls.__del__ = _del

            @functools.wraps(cls)
            def initwrapper(*args, name=None, **kwargs):
                if name is None:
                    name = uuid4().hex

                obj = cls(*args, **kwargs)

                self.nodes[key].append(obj)
                obj.name = name
                obj._ixs = next(self.counters[key])
                obj.global_index = obj._ixs
                obj.uuid = f'{self.key}_{cls.__component_key__}_{name}'

                return obj

            initwrapper.__wrapped_component__ = cls
            wrapper.__wrapped_component__ = cls
            return initwrapper

        return wrapper


def ravel_index(item, dims):
    """
    >>> ravel_index((3, ), (13, ))
    Out: 3
    >>> ravel_index((1, 0), (13, 3))
    Out: 3
    >>> ravel_index((2, 2), (13, 3))
    Out: 8
"""
    i = 0
    for dim, j in zip(dims, item):
        i *= dim
        i += j
    return i


def unravel_index(index, shape, order="C"):
    """
    >>> unravel_index(3, (13,))
    Out: (3,)
    >>> unravel_index(3, (13, 3))
    Out: (1, 0)
    >>> unravel_index(8, (13, 3))
    Out: (2, 2)
    """
    return np.unravel_index(index, shape, order=order)


class EntityComponent:
    __labels__ = ()

    def __hash__(self):
        return hash(repr(self))

    def __new__(cls, *args, **kwargs):

        obj = super().__new__(cls)
        components.append(obj)
        ix = len(components)
        for lbl in cls.__labels__:

            if lbl not in labels:
                labels[lbl] = []
            labels[lbl].append(ix)
        return obj

    @property
    def component_index(self):
        return components.index(self)

    def __del__(self):
        ix = self.component_index

        for lbl in self.__labels__:
            labels[lbl].remove(ix)
        components[ix] = None
        del self
