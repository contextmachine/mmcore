import collections
import copy
import json
import typing
from collections import Counter
from typing import Callable, Sequence, Type, Union

import more_itertools
import numpy
import numpy as np

"""
    class traverse(Callable):
        def __call__(self, seq: Sequence | Any) -> Any:
            ...
            
            for typ in self.type_extras
                if isinstance(seq, typ.cls):
                    return typ.resolver(seq)
                else:
                    continue
                
            
"""
TraverseTypeExtra = collections.namedtuple("TraverseTypeExtra", ["cls", "resolver"])


class TypeExtras:
    def __init__(self, *args):
        self._args = set(args)

    def __set_name__(self, owner, name):
        self.name = name
        self.owner = owner
        setattr(owner, "__" + self.name, self._args)

    def __get__(self, instance, owner):
        cls_extras = resolve_class_extras(owner)
        if instance is not None:
            instance_overrides = getattr(instance, "_" + self.name)
            extras = copy.deepcopy(cls_extras)

            extras.update(instance_overrides)
            return extras
        else:
            return cls_extras


def resolve_class_extras(cls):
    extras_set = set()
    for parent in cls.__bases__:
        if hasattr(parent, "type_extras"):
            extras_set.update(parent.__type_extras)
    extras_set.update(cls.__type_extras)
    return extras_set


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
    >>> b = next(traverse()(a))
    >>> b.shape, a.shape
    ((4, 2, 3, 7), (4, 2, 3, 7))
    >>> np.allclose(a,b)
    True
    """
    __type_extras: set[TraverseTypeExtra] = set()
    type_extras = TypeExtras()

    def __init__(self, callback, traverse_dict=True, traverse_seq=True, type_extras=()):
        super().__init__()

        self._traverse_dict = traverse_dict
        self._traverse_seq = traverse_seq
        self._type_extras = type_extras

        if callback is None:
            self.callback = lambda x: x
        else:
            self.callback = callback

    def traverse_dict(self, dct):
        if self._traverse_dict:
            dt = {}
            for k, v in dct.items():
                dt[k] = self(v)
            return dt
        else:
            return self.callback(dct)


    def traverse_seq(self, seq):
        if self._traverse_seq:

            typeset = set([self(l) for l in seq])
            if len(typeset) > 1:
                typ = typing.Union[tuple(typeset)]
            elif len(typeset) == 0:
                typ = typing.Any
            else:
                typ = list(typeset)[0]
            return seq.__class__.__class_getitem__(typ)

        else:
            return self.callback(seq)

    @property
    def extra_classes(self):
        return [typ.cls for typ in self.type_extras]

    def __call__(self, seq):
        if seq.__class__ in self.extra_classes:
            for typ in self.type_extras:
                if isinstance(seq, typ.cls):
                    return typ.resolver(seq)
                else:
                    continue

        elif isinstance(seq, str):
            # Это надо обработать в самом начале
            return self.callback(seq)
        elif isinstance(seq, Sequence):
            return self.traverse_seq(seq)

        elif isinstance(seq, dict):
            return self.traverse_dict(seq)
        else:
            return self.callback(seq)


def itm(x):
    if not type(x) == list:
        return type(x)

def itm2(x):
    if not (isinstance(x,Sequence) and not isinstance(x, str)):
        return type(x)
type_extractor = traverse(lambda x: type(x), traverse_dict=False)
dict_type_extractor = traverse(lambda x: type(x), traverse_dict=True)
dict_type_print = lambda tp: json.dumps(repr(traverse(lambda x: type(x).__name__, traverse_dict=True)(tp)),indent=3)
item_type_extractor = traverse(itm, traverse_dict=False)
item_type_extractor2 = traverse(itm2, traverse_dict=False)
class NumSenseTrav(traverse):
    type_extras = TypeExtras(
        TraverseTypeExtra(int, lambda x: typing.Union[int, float]),
        TraverseTypeExtra(float, lambda x: typing.Union[float, int])
        )


class NoneSenseTrav(traverse):
    type_extras = TypeExtras(
        TraverseTypeExtra(type(None), lambda x: typing.Any),
        TraverseTypeExtra(type(None), lambda x: typing.Any),
        )


class StrSenseTrav(traverse):
    type_extras = TypeExtras(
        TraverseTypeExtra(str, lambda x: str),
        TraverseTypeExtra((list, str), lambda x: list[str]),
        )

class DctSenseTrav(traverse):
    type_extras = TypeExtras(
        TraverseTypeExtra(dict, lambda x: dict_type_extractor(x))
        )

class TypeSensevityTraverse(NumSenseTrav, NoneSenseTrav, StrSenseTrav, traverse):
    """
    It is identical this:
    >>> type_sensevity_traverse = traverse(lambda x: type(x), traverse_dict=True, type_extras=(
    ...     TraverseTypeExtra(int, lambda x: typing.Union[int, float]),
    ...     TraverseTypeExtra(float, lambda x: typing.Union[float, int]),
    ...     TraverseTypeExtra(type(None), lambda x: typing.Any),
    ...                 ))
    """
    type_extras = TypeExtras()

class TravDictpretty(NumSenseTrav, NoneSenseTrav, StrSenseTrav, DctSenseTrav, traverse):
    ...
def walk(target, names):
    if isinstance(names, str):
        return getattr(target, names)
    elif isinstance(names, list) and isinstance(names[-1], str):
        return [walk(target, n) for n in names]
    else:

        return walk(getattr(target, names.pop(0)), names.pop())



class Walk:
    """
    >>> from mmcore.addons import ModuleResolver
with ModuleResolver() as rsl:
    import rhino3dm
import rhino3dm

    >>> frame = rhino3dm.BspPlane(0, 3, 1, 3)
    >>> xaxis = Walk(["XAxis",["X","Y","Z"]])
    >>> yaxis = Walk(["YAxis",["X","Y","Z"]])
    >>> zaxis = Walk(["ZAxis",["X","Y","Z"]])
    >>> def query(plane): return xaxis[plane],yaxis[plane],zaxis[plane]
    >>> query(frame)
    ([0.0, -0.31622776601683794, 0.9486832980505138],
     [1.0, 0.0, -0.0],
     [0.0, 0.9486832980505138, 0.31622776601683794])
    """

    def __init__(self, names):
        super().__init__()
        self.names = names

    def __getitem__(self, target):
        return walk(target, copy.deepcopy(self.names))


class Query(Walk):
    def __init__(self, names):
        super().__init__(names)

    def __getitem__(self, target):
        return super().__getitem__(target)

    def __setitem__(self, target, value):
        ...

    def __call__(self, f):
        self.name = f.__name__
        self._f = f

        return self

    def __get__(self, instance, owner):
        if instance is not None:
            return self._f(instance, self[instance])
        else:
            return self


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

    trv = traverse(lambda x: type(x), traverse_dict=True)

    tps = set(numpy.asarray(type_extractor(seq)).flatten())

    return Union[tuple(tps)] if len(tps) != 0 else tuple(tps)[0]


def ismonotype(seq: Sequence) -> bool:
    """
    @param seq:
    @return: bool

    >>> ints = [2, 3, 4, 9]
    >>> some = [2, 3, 4, "9", {"a": 6}]
    >>> ismonotype(some)
    False
    >>> ismonotype(ints)
    True


    """
    tp = sequence_type(seq)
    if hasattr(tp, "__origin__"):

        return not tp.__origin__ == Union
    else:
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

def trv(tp):
    if hasattr(tp, "__args__"):
        return more_itertools.collapse([trv(arg) for arg in tp.__args__])
    else:
        return tp
def ttrv(d):
    R=list(trv(item_type_extractor(d)))
    if len(R)>1:
        return typing.Union[tuple(R)]
    else:


        return R[0]

def traverse_sequence_types(d):
    return ttrv(d)
