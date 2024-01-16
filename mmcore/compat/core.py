import sys

import typing

import inspect
from types import FunctionType, LambdaType

import numpy as np
import json

from dataclasses import dataclass, asdict, is_dataclass
from mmcore.numeric.routines import add_dim

from more_itertools import flatten


class NdArrayGeneric:
    __shape__ = ...
    __dtype__ = object

    def __class_getitem__(cls, item):
        dct = dict(__shape__=cls.__shape__, __dtype__=cls.__dtype__, __attrs__=(cls.__dtype__,))
        _ks = list(dct.keys())
        for i, val in enumerate(item):
            dct[_ks[i]] = val

        meta = AutoCastMeta[list[dct['__dtype__']]]
        meta.cast_type = dct['__dtype__']
        meta.shape = dct['__shape__']

        def container_type(x):
            shp = meta.shape

            if shp == ...:
                # print(x)
                return np.array(x, meta.cast_type)

            if isinstance(shp, int):
                shp = (shp,)

            if ... in shp:
                shp = shp[shp.index(...):]

            return add_dim(np.array(tuple(x), meta.cast_type), shp)

        meta.container_type = container_type
        meta.container = True
        return meta


class AutoCastMeta(type):
    __autocast_aliases__ = dict()
    cast_type: type | FunctionType | LambdaType = lambda *args: args[0] if args else None
    container: bool = False
    container_type: type = list

    def __class_getitem__(cls, item):
        if isinstance(item, type):
            if issubclass(item, AutoCastMeta):
                return item


        attrs = dict(cast_type=None, container=False, container_type=list)

        _origin = getattr(item, '__origin__', item)
        _args = getattr(item, '__args__', ())
        if _origin in [list, tuple, np.ndarray]:

            if _args:
                attrs['container'] = True
                attrs['container_type'] = _origin
                attrs['cast_type'] = _args[0]
            else:
                raise TypeError(f'Container type {_origin} has no arguments: {item} ')

        else:

            attrs['cast_type'] = _origin
        return type(f'TypeCastAlias[{_origin}]', (cls,), attrs)

    def __new__(metacls, name, bases, attrs, **kwargs):
        annotations = attrs.get('__annotations__', {})
        attrs['__autocast_aliases__'] = {}
        for k, v in annotations.items():
            if not k.startswith('__'):
                attrs['__autocast_aliases__'][k] = AutoCastMeta[v]

        cls = super().__new__(metacls, name, bases, attrs, **kwargs)
        # metacls.__autocast_aliases__[cls] = cls.__autocast_aliases__
        cls.__defined_attributes__ = list(attrs.keys())

        return cls

    @classmethod
    def process_collection(cls, data):
        if isinstance(data, dict):

            return cls.cast(data)
        elif isinstance(data, (list, tuple, np.ndarray)):
            if cls.container:

                return cls.container_type(cls.process_collection(item) for item in data)
            else:
                raise TypeError(f"Unsupported data type. This type is not a container! \n{data} ")
        else:
            raise TypeError(
                    f"Unsupported data type {type(data)}. Input data can be dict or container (if self.container_type "
                    f"is True) \n{data}"
                    )

    @classmethod
    def cast(cls, data):

        if isinstance(data, dict):
            if hasattr(cls.cast_type, 'from_dict'):
                return cls.cast_type.from_dict(data)

        elif isinstance(data, (list, tuple, np.ndarray)):
            if isinstance(data, np.ndarray):
                data = data.tolist()
            return cls.container_type([cls.cast(i) for i in data])

        elif inspect.isfunction(cls.cast_type) or inspect.ismethod(cls.cast_type):

            return cls.cast_type(data) if data is not None else cls.cast_type()
        elif isinstance(data, cls.cast_type):
            return data
        else:

            return cls.cast_type(data) if data is not None else cls.cast_type()


class ComparsionResultItem:
    class Operands:
        left: typing.Any
        right: typing.Any

        def __init__(self, left: typing.Any, right: typing.Any):
            self.left = left
            self.right = right

        def __iter__(self):
            return iter((self.left, self.right))

        def __repr__(self):
            return (f'(left={self.left}, right={self.right})')

        def to_dict(self):
            return dict(right=self.right, left=self.left)

    def __init__(self, result: bool, left, right, method='__eq__'):
        self.operands = self.Operands(left, right)
        self.result = result

    def __bool__(self):
        return bool(self.result)

    def __iter__(self):
        return iter((self.result, self.operands))

    def to_dict(self):
        return dict(result=self.result, operands=self.operands.to_dict())

    def __repr__(self):
        return f'({self.result}, operands={self.operands})'


from typing import Union

_lrdicr = dict(left='right', right='left')


class ComparsionResult(dict):

    def __bool__(self):
        return all(list(self.values()))

    def __eq__(self, other):
        return [bool(self[k]) == bool(v) for k, v in other.items()]

    def to_dict(self):
        return {key: value.to_dict() for key, value in self.items()}

    @property
    def operands(self):
        return {key: value.operands for key, value in self.items()}

    def different_operands(self):
        return {key: value.operands for key, value in self.items()}

    @property
    def result(self):

        return bool(self)

    def __repr__(self):

        return f'ComparsionResult({self.result}, operands={self.operands})'

    def get_diffs(self):
        for key, value in self.items():

            if not value:

                yield key, value

    def get_diffs_recursieve(self):
        for key, value in self.get_diffs():
            if isinstance(value, ComparsionResult):
                yield key, dict(value.get_diffs_recursieve())
            else:
                yield key, value

    def update_logger(self, use=False, file=sys.stdout):

        def logger(item, key=None, value=None):
            if use:
                if value is not None:
                    print(f'.{key}: {getattr(item, key, None)} -> {value}\n\t', end='', file=file)
                elif key is not None:
                    print(f'.{key}', end='', file=file)
                else:

                    print(f'\n\t', end='', file=file)

        return logger

    def update_other(self, other, side='left', logger=lambda *args: ...):

        for key, value in (self.get_diffs()):

            if isinstance(value, ComparsionResult):

                logger(other, key)

                next_other = getattr(other, key)
                value.update_other(next_other, side=side, logger=logger)
            else:
                _vl = getattr(value.operands, _lrdicr[side])
                logger(other, key, _vl)

                setattr(other, key, _vl)

    def update_right(self, right, logger=False):
        _logger = logger if isinstance(logger, FunctionType) else self.update_logger(logger)
        _logger(right)
        self.update_other(right, side='right', logger=_logger)

    def update_left(self, left, logger=False):
        _logger = logger if isinstance(logger, FunctionType) else self.update_logger(logger)
        _logger(left)
        self.update_other(left, side='left', logger=_logger)


@dataclass
class AutoData(metaclass=AutoCastMeta):

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**{k: v.cast(data.get(k, None)) for k, v in cls.__autocast_aliases__.items()})

    def to_dict(self):
        return asdict(self)

    def compare(self, other):

        comparsion_result = ComparsionResult()
        for k in self.__dict__.keys():
            first = getattr(self, k, None)
            second = getattr(other, k, None)
            try:
                if hasattr(first, 'compare'):
                    comparsion_result[k] = first.compare(second)
                elif isinstance(first, np.ndarray):

                    comparsion_result[k] = ComparsionResultItem(np.allclose(first, second), first, second)
                else:

                    comparsion_result[k] = ComparsionResultItem(first == second, first, second)
            except Exception as err:
                print(k, first, second)
                raise err.__class__("{}, {}, {}, {}".format(k, first, second, err))
        return comparsion_result
