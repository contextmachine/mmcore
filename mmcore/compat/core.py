import abc
import dataclasses
import functools
import itertools
import sys

import typing

import inspect
import warnings
from abc import ABC
from types import FunctionType, LambdaType, ModuleType



import json

NUMPY_AVAILABLE = True
MMCORE_AVAILABLE = True
from dataclasses import dataclass, asdict, is_dataclass

import numpy as np
from numpy import ndarray




from mmcore.numeric.routines import add_dim



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


class Dict:
    @classmethod
    def cast_pair(cls, kt, vt, k, v):
        # print( kt, vt, k, v)
        return kt.cast(k), vt.cast(v)

    def __class_getitem__(cls, ktvt):
        meta = AutoCastMeta[dict[ktvt[0], ktvt[1]]]
        meta.key_caster = AutoCastMeta[ktvt[0]]

        meta.value_caster = AutoCastMeta[ktvt[1]]

        def cast_p(kv=None):
            if kv is not None:
                return cls.cast_pair(meta.key_caster, meta.value_caster, kv[0], kv[1])

        meta.cast_type = cast_p

        def container_type(x):
            return {k: v for k, v in x}

        meta.container_type = container_type
        meta.is_mapping = True
        meta.container = True
        return meta

class AutoCastMeta(type):
    """
    >>> ListOfStrings=AutoCastMeta[list[str]]
    >>> ListOfStrings
    mmcore.compat.core.TypeCastAlias[<class 'list'>]
    >>> ListOfStrings.cast((3,4,5))
    ['3', '4', '5']

    """
    __autocast_aliases__ = dict()
    cast_type: type | FunctionType | LambdaType = lambda *args: args[0] if args else None
    container: bool = False
    container_type: type = list
    default: typing.Any = None
    is_mapping = False

    def __class_getitem__(cls, item, /, cast_type=None):
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
        elif _origin in [dict]:
            if _args:
                attrs['container'] = True
                attrs['container_type'] = _origin
                attrs['cast_type'] = _origin
                attrs['is_mapping'] = True
            else:
                raise TypeError(f'Container type {_origin} has no arguments: {item} ')

        else:

            attrs['cast_type'] = _origin
        return type(f'TypeCastAlias[{_origin}]', (cls,), attrs)

    def __new__(metacls, name, bases, attrs, **kwargs):
        annotations = attrs.get('__annotations__', {})
        attrs['__autocast_aliases__'] = {}

        for base in reversed(bases):
            attrs['__autocast_aliases__'] |= getattr(base, '__autocast_aliases__', {})
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
            if cls.container and cls.is_mapping:
                return cls.container_type((cls.cast_type(i) for i in data.items()))

            elif hasattr(cls.cast_type, 'from_dict'):
                return cls.cast_type.from_dict(data)


        elif isinstance(data, dataclasses.Field) and (data.default is not dataclasses.MISSING):
            # print(data,data.default_factory )

            return cls.cast_type(data.default)
        elif isinstance(data, dataclasses.Field) and data.default_factory is not dataclasses.MISSING:
            return cls.cast_type(data.default_factory())

        elif isinstance(data, (list, tuple, np.ndarray)):
            if isinstance(data, np.ndarray):
                data = data.tolist()
            return cls.container_type([cls.cast(i) for i in data])

        elif inspect.isfunction(cls.cast_type) or inspect.ismethod(cls.cast_type):

            return cls.cast_type(data) if data is not None else cls.cast_type()

        elif isinstance(data, cls.cast_type):
            return data

        else:
            try:
                return cls.cast_type(data) if data is not None else cls.cast_type()
            except ValueError:
                msg = f'Could not cast {data}, returning {cls.cast_type}()'
                warnings.warn(msg)
                return cls.cast_type()

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


class ComparsionCollectionResult(list):
    @property
    def operands(self):
        return {key: value.operands for key, value in self.items()}

    def __bool__(self):
        return all(self)

    def __eq__(self, other):
        if isinstance(other, ComparsionCollectionResult):

            return all([bool(self[k]) == bool(v) for k, v in other])
        else:
            return False

    def to_dict(self):
        return [value.to_dict() for key, value in self]

    @property
    def operands(self):
        return [value.operands for value in self]

    def different_operands(self):
        return [value.operands for value in self]

    def get_diffs(self):
        for value in self:

            if not value:
                yield value

    def get_diffs_recursieve(self):
        for value in self.get_diffs():
            if isinstance(value, ComparsionResult):
                yield dict(value.get_diffs_recursieve())
            elif isinstance(value, ComparsionCollectionResult):
                yield list(value.get_diffs_recursieve())
            else:
                yield value

    def __repr__(self):

        return f'ComparsionCollectionResult({self.result}, operands={self.operands})'

    def update_other(self, other, side='left', logger=lambda *args: ...):
        _other = []
        _other.extend(itertools.repeat(None, len(self)))
        for j, v in enumerate(other):
            _other[j] = v

        for i, value in enumerate(self.get_diffs()):

            if isinstance(value, (ComparsionResult, ComparsionCollectionResult)):

                logger(other, i)

                next_other = _other[i]
                value.update_other(next_other, side=side, logger=logger)



            else:
                _vl = getattr(value.operands, _lrdicr[side])
                logger(other, i, _vl)

                _other[i] = _vl

        other[:] = _other

    def update_right(self, right, logger=False):
        _logger = logger if isinstance(logger, FunctionType) else self.update_logger(logger)
        _logger(right)
        self.update_other(right, side='right', logger=_logger)

    def update_left(self, left, logger=False):
        _logger = logger if isinstance(logger, FunctionType) else self.update_logger(logger)
        _logger(left)
        self.update_other(left, side='left', logger=_logger)

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
class ComparsionResult(dict):

    def __bool__(self):
        return all(list(self.values()))

    def __eq__(self, other):

        return [bool(self[k]) == bool(v) for k, v in other]

    @property
    def result(self):

        return bool(self)

    def to_dict(self):
        return {key: value.to_dict() for key, value in self.items()}

    @property
    def operands(self):
        return {key: value.operands if value is not None else value for key, value in self.items()}

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
            elif isinstance(value, ComparsionCollectionResult):
                yield key, list(key.get_diffs_recursieve())
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


from typing import Optional, Any, Union


class ChainHandler:
    """
    The default chaining behavior can be implemented inside a base handler
    class.
    """
    _next_handler: 'Optional[AbstractHandler]' = None

    def __init__(self, next_handle=None):
        self._next_handler = next_handle

    def __call__(self, request: Any) -> Any:

        try:
            if self._next_handler:
                return self._next_handler(request)
            return None
        except Exception as err:
            raise RuntimeError(err, f'Error {repr(err)} \n\n\tin {self._next_handler.__class__.__name__} \n\n\tafter'
                                    f' {self.__class__.__name__} '
                                    f'\n\n\twith {request}'
                               )


class ChainSwitch(ChainHandler):
    def __init__(self, next_handle: ChainHandler = None, /, *switch_handles, ):

        super().__init__(next_handle)
        self._switch_handles = switch_handles

    def __call__(self, request: Any) -> Any:

        """
        def __call__(request: Any) :
            if <condition(request)>:
                return self._right_next_handle.__call__(request)
            else:
                return super().__call__(request)
        :param request: Chain of Responsibility Requests
        :type request: Any
        :return: Chain of Responsibility Result
        :rtype: Any
        """

        for h in self._switch_handles:
            r = h(request)
            if r is not None:
                return r

        return super().__call__(request)

    def derive_handler(self, request):
        res = self.switch(request)

        return self._switch_handles[int(res)]

    @abc.abstractmethod
    def switch(self, request: Any) -> int:
        """
        return int from 1+<switch handle i to <switch handlers count>+1 if one of condition success
        :param self:
        :type self:
        :param request:
        :type request:
        :return:
        :rtype:
        """
        return 0


@dataclass
class CompareRequest:
    left: Any
    right: Any

    origin: 'Optional[CompareHandler]' = None
    prev: 'Optional[CompareHandler]' = None


def prepare_request(m):
    @functools.wraps(m)
    def wrapper(self, request: CompareRequest):
        if request.origin is None:
            request.origin = self
        request.prev = self
        #print(self, m)
        res = m(self, request)
        #print(res)

        return res

    return wrapper


def root_provider(cls: ChainHandler):
    def wrapper(next_handle_wrapper=None, root=None):
        if root is None:
            next_handle_wrapper(

            )


class CompareHandler(ChainHandler):

    def __call__(self, request: CompareRequest) -> Union[
        ComparsionResult, ComparsionCollectionResult, ComparsionResultItem, None]:
        return super().__call__(request)


from collections.abc import Collection, Container, Sequence, Iterator, Mapping, Set, Sized
from typing import AnyStr


def starmap_longest(fun, iterable, *iterables):
    return itertools.starmap(fun, itertools.zip_longest(iterable, *iterables))


def _cast_if_need(val, typ):
    return val if isinstance(val, typ) else typ(val)


class CompareDefault(ChainSwitch):
    def switch(self, request: CompareRequest) -> int:

        if not issubclass(request.left.__class__, Container):
            return 1
        elif isinstance(request.left, (str, bytes)):
            return 2

        else:
            return super().switch(request)


class CompareScalar(CompareHandler):
    @prepare_request
    def __call__(self, request: CompareRequest):

        if isinstance(request.left, (int, bool, float, str, bytes)):

            return ComparsionResultItem(request.left == request.right, left=request.left, right=request.right)
        else:
            return super().__call__(request)


class CompareAnyScalar(CompareHandler):
    @prepare_request
    def __call__(self, request: CompareRequest):
        return ComparsionResultItem(request.left == request.right, left=request.left, right=request.right)


class CompareHashable(CompareHandler):
    @prepare_request
    def __call__(self, request: CompareRequest):

        if hasattr(request.left, '__hash__'):
            return ComparsionResultItem(request.left.__hash__() == request.right.__hash__(), left=request.left,
                                        right=request.right, method='__hash__')
        else:
            return super().__call__(request)


class CompareStr(CompareHandler):
    @prepare_request
    def __call__(self, request: CompareRequest):
        if issubclass(request.left.__class__, str):
            return ComparsionResultItem(request.left == _cast_if_need(request.right), left=request.left,
                                        right=request.right)
        else:
            return super().__call__(request)


class CompareBytes(CompareHandler):
    @prepare_request
    def __call__(self, request: CompareRequest):

        if issubclass(request.left.__class__, bytes):
            return ComparsionResultItem(request.left == request.right if isinstance(request.right, bytes) else
                                        request.right.encode(),
                                        left=request.left,
                                        right=request.right

                                        )
        else:
            return super().__call__(request)


class CompareSequence(CompareHandler):
    @prepare_request
    def __call__(self, request: CompareRequest):

        if issubclass(request.left.__class__, (list, tuple, Set)) and not isinstance(request.left, (str, bytes)):
            return ComparsionCollectionResult(self.equal_longest(request))
        else:
            return super().__call__(request)

    def equal_longest(self, request: CompareRequest):
        return starmap_longest(lambda x, y:
                               request.origin(CompareRequest(x, y, origin=None, prev=self)),
                               request.left,
                               request.right
                               )


class CompareMapping(CompareHandler):
    @prepare_request
    def __call__(self, request: CompareRequest):

        if issubclass(request.left.__class__, Mapping):
            return ComparsionResult(self.equal_longest(request))
        else:
            return super().__call__(request)

    def equal_longest(self, request: CompareRequest):
        for k, v in request.left.items():
            r = request.right.get(k, None)
            yield k, request.origin(CompareRequest(request.left, r, origin=None, prev=self))


class CompareNDArray(CompareSequence):
    @prepare_request
    def __call__(self, request: CompareRequest):

        if isinstance(request.left, np.ndarray):
            left = np.atleast_1d(request.left)
            right = np.atleast_1d(request.right) if isinstance(request.right, np.ndarray) else np.array(
                np.atleast_1d(request.right),
                dtype=request.left.dtype
            )

            if left.dtype == object:
                return super().__call__(CompareRequest(left.tolist(), right.tolist(),
                                                       origin=request.origin, prev=request.prev))

            else:

                if request.left.shape == request.right.shape:
                    return ComparsionResultItem(np.allclose(left, right), left=left, right=right)
                else:
                    return ComparsionResultItem(False, left=request.left, right=right)


        else:
            return CompareHandler.__call__(self, request)


class CompareObject(CompareMapping):
    @prepare_request
    def __call__(self, request: CompareRequest):
        if request.origin is None:
            request.origin = self
        if hasattr(request.left, '__dict__'):
            return super().__call__(CompareRequest(left=request.left.__dict__,
                                                   right=request.right.__dict__,
                                                   origin=request.origin,
                                                   prev=request.prev))

        else:
            return CompareHandler.__call__(self, request)


class CompareObjectSwitch(ChainSwitch):

    def switch(self, request: CompareRequest) -> int:
        if isinstance(request.left, (int, bool, float, str, bytes)):
            return 1
        elif hasattr(request.left, '__dict__'):
            return 2
        elif hasattr(request.left, '__hash__'):
            return 3
        elif hasattr(request.left, '__slots__'):
            return 4
        else:
            return super().switch(request)


class CompareSlots(CompareMapping):

    @prepare_request
    def __call__(self, request: CompareRequest):

        if hasattr(request.left, '__slots__'):
            return super().__call__(
                CompareRequest(left=self.prepare_slots(request.left),
                               right=self.prepare_slots(request.right),
                               origin=request.origin,
                               prev=request.prev
                               )
            )

        else:
            return CompareHandler.__call__(self, request)

    def prepare_slots(self, obj):
        return {k: getattr(obj, k, None) for k in obj.__slots__}


scalarComparsionChain = CompareScalar(CompareObject(CompareSlots()))

comparsionHandler = CompareDefault(CompareNDArray(
    CompareSequence(
        CompareMapping(
        )
    )
),
    CompareObjectSwitch(CompareAnyScalar(),
                        CompareScalar(),
                        CompareObject(),
                        CompareHashable(),
                        CompareSlots()
                        ),
    CompareStr()

)













@dataclass
class AutoData(metaclass=AutoCastMeta):

    @classmethod
    def from_dict(cls, data: dict):
        dct = {}
        for k, v in cls.__autocast_aliases__.items():
            dct[k] = v.cast(data.get(k, getattr(cls, '__dataclass_fields__', {}).get(k)))

        return cls(**dct)

    def to_dict(self):

        return asdict(self, dict_factory=getattr(self, 'dict_factory', dict))

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
                #print(k, first, second)
                raise err.__class__("{}, {}, {}, {}".format(k, first, second, err))
        return comparsion_result


from dataclasses import Field, make_dataclass


def make_autodata(name, dct, bases=()):
    """
    Make AutoData instance from a dictionary with name

    :param name:
    :type name:
    :param dct:
    :type dct:
    :return:
    :rtype:

    >>> FooData=make_autodata("FooData",{
    ...   "foo":{
    ...       "foo":int,
    ...       "bar":list[str]
    ...   },
    ...   "baz": list[float] })

    >>> FooData
    mmcore.compat.core.FooData
    >>>FooData.from_dict({ "foo":{"foo":3,"bar":['d','d']},"baz": [1,3.0]})
        FooData(foo=FooDataFoo(foo=3, bar=['d', 'd']), baz=[1.0, 3.0])

    """
    if isinstance(dct, dict):
        return make_dataclass(name, [(k, make_autodata(name + k.capitalize(), v)) for k, v in dct.items()],
                              bases=(*bases, AutoData))
    elif isinstance(dct, list):
        return AutoCastMeta[list[[make_autodata(name, v) for v in dct]]]
    else:
        return AutoCastMeta[dct]
