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


        elif isinstance(data, cls.cast_type):
            return data
        else:

            return cls.cast_type(data) if data is not None else cls.cast_type()


@dataclass
class AutoData(metaclass=AutoCastMeta):

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**{k: v.cast(data.get(k, None)) for k, v in cls.__autocast_aliases__.items()})

    def to_dict(self):
        return asdict(self)
