import dataclasses

import numpy as np


def object_from_bytes(cls,
                      btc: 'bytearray|bytes',
                      dtype=None,
                      **kwargs):
    return np.frombuffer(btc, dtype=cls._dtype if dtype is None else dtype, **kwargs).reshape(cls._shape)


def object_to_bytes(obj, dtype=None, **kwargs) -> bytes:
    arr = obj._array if dtype is None else np.array(obj._array, dtype=dtype)
    return arr.flatten()[:np.prod(obj._shape)].tobytes(**kwargs)


import functools


@functools.lru_cache()
def make_alias(cls):
    a = str.split(cls.__module__ + '.' + cls.__name__, '.')
    return np.array([(a[0], ".".join(a[1:]), a[-1])], dtype=Alias)


Alias = np.dtype([('lib', np.str_, 16), ('source', np.str_, 16), ('type', np.str_, 16)])
View = np.dtype([
    ('offset', int), ('length', int), ('_dtype', np.str_, 4), ('alias', Alias)
])


def object_to_buffer(obj, buffer: bytearray, dtype=None, **kwargs) -> dict[str, int]:
    if dtype is None:
        dtype = obj._dtype
    btc = object_to_bytes(obj, dtype=dtype, **kwargs)

    buffer.extend(btc)
    return len(buffer), len(btc), np.dtype(dtype).str, make_alias(obj.__class__)


@dataclasses.dataclass
class MmCoreProtocol:
    buffer: bytes
    spec: np.void

    spec_header: bytes = b"SPEC"

    def to_bytes(self):
        b = bytearray()
        b.extend(self.buffer)
        b.extend(self.spec_header)
        b.extend(self.spec.tobytes())
        return b


def geometry_to_proto(geoms) -> MmCoreProtocol:
    b = bytearray()
    return MmCoreProtocol(buffer=b, spec=np.array([object_to_buffer(geom, b) for geom in geoms], dtype=View))
