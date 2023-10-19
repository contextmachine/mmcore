import functools
import operator
import struct

import numpy as np
from more_itertools import chunked

from mmcore.compat.gltf.consts import typeTargetsMap, TYPE_TABLE, componentTypeCodeTable


def finalize_gltf_buffer(data: tuple, typ_size: int):
    if typ_size == 1:
        return data
    else:
        return chunked(data, typ_size)


def gltf_decode_buffer(buffers):
    _prefixes = []
    _buffers = []
    for buffer in buffers:
        pref, buff = buffer.uri.split(",")
        _buffers.append(buff)
        _prefixes.append(pref)
    return _prefixes, _buffers


def todict_minify(self):
    def gen(obj):
        for k in obj.__slots__:
            v = getattr(obj, k)
            if v is not None:
                yield k, v

    return dict(gen(self))


def todict_nested(self, base_cls):
    def gen(obj):
        if isinstance(obj, (list, tuple)):
            if len(obj) > 0:
                return [gen(o) for o in obj]
        elif isinstance(obj, base_cls):
            dct = dict()
            for k in obj.__slots__:
                v = getattr(obj, k)
                val = gen(v)
                if val is not None:
                    dct[k] = val
            return dct
        else:

            return obj

    return gen(self)


def array_flatlen(arr):
    if isinstance(arr, (list, tuple)):
        arr = np.array(arr)
    return int(functools.reduce(operator.mul, arr.shape))


def byte_stride(type: str, componentType: int):
    return TYPE_TABLE[type]['size'] * componentTypeCodeTable[componentType]['size']


def struct_fmt(data, dtype: int):
    return f"{array_flatlen(data)}{componentTypeCodeTable[dtype]['typecode']}"


def packer(buffer, data, dtype=5126, offset=0):
    res = np.array(data, dtype=componentTypeCodeTable[dtype]['numpy']).flatten()
    fmt = struct_fmt(data, dtype)
    struct.pack_into(fmt, buffer, offset, *res)


def pack(data, dtype=5126):
    res = np.array(data, dtype=componentTypeCodeTable[dtype]['numpy']).flatten()

    return res.tobytes()


def addBufferView(arr, buffer: bytearray, gltf_type="VEC3", dtype=5126, offset=0, name=None):
    flatview = np.array(arr, dtype=componentTypeCodeTable[dtype]['numpy']).flatten()

    struct.pack_into(struct_fmt(flatview, dtype), buffer, offset, flatview)
    return {
        "buffer": 0,
        "byteLength": len(flatview) * componentTypeCodeTable[dtype]['size'],
        "byteOffset": offset,
        "byteStride": byte_stride(gltf_type, dtype),
        "name": name,
        "target": typeTargetsMap[gltf_type]
    }


def appendBufferView(arr, buffer: bytearray, gltf_type="VEC3", dtype=5126, name=None, use_stride=False):
    flatview = np.array(arr, dtype=componentTypeCodeTable[dtype]['numpy']).flatten()

    res = {
        "buffer": 0,
        "byteLength": len(flatview) * componentTypeCodeTable[dtype]['size'],
        "byteOffset": len(buffer),

        "name": name,
        "target": typeTargetsMap[gltf_type][0]
    }
    if all([use_stride, (gltf_type != 'SCALAR')]):
        res["byteStride"] = byte_stride(gltf_type, dtype)
    buffer.extend(flatview.tobytes())
    return res
