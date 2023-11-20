import struct
from enum import Enum

import more_itertools
import numpy as np
from more_itertools import chunked


class BufferType(tuple, Enum):
    double = ('d', 8, float)
    float = ('f', 4, np.float32)
    int = ('i', 4, int)
    ushort = ('H', 2, np.uint8)
    char = ('c', 1, np.char)


DTYPES = dict(double=('d', 8, float), float=('f', 4, np.float32), int=('i', 4, int), ushort=('H', 2, np.uint8),
              char=('c', 1, np.char))

from itertools import count
import base64


class Buffer:
    def __init__(self):

        self.views = []
        self.counter = count()
        self.shifts = [0]

    def add_view(self, view):
        self.views.append(view)
        ls = next(self.counter)
        _ = self.shifts[ls]
        self.shifts.append(_ + view.size)
        return ls, _

    def tob64(self):
        return base64.b64encode(self.tobytes()).decode()

    def toviews(self, index=0):
        sz = 0
        for i, view in enumerate(self.views):
            yield {
                'buffer': index,
                'bufferView': i,
                'byteShift': sz,
                'size': view.size,
                'byteStride': view.stride
            }
            sz = sz + view.size

    def tobytes(self):
        b = bytearray()
        for i in self.views:
            b.extend(i._buffer)
        return b

    def get_shift(self, i):
        r = 0
        if i == 0:
            return r
        else:

            for view in self.views[:i]:
                r += view.size
            return r


class BufferView:
    shift: int
    length: int
    dtype: str

    def __init__(self, shift, length, dtype):
        self.shift = shift
        self.dtype = dtype, DTYPES[dtype]
        self.length = length
        self.type_length = self.length * self.shift
        self.size = self.type_length * self.dtype[1][1]
        self._buffer = bytearray(self.size)
        self.stride = self.shift * self.dtype[1][1]
        self.chunked = lambda x: list(chunked(x, self.shift)) if self.shift > 1 else lambda x: x

    def extend(self, count):
        self._buffer.extend(bytearray(count * self.stride))
        self.length += count
        new = count * self.shift
        self.type_length += new
        self.size += new * self.dtype[1][1]

    def get_bytes(self, i):
        _ = self.stride * i
        return self._buffer[_:_ + self.stride]

    def set_bytes(self, i, bts):
        _ = self.stride * i
        self._buffer[_:_ + self.stride] = bts

    def set_many_bytes(self, i, j, bts):
        _i, _j = self.stride * i, self.stride * j
        self._buffer[_i:_j] = bts

    def get_many_bytes(self, i, j):
        _i, _j = self.stride * i, self.stride * j

        return self._buffer[_i:_j]

    def get(self, i):
        return struct.unpack(f'{self.shift}{self.dtype[1][0]}', self.get_bytes(i))

    def get_many(self, i, j):
        if self.shift > 1:
            return self.chunked(struct.unpack(f'{self.slice_size(i, j)}{self.dtype[1][0]}', self.get_many_bytes(i, j)))
        else:
            return struct.unpack(f'{self.slice_size(i, j)}{self.dtype[1][0]}', self.get_many_bytes(i, j))

    def slice_size(self, i, j):
        return (j - i) * self.shift

    def slice_full_size(self, i, j):
        return (j - i) * self.stride

    def set(self, i, dat):
        if self.shift > 1:
            self.set_bytes(i, struct.pack(f'{self.shift}{self.dtype[1][0]}', *dat))
        else:
            self.set_bytes(i, struct.pack(f'{self.dtype[1][0]}', dat))

    def set_many(self, i, j, dat):
        if self.shift > 1:
            self.set_many_bytes(i, j,
                                struct.pack(f'{self.slice_size(i, j)}{self.dtype[1][0]}', *more_itertools.flatten(dat)))
        else:
            self.set_many_bytes(i, j,
                                struct.pack(f'{self.slice_size(i, j)}{self.dtype[1][0]}', *dat))

    def decode_all(self):
        dat = struct.unpack(f'{self.type_length}{self.dtype[1][0]}', self._buffer[:self.stride * self.length])
        if self.shift > 1:
            return self.chunked(dat)
        else:
            return dat

    def __array__(self):
        return np.frombuffer(self._buffer, dtype=self.dtype).reshape((self.length, self.shift))

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.get(item)
        else:
            return self.get_many(*item)

    def __setitem__(self, item, value):
        if isinstance(item, int):
            self.set(item, value)
        else:
            self.set_many(*item, value)

    def __len__(self):
        return self.length

    def __iter__(self):
        return iter(self.decode_all())

    def __repr__(self):
        return f'BufferView(length={self.length}, shift={self.shift}, size={self.size}, dtype={str(self.dtype)})'
