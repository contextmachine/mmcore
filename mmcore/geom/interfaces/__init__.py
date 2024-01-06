# noinspection PyTypeChecker
"""
Single case:
>>> Ray((0,0,0),(1,2,3))
Ray(origin=(0, 0, 0), normal=(1, 2, 3))

Collection case:
>>> import numpy as np
>>> from mmcore.geom.vec import unit
>>> from mmcore.geom.interfaces import Ray
>>> origins = np.array([[0.88014699, 0.6794143 , 0.96630839],
...    [0.50213094, 0.17339094, 0.60248626],
...    [0.62116803, 0.47791134, 0.42499213]])
>>> normals = unit([[0.80270708, 0.06358589, 0.81143557],
...                 [0.40430152, 0.04015355, 0.61569626],
...                 [0.52316812, 0.1823291 , 0.58201526]])
>>> Ray(origins, normals)
Ray(origin=array([[0.88014699, 0.6794143 , 0.96630839],
       [0.50213094, 0.17339094, 0.60248626],
       [0.62116803, 0.47791134, 0.42499213]]), normal=array([[0.70218404, 0.05562303, 0.70981946],
       [0.54808068, 0.0544331 , 0.83465238],
       [0.65107206, 0.22690485, 0.72430613]]))
Also, you can:
>>> from mmcore.geom.interfaces import collection, Ray
>>> rays =collection(np.array([origins, normals]), Ray)
>>> rays
Ray(origin=array([[0.88014699, 0.6794143 , 0.96630839],
       [0.50213094, 0.17339094, 0.60248626],
       [0.62116803, 0.47791134, 0.42499213]]), normal=array([[0.70218404, 0.05562303, 0.70981946],
       [0.54808068, 0.0544331 , 0.83465238],
       [0.65107206, 0.22690485, 0.72430613]]))

Эти варианты будут корректно обрабатываться всеми функциями с множкственной диспетчерезацией, такими как intersect из mmcore.geom.intersect.

Также может потребоваться узнать является ли объект коллекцией:
>>> from mmcore.geom.interfaces  import collection, is_collection, Ray
>>> ray = Ray(origins[0], normals[0])
>>> is_collection(rays)
True
>>> is_collection(ray)
False

"""
# inspection PyTypeChecker
from collections import namedtuple
from enum import Enum
from typing import Sequence, Type, TypeVar

import numpy
import numpy as np

T = TypeVar('T')

Ray = namedtuple('Ray', ['origin', 'normal'])

Line2Pt = namedtuple('Line2Pt', ['start', 'end'])


def is_collection(instances: 'Sequence[T]'):
    return len(np.array(instances).shape) > 2


collection_types = dict()


class CollectionOrder(str, Enum):
    row = 'row'
    column = 'column'


def collection(instances: 'Sequence[T]', typ: 'Type[T]', dims=3, order=CollectionOrder.column):
    if typ not in collection_types.keys():
        collection_types[typ] = CollectionDispatchType(typ.__name__ + "Collection", (np.ndarray,),
                                                       {'typ_hash': hash(typ)})
    arr = np.array(instances)
    column_count = len(typ._fields)

    items_count = arr.size // column_count // dims
    arrt = collection_types[typ]((items_count, column_count, dims), arr.dtype)
    if order == CollectionOrder.column:
        for i in range(items_count):
            for j in range(column_count):
                arrt[i, j, ...] = arr[j, i, ...]
    else:
        for i in range(items_count):
            for j in range(column_count):
                arrt[i, j, ...] = arr[i, j, ...]

    del arr
    return arrt


class CollectionDispatchType(type):
    def __hash__(cls):
        return hash(cls.typ_hash)

    def __eq__(cls, other):
        return cls.typ_hash == hash(getattr(other, 'typ', other))


class ArrayInterface:
    @classmethod
    def __new_array_hook__(cls, arr):
        return arr
    def __add__(self, other):
        return self.__class__.__new_array_hook__(np.array(self).__add__(other))

    def __sub__(self, other):
        return self.__class__.__new_array_hook__(np.array(self).__sub__(other))

    def __mul__(self, other):
        return self.__class__.__new_array_hook__(np.array(self).__mul__(other))

    def __truediv__(self, other):
        return self.__class__.__new_array_hook__(np.array(self).__truediv__(other))
    def __divmod__(self, other):
        return self.__class__.__new_array_hook__(np.array(self).__divmod__(other))

    def __matmul__(self, other):
        return self.__class__.__new_array_hook__(np.array(self).__matmul__(other))

    def __getitem__(self, item):
        ...

    def __setitem__(self, item):
        ...
