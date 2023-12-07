import typing
from operator import methodcaller

import itertools
import numpy as np

from mmcore.tree.avl import AVL


def vectorize(**kws):
    def decorate(fun):
        if 'doc' not in kws:
            kws['doc'] = fun.__doc__

        vecfun = np.vectorize(fun, **kws)

        @functools.wraps(fun)
        def wrap(*args, **kwargs):
            return vecfun(*args, **kwargs)

        return wrap

    return decorate
class MapMethodDescriptor:
    def __init__(self, fun):
        super().__init__()
        self._fun = fun

    def __set_name__(self, owner, name):
        self._name = "_" + name
        self._caller = self._fun(name)

    def __get__(self, instance, owner):
        if instance:
            return self._caller(instance)
        elif owner:
            return self


@MapMethodDescriptor
def mapper(name):
    def wrap(seq):
        def inner(*args, **kwargs):
            if args == () and len(kwargs) == 0:
                return map(methodcaller(name), seq)
            else:
                return map(methodcaller(name, *args, **kwargs), seq)

        return inner

    return wrap


def even_filter(iterable, reverse=False):
    def even_filter_num(item):
        return reverse != ((iterable.index(item) % 2) == 0)

    return filter(even_filter_num, iterable)


SIGN = 'ijklmnopqrstuvwxyzabcdefgh'


def _extract_shape_sign(arg, sign, cnt):
    for j in arg.shape:
        if j not in sign:
            sign[j] = SIGN[divmod(next(cnt), 26)[1]]
        yield sign[j]


def npextracttp(arg, sign=None, cnt=None):
    if sign is None:
        sign = dict()
        cnt = itertools.count()

    if isinstance(arg, np.ndarray):
        s = tuple(_extract_shape_sign(arg, sign, cnt))
        tp = np.ndarray[s, arg.dtype]

        return tp
    elif isinstance(arg, (tuple, list)):
        signs = []
        tps = []
        for i in arg:
            tp = npextracttp(i, sign=sign, cnt=cnt)
            signs.append(tp.signature)
            tps.append(tp)

        tpp = GenericAlias(type(arg), tps)

        return tpp



    else:

        return type(arg)


from types import GenericAlias


def extract_type(arg, sign=None,
                 cnt=None):
    """

    Parameters
    ----------
    arg :

    Returns
    -------
    >>> extract_type([3, 'dd', (3.4, 1.0),(("foo", 1), (3.4, 1.0))])
    list[int, str, tuple[float, float], tuple[tuple[str, int], tuple[float, float]]]

    >>> import itertools
    >>> from mmcore.func import dsp,vectorize,extract_type
    >>> from mmcore.geom.mesh import simpleMaterial,union_mesh
    >>> from mmcore.geom.mesh.shape_mesh import mesh_from_bounds, mesh_from_shapes
    >>> from mmcore.geom.extrusion import Extrusion,MultiExtrusion
    >>> ext=MultiExtrusion([[(33.918146530459282, 9.7709589114110127, 0.0), (39.047486045598042, -0.86767119406203452, 0.0), (53.177523693468643, 5.9450255290184026, 0.0), (48.048184178329883, 16.583655634491450, 0.0)], [(40.992682917195808, 10.961570234677605, 0.0), (39.860231578544713, 8.0495525067176406, 0.0), (47.787390949102402, 5.6228710667510029, 0.0), (49.162510431750164, 9.6673401333620692, 0.0)], [(37.514439519910297, 6.9171011680665426, 0.0), (39.698452815880266, 2.3872958134621438, 0.0), (42.367802399843569, 2.9535214827876981, 0.0), (40.830904154531368, 6.1890967360765430, 0.0)]],h=10.0)
    >>> um2=union_mesh(list(mesh_from_shapes(ext.caps,[(0.3, 0.3, 0),(0.3, 0.3, 0)],(dict(),dict()))))
    >>> aa=dict()
    >>> extract_type(um2,aa , itertools.count())
    mmcore.geom.mesh.MeshTuple[dict[str, typing.Union[numpy.ndarray[('j',), dtype('int64')], numpy.ndarray[('i',), dtype('float64')]]], #attributes
                               numpy.ndarray[('k',), dtype('int64')],                                                                   #indices
                               dict[str, tuple[dict[str, dict], dict[str, dict]]]]                                                      #extras
    >>> aa
    {72: 'i', 24: 'j', 84: 'k'}

    """

    if isinstance(arg, np.ndarray):
        if sign is None:
            return np.ndarray[arg.shape, arg.dtype]
        else:
            return npextracttp(arg, sign=sign, cnt=cnt)

    if not isinstance(arg, (str, bytes, bytearray)) and hasattr(arg, '__iter__'):

        if isinstance(arg, (tuple, set)):

            return GenericAlias(type(arg), [*tuple(extract_type(i, sign=sign, cnt=cnt) for i in arg)])
        elif isinstance(arg, list):
            return GenericAlias(type(arg), [*tuple(set(tuple(extract_type(i, sign=sign, cnt=cnt) for i in arg)))])
        elif isinstance(arg, dict):
            kt = tuple(set(extract_type(i, sign=sign, cnt=cnt) for i in arg.keys()))
            vt = tuple(set(extract_type(i, sign=sign, cnt=cnt) for i in arg.values()))
            if len(kt) == 1:
                kt = kt[0]
            elif len(kt) == 0:
                return dict
            else:
                kt = GenericAlias(typing.Union, kt)
            if len(vt) < 2:
                vt = vt[0]
            else:
                vt = GenericAlias(typing.Union, vt)
            return dict[kt, vt]
        else:
            return GenericAlias(type(arg), [*tuple(extract_type(i, sign=sign, cnt=cnt) for i in arg)])
    else:

        return type(arg)


import functools

registry = dict()


def dsp(*types, excluded=()):
    def wrap(fun):
        name = fun.__name__
        if name not in registry:
            registry[name] = AVL()
        tree = registry[name]

        tree.insert(hash(str(GenericAlias(tuple, [*types]))), fun)

        @functools.wraps(fun)
        def wrapper(*args):
            nonlocal name, excluded
            l = ()
            for i, a in enumerate(args):
                if i not in excluded:
                    l = l + (a,)

            sign = dict()
            cnt = itertools.count()
            return registry[name].search(hash(str(
                extract_type(l, sign=sign, cnt=cnt)))).data(*args)

        return wrapper

    return wrap
