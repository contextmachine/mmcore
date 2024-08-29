


import numpy as np
cimport numpy as np

cdef class Tree:
    pass

cdef class Leaf(Tree):
    cdef public tuple data
    __slots__=("data"   ,)

    def __init__(self, tuple data):
        self.data = data

cdef class Empty(Tree):
    cdef Tree node
    __slots__=("node",)
    def __init__(self, Tree node=None):
        self.node = node

cdef class Full(Tree):
    cdef Tree node
    __slots__=("node",)
    def __init__(self, Tree node=None):
        self.node = node

cdef class Root(Tree):
    cdef public Tree a, b, c, d, e, f, g, h
    __slots__=("a", "b", "c", "d", "e", "f", "g", "h",)
    def __init__(self, Tree a, Tree b, Tree c, Tree d, Tree e, Tree f, Tree g, Tree h):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f
        self.g = g
        self.h = h

def build_tree3d(tuple min, tuple max, int i) -> Tree:
    cdef double  xmin  = min[0]
    cdef double  ymin  = min[1]
    cdef double  zmin  = min[2]
    cdef double  xmax  = max[0]
    cdef double  ymax  = max[1]
    cdef double  zmax  = max[2]



    return _build_tree3d(xmin, ymin, zmin,xmax, ymax, zmax, i)

cdef Tree _build_tree3d(double xmin,double ymin ,double zmin,double xmax,double ymax ,double zmax, int i):
    if i == 0:
        return Leaf(((xmin,ymin,zmin), (xmax,ymax,zmax)))


    cdef double xmid = (xmin + xmax) / 2
    cdef double ymid = (ymin + ymax) / 2
    cdef double zmid = (zmin + zmax) / 2

    cdef Tree a, b, c, d, e, f, g, h


    a = _build_tree3d(xmin, ymin, zmin,xmid, ymid, zmid, i - 1)
    b = _build_tree3d(xmid, ymin, zmin, xmax, ymid, zmid, i - 1)
    c = _build_tree3d(xmin, ymid, zmin, xmid, ymax, zmid, i - 1)
    d = _build_tree3d(xmid, ymid, zmin, xmax, ymax, zmid, i - 1)
    e = _build_tree3d(xmin, ymin, zmid, xmid, ymid, zmax, i - 1)
    f = _build_tree3d(xmid, ymin, zmid, xmax, ymid, zmax, i - 1)
    g = _build_tree3d(xmin, ymid, zmid, xmid, ymax, zmax, i - 1)
    h = _build_tree3d(xmid, ymid, zmid, xmax, ymax, zmax, i - 1)

    return Root(a, b, c, d, e, f, g, h)