from __future__ import annotations

from collections import namedtuple
from functools import lru_cache
from types import GenericAlias
from typing import TypeVar, Generic

from numpy.typing import NDArray
import numpy as np
from mmcore.geom.bvh import BVHNode, build_bvh
from mmcore.geom.nurbs import NURBSSurface, NURBSCurve
from mmcore.geom.proto import CurveProtocol, SurfaceProtocol
from mmcore.geom.surfaces import surface_bvh,Surface,CurveOnSurface
from mmcore.geom.implicit.tree import ImplicitTree3D
from dataclasses import dataclass,field,InitVar

from mmcore.numeric.vectors import scalar_norm
from mmcore.numeric.intersection.ssx.boundary_intersection import extract_surface_boundaries

T = TypeVar('T')


@dataclass(slots=True)
class GeometryRepresentation(Generic[T]):
    index: int
    value: T
    def __hash__(self):
        if isinstance(self.value,(np.ndarray,memoryview)):
            if hasattr(self.value,'tobytes'):

                return hash(np.round(self.value,8).tobytes())


        return self.value.__hash__()
    def __eq__(self, other):
        return self.__hash__()==other.__hash__()


@dataclass(slots=True, unsafe_hash=True)
class Vertex:
    represented_by:GeometryRepresentation[NDArray[float]]

@dataclass(slots=True,frozen=True)
class _EndParameterResult:
    success:bool
    param: float
    dist: float

@dataclass(slots=True,frozen=True)
class _EdgeCurveParametersResult:

    start:_EndParameterResult
    end:_EndParameterResult
    def check(self):
        if not self.start.success and not self.end.success :
            raise ValueError(
                f"The starting and ending vertices does not belong to the edge. Error values: ({self.start.dist},{self.end.dist})")
        if not self.start.success:
            raise ValueError(
                f"The starting vertex does not belong to the edge. Error value: {self.start.dist}")
        if  not self.end.success:
            raise ValueError(
                f"The starting vertex does not belong to the edge. Error value: {self.end.dist}")

@lru_cache(maxsize=None)
def _get_edge_curve_parameters_at_points(curve:NURBSCurve,
                       start:NDArray[float],
                       end:NDArray[float],
                       tol=1e-6
                       )->_EdgeCurveParametersResult:


    success_start, (start_param, start_dist)=closest_point_on_nurbs_curve(curve,start,tol=tol,on_curve=True)
    success_end, (end_param, end_dist)=closest_point_on_nurbs_curve(curve, end,tol=tol,on_curve=True)



    return _EdgeCurveParametersResult(_EndParameterResult(success_start, start_param, start_param), _EndParameterResult(success_end, end_param, end_dist))


def get_edge_ends_parameters(edge:Edge,check:bool=False)->tuple[float,float]:
    res=_get_edge_curve_parameters_at_points(edge.represented_by.value, edge.bounded_by[0].represented_by.value,edge.bounded_by[1].represented_by.value)
    if check:res.check()
    return res.start.param,res.end.param


@dataclass(slots=True, unsafe_hash=True)
class Edge:
    represented_by: GeometryRepresentation[NURBSCurve]
    bounded_by:list[Vertex]=field(default_factory=list)

    def get_ends_parameters(self,check:bool=False):
        return get_edge_ends_parameters(self,check=check)

    def tessellate(self)->NDArray[float]:
        return self.represented_by.value.evaluate_multi(np.linspace(*self.get_ends_parameters(),100))





@dataclass(slots=True, unsafe_hash=True, frozen=True)
class Loop:
    edges:list[Edge]=field(default_factory=list)

    def tessellate(self)->NDArray[float]:
       return np.concatenate([edge.tessellate() for edge in self.edges])







@dataclass(slots=True, unsafe_hash=True)
class Face:
    represented_by:GeometryRepresentation[NURBSSurface]
    bounded_by:list[Loop]=field(default_factory=list)






@dataclass(slots=True, unsafe_hash=True, frozen=True)
class Shell:
    faces:list[Face]=field(default_factory=list)


@dataclass(slots=True, unsafe_hash=True, frozen=True)
class Solid:
    bounded_by:list[Shell]=field(default_factory=list)

@dataclass(slots=True, unsafe_hash=True, frozen=True)
class BRepGeometry:
    points:list[GeometryRepresentation[NDArray[float]]]=field(default_factory=list)
    curves:list[GeometryRepresentation[NURBSCurve]]=field(default_factory=list)
    surfaces: list[GeometryRepresentation[NURBSSurface]]=field(default_factory=list)

    def add(self, geom: NDArray[float] | NURBSCurve | NURBSSurface, check: bool = True):

        if isinstance(geom, NURBSCurve):
            container = self.curves
        elif isinstance(geom,NURBSSurface):
            container=self.surfaces
        else:
            container=self.points

        index = len(container)

        r = GeometryRepresentation(index, geom)

        if check:
            if r in container:
                r = container[container.index(r)]

                return r
        container.append(r)

        return r
    def add_point(self, point:NDArray, check=True):
        return self.add(point,check=check)

    def add_curve(self, curve:NURBSCurve, check=True):
        return self.add(curve,check=check)
    def add_surface(self, surf:NURBSSurface, check=True):
        return self.add(surf,check=check)

class Sparce(dict):
    def __init__(self):

        super().__init__()
        self._rc = dict()

    def __setitem__(self, key, value):
        a, b = key
        if a not in self._rc:
            self._rc[a] = []
        if b not in self._rc:
            self._rc[b] = []
        if b not in self._rc[a]:
            self._rc[a].append(b)
        if a not in self._rc[b]:
            self._rc[b].append(a)
        super().__setitem__(key, value)
        super().__setitem__((b, a), value)

    def __getitem__(self, item):
        if isinstance(item, (tuple, list)) and len(item) == 2:
            return super().__getitem__(item)
        else:
            return {k: self.get((item, k), None) for k in self._rc.keys()}

from mmcore.numeric.closest_point import min_distance, closest_point_on_nurbs_curve


@dataclass(slots=True, unsafe_hash=True, frozen=True)
class BRep:

    solids:list[Solid]=field(default_factory=list)
    shells:list[Shell]=field(default_factory=list)
    faces:list[Face]=field(default_factory=list)
    loops:list[Loop]=field(default_factory=list)
    edges:list[Edge]=field(default_factory=list)
    vertices:list[Vertex]=field(default_factory=list)
    geometry:BRepGeometry=field(default_factory=BRepGeometry)
    def add_vertex(self, representation:GeometryRepresentation[NDArray[float]], check=True):
        if check:
            for v in self.vertices:
                if v.represented_by == representation:
                    return v

        return self.add_vertex_unsafe(representation)

    def add_edge_unsafe(self, representation: GeometryRepresentation[NURBSCurve], start: Vertex, end: Vertex
                        ):
        edge = Edge(represented_by=representation, bounded_by=[start, end])
        self.edges.append(edge)
        return edge

    def add_edge(self, representation: GeometryRepresentation[NURBSCurve],start:Vertex,end:Vertex,check=True
                        ):
        if check:
            for v in self.edges:
                pts=(v.bounded_by[0].represented_by,v.bounded_by[1].represented_by)
                if v.represented_by == representation and start.represented_by in pts and end.represented_by in pts:
                    return v


        return self.add_edge_unsafe(representation,start,end)




    def add_vertex_unsafe(self,representation: GeometryRepresentation[NDArray[float]]
                        ):

        v = Vertex(representation)
        self.vertices.append(v)

        return v





    def add_loop(self, edges:list[Edge]):
        v = Loop(edges)
        self.loops.append(v)
        return v

    def add_face_unsafe(self, representation:GeometryRepresentation[NURBSSurface], bounded_by:list[Loop]):
        v = Face(representation, bounded_by)
        self.faces.append(v)
        return v

    def add_face(self,representation:GeometryRepresentation[NURBSSurface], bounded_by:list[Loop],check=True ):

        if check:


                for v in self.faces:
                    if hash(v.represented_by) == hash(representation):
                            if bounded_by[0] is v.bounded_by[0]:

                                if sorted((hash(b) for b in bounded_by[1:]))==sorted((hash(b) for b in bounded_by[1:])):
                                    return v




        return self.add_face_unsafe(representation,bounded_by)


    def add_shell(self, faces:list[Face],check=True):
        if check:
            raise NotImplemented
        return self.add_shell_unsafe(faces)

    def add_shell_unsafe(self, faces:list[Face]):
        v = Shell(faces)
        self.shells.append(v)
        return v
    def add_solid(self, shells:list[Shell],check=True):
        if check:
            raise NotImplemented
        return self.add_solid_unsafe(shells)

    def add_solid_unsafe(self, shells:list[Shell]):
        v = Solid(shells)
        self.solids.append(v)
        return v


def nurbs_surf_to_brep(surf:NURBSSurface):


    geom=BRepGeometry()
    boundaries = extract_surface_boundaries(surf)
    geom.surfaces.append(surf)
    face=Face(0)
    loop=Loop()

    for i,boundary in enumerate(boundaries):
        geom.curves.append(boundary)
        start,end=np.array(boundary.start()),np.array(boundary.end())
        edge=Edge(i, [None,None])

        for j,pt in enumerate([start,end]):
            if len(geom.points)==0:
                geom.points.append(tuple(pt))
                edge.bounded_by[j] = Vertex(0)
            else:
                if tuple(pt) in geom.points:
                    edge.bounded_by[j]=Vertex(geom.points.index(tuple(pt)))
                else:
                    edge.bounded_by[j] = Vertex(len(geom.points))
                    geom.points.append(
                        tuple(pt)
                    )


        loop.edges.append(edge)
    face.bounded_by.append(loop)
    return Shell([face]),geom


