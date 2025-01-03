from __future__ import annotations
from numpy.typing import NDArray
import numpy as np
from mmcore.geom.bvh import BVHNode, build_bvh
from mmcore.geom.nurbs import NURBSSurface
from mmcore.geom.proto import CurveProtocol, SurfaceProtocol
from mmcore.geom.surfaces import surface_bvh,Surface,CurveOnSurface
from mmcore.geom.implicit.tree import ImplicitTree3D
from dataclasses import dataclass,field,InitVar

from mmcore.numeric.intersection.ssx.boundary_intersection import extract_surface_boundaries


@dataclass(slots=True, unsafe_hash=True)
class Vertex:
    represented_by:int

@dataclass(slots=True, unsafe_hash=True)
class Edge:


    represented_by: int
    bounded_by:list[Vertex]=field(default_factory=list)


@dataclass(slots=True, unsafe_hash=True, frozen=True)
class Loop:
    edges:list[Edge]=field(default_factory=list)


@dataclass(slots=True, unsafe_hash=True)
class Face:
    represented_by: int
    bounded_by:list[Loop]=field(default_factory=list)


@dataclass(slots=True, unsafe_hash=True, frozen=True)
class Shell:
    faces:list[Face]=field(default_factory=list)


@dataclass(slots=True, unsafe_hash=True, frozen=True)
class Solid:
    bounded_by:list[Shell]=field(default_factory=list)

@dataclass(slots=True, unsafe_hash=True, frozen=True)
class BRepGeometry:
    points:list[NDArray[float]]=field(default_factory=list)
    curves:list[CurveProtocol]=field(default_factory=list)
    surfaces: list[SurfaceProtocol]=field(default_factory=list)


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

@dataclass(slots=True, unsafe_hash=True, frozen=True)
class BRepTopology:
    solids:list[Solid]=field(default_factory=list)
    shells:list[Shell]=field(default_factory=list)
    faces:list[Face]=field(default_factory=list)
    loops:list[Loop]=field(default_factory=list)
    edges:list[Edge]=field(default_factory=list)
    vertices:list[Vertex]=field(default_factory=list)
    def add_vertex(self, representation:int):
        v=Vertex(representation)
        self.vertices.append(v)
        return v

    def add_edge(self, representation: int,start:Vertex,end:Vertex):
        v = Edge(representation,[start,end])
        self.edges.append(v)
        return v

    def add_loop(self, edges:list[Edge]):
        v = Loop(edges)
        self.loops.append(v)
        return v
    def add_face(self, representation:int, bounded_by:list[Loop]):
        v = Face(representation, bounded_by)
        self.faces.append(v)
        return v
    def add_shell(self, faces:list[Face]):
        v = Shell(faces)
        self.shells.append(v)
        return v
    def add_solid(self, shells:list[Shell]):
        v = Solid(shells)
        self.solids.append(v)
        return v

@dataclass(slots=True, unsafe_hash=True)
class BRep:
    geometry:BRepGeometry=field(default_factory=BRepGeometry)
    topology:BRepTopology=field(default_factory=BRepTopology)


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



class BRepStructure:
    def __init__(self):
        self.vertices = []
        self.edges = []
        self.faces = []
        self.vertex_bvh = None

    def add_vertex(self, point):
        vertex = Vertex(point)
        self.vertices.append(vertex)
        self.vertex_bvh = None  # Invalidate BVH
        return vertex

    def add_edge(self, start_vertex, end_vertex, curve):
        edge = Edge(start_vertex, end_vertex, curve)
        self.edges.append(edge)
        start_vertex.edges.append(edge)
        end_vertex.edges.append(edge)
        return edge

    def add_face(self, surface, loops):
        face = Face(surface)
        for loop in loops:
            face.loops.append(loop)
            for edge in loop.edges:
                edge.faces.append(face)
        self.faces.append(face)
        face.build_implicit_tree()
        return face

    def build_vertex_bvh(self):
        vertex_objects = [BVHNode(vertex.point, vertex.point, vertex) for vertex in self.vertices]
        self.vertex_bvh = build_bvh(vertex_objects)

    def find_nearest_vertex(self, point):
        if self.vertex_bvh is None:
            self.build_vertex_bvh()

        def distance_sq(node):
            return np.sum((node.bounding_box.center - point) ** 2)

        nearest = min(self.vertex_bvh.traverse(), key=distance_sq)
        return nearest.object

    def get_adjacent_faces(self, vertex):
        return list(set(face for edge in vertex.edges for face in edge.faces))

    def get_face_boundary(self, face):
        boundary_edges = set()
        for loop in face.loops:
            boundary_edges.update(loop.edges)
        return list(boundary_edges)

    def is_point_on_face(self, point, face):
        if face.implicit_tree is None:
            face.build_implicit_tree()
        return face.implicit_tree.implicit(point) <= 1e-6  # Tolerance value

