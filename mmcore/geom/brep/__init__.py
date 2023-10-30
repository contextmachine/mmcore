from dataclasses import dataclass, field

import more_itertools
from itertools import count

__breps__ = []
__brep_counter__ = count()


def todict_gen(self):
    for k, v in self.__dict__:
        if not k.startswith('_'):
            yield k, v


@dataclass
class BrepComponent:
    ixs: int
    _brep: int

    @property
    def is_container(self) -> bool:
        return False

    @property
    def is_bounded(self) -> bool:
        return False

    @property
    def is_leaf(self) -> bool:
        return False

    @property
    def brep(self):
        return __breps__[self._brep]

    def todict(self) -> dict:
        return dict(todict_gen(self))


import typing


@dataclass
class BrepContainerComponent(BrepComponent):
    children: list[int] = field(default_factory=list)

    def __contains__(self, val):
        return val.ixs in self.children

    @property
    def is_container(self):
        return True

    @property
    def is_bounded(self):
        return False

    @property
    def is_leaf(self):
        return False


@dataclass
class BrepBoundedComponent(BrepComponent):
    bounded_by: list[int] = field(default_factory=list)

    @property
    def is_container(self):
        return False

    @property
    def is_bounded(self):
        return True

    @property
    def is_leaf(self):
        return False


@dataclass
class BrepVertex(BrepComponent):
    geometry: typing.Optional[int] = None

    @property
    def is_leaf(self):
        return True

    @property
    def point(self):
        if self.geometry is not None:
            return self.brep.geometry.points[self.geometry]


@dataclass
class BrepEdge(BrepBoundedComponent):
    @property
    def bounds(self):
        return [self.brep.topology.vertices[i] for i in self.bounded_by]


@dataclass
class BrepLoop(BrepContainerComponent):
    @property
    def contains(self):
        return [self.brep.topology.edges[i] for i in self.children]

    def __iter__(self):
        return (self.brep.topology.edges[i] for i in self.children)


@dataclass
class BrepFace(BrepBoundedComponent):
    @property
    def bounds(self):
        return [self.brep.topology.loops[i] for i in self.bounded_by]


@dataclass
class BrepShell(BrepContainerComponent):
    @property
    def contains(self):
        return [self.brep.topology.faces[i] for i in self.children]

    def __iter__(self):
        return (self.brep.topology.faces[i] for i in self.children)


@dataclass
class BrepTopology:
    vertices: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    loops: list = field(default_factory=list)
    faces: list = field(default_factory=list)
    shells: list = field(default_factory=list)


@dataclass
class BrepGeometry:
    surfaces: list = field(default_factory=list)
    curves: list = field(default_factory=list)
    points: list = field(default_factory=list)


@dataclass
class Brep:
    topology: BrepTopology = field(default_factory=BrepTopology)
    geometry: BrepGeometry = field(default_factory=BrepGeometry)

    def __post_init__(self):
        __breps__.append(self)
        self._ixs = next(__brep_counter__)

    def add_point(self, xyz):
        if xyz in self.geometry.points:
            i = self.geometry.points.index(xyz)
            if self.topology.vertices[i].geometry == i:
                return self.topology.vertices[i]
            else:
                vi = 0
                success = False
                while True:
                    if self.topology.vertices[vi] == i:
                        success = True
                        break
                    elif i >= len(self.topology.vertices):

                        break
                    else:
                        vi += 1
                if success:
                    return self.topology.vertices[vi]
                else:
                    return
        else:
            i = len(self.geometry.points)
            self.geometry.points.append(xyz)
            v = self.add_vertex()
            v.geometry = i
            return v

    def add_line_edge(self, a, b):
        va = self.add_point(a)
        vb = self.add_point(b)
        edge = self.add_edge()
        edge.bounded_by.extend([va.ixs, vb.ixs])
        return edge

    def add_line_edge_from_vertices(self, va, vb):

        edge = self.add_edge()
        edge.bounded_by.extend([va.ixs, vb.ixs])
        return edge

    def add_loop_from_edges(self, edges):
        loop = self.add_loop()
        loop.children.extend(edge.ixs for edge in edges)
        return loop

    def add_loop_from_points(self, points):
        pts = list(points)
        pts.append(points[0])
        return self.add_loop_from_edges(self.add_line_edge(a, b) for a, b in more_itertools.pairwise(pts))

    def add_face_from_loops(self, loops):
        face = self.add_face()
        face.bounded_by.extend(loop.ixs for loop in loops)
        return face

    def add_shell_from_faces(self, faces):
        shell = self.add_shell()
        shell.children.extend(face.ixs for face in faces)
        return shell

    def add_vertex(self):
        o = BrepVertex(len(self.topology.vertices), self._ixs)
        self.topology.vertices.append(o)
        return o

    def add_edge(self):
        o = BrepEdge(len(self.topology.edges), self._ixs)
        self.topology.edges.append(o)
        return o

    def add_loop(self):
        o = BrepLoop(len(self.topology.loops), self._ixs)
        self.topology.loops.append(o)
        return o

    def add_face(self):
        o = BrepFace(len(self.topology.faces), self._ixs)
        self.topology.faces.append(o)
        return o

    def add_shell(self):
        o = BrepShell(len(self.topology.shells), self._ixs)
        self.topology.shells.append(o)
        return o


import numpy as np


def extrude_vertex(v: BrepVertex, vec: tuple[float, float, float]):
    return v.brep.add_line_edge(v.point, tuple(np.array(v.point) + np.array(vec)))


def extrude_line_edge(edge: BrepEdge, vec: tuple[float, float, float]):
    v1, v2 = edge.bounds
    e1 = extrude_vertex(v1, vec)
    e2 = extrude_vertex(v2, vec)
    e3 = edge.brep.add_line_edge_from_vertices(e1.bounds[1], e2.bounds[1])
    return edge.brep.add_loop_from_edges([edge, e1, e2, e3])


def get_loop_vertices(loop: BrepLoop):
    return [loop.brep.topology.vertices[i] for i in sort_chain([edge.bounded_by for edge in loop])]


def extrude_polyline_loop(loop: BrepLoop, vec: tuple[float, float, float]):
    faces = []
    for edge in loop:
        faces.append(loop.brep.add_face_from_loops((extrude_line_edge(edge, vec),)))

    return loop.brep.add_shell_from_faces(faces)


def extrude_polyline_face(face: BrepFace, vec: tuple[float, float, float]):
    new_face_loops = []
    faces = [face]
    for loop in face.bounds:
        fss = list(extrude_polyline_loop(loop, vec))
        new_face_loops.append(face.brep.add_loop_from_edges(list(face.bounds[0])[-1] for face in fss))
        faces.extend(fss)

    faces.append(face.brep.add_face_from_loops(new_face_loops))

    return faces


def sort_chain(ch):
    count = i = j = 0
    res = []
    while True:
        if count == len(ch):
            break
        elif j == i:
            i += 1
            pass
        elif ch[j][1] in ch[i]:
            if ch[i].index(ch[j][1]) == 1:
                ch[i].reverse()
            res.append(ch[j][0])
            j = i
            i = 0
            count += 1
        else:
            i += 1
    return res


from mmcore.geom.shapes import offset


def get_loop_points(loop: BrepLoop):
    return [loop.brep.topology.vertices[i].point for i in sort_chain([edge.bounded_by for edge in loop])]


def offset_polyline_loop(loop: BrepLoop, distance=-0.1):
    return loop.brep.add_loop_from_points(offset(get_loop_points(loop), distance))


def face_by_offset_loop(loop, distance=-0.1, offset_func=offset_polyline_loop):
    return loop.brep.add_face_from_loops((loop, offset_func(loop, distance)))


from mmcore.geom.shapes import Shape


def tess_polyline_face(face: BrepFace):
    lst = [[vert.point for vert in get_loop_vertices(loop)] for loop in face.bounds]
    if len(lst) == 1:
        return Shape(lst[0]).mesh_data
    else:
        return Shape(lst[0], holes=lst[1:]).mesh_data


from mmcore.base import AGroup


def tess_polyline_brep(bp: Brep, name="Brep", **kwargs):
    grp = AGroup(**kwargs, name=name)

    for face in bp.topology.faces:
        grp.add(tess_polyline_face(face).to_mesh(uuid=f'{grp.uuid}-face-{face.ixs}', name=f'{name} face {face.ixs}'))
    return grp
