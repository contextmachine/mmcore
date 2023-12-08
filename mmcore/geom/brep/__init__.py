from dataclasses import dataclass, field

import more_itertools
from itertools import count

__breps__ = []
__brep_counter__ = count()


def todict_gen(self):
    """
    Convert the object's attributes to a generator of key-value pairs.

    :param self: The object itself.
    :type self: object

    :return: A generator of key-value pairs.
    :rtype: generator
    """
    for k, v in self.__dict__.items():
        if not k.startswith('_'):
            yield k, v


@dataclass
class BrepComponent:
    """

    """
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
    """A class representing a container Brep component.

    A container component can contain other Brep components.

    Attributes:
        children (list[int]): The list of indices of the child components.

    Properties:
        is_container (bool): Whether the component is a container or not.
        is_bounded (bool): Whether the component is bounded or not.
        is_leaf (bool): Whether the component is a leaf or not.
    """
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
    """
    Represents a bounded component in a B-rep model.
    """
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
    """
    Represents a vertex in a Brep (Boundary Representation) model.

    :param geometry: The index of the vertex in the Brep geometry points. If not specified, the vertex has no geometry.
    """
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
    """Class representing an edge in a Brep.

    This class inherits from the BrepBoundedComponent class and represents an edge in a Boundary Representation (Brep) model.

    Attributes:
        bounded_by (List[int]): A list of indices representing the vertices that bound the edge.
        brep (Brep): The parent Brep object that the edge belongs to.

    Properties:
        bounds (List[BrepVertex]): A list of BrepVertex objects that bound the edge.

    """
    @property
    def bounds(self):
        return [self.brep.topology.vertices[i] for i in self.bounded_by]


@dataclass
class BrepLoop(BrepContainerComponent):
    """
    BrepLoop represents a loop in a boundary representation (BREP) model.

    Attributes:
        brep (BrepTopology): The BrepTopology object to which the loop belongs.
        children (List[int]): A list of indices representing the edges contained in the loop.

    Properties:
        contains (List[BrepTopology.Edge]): Returns a list of BrepTopology.Edge objects contained in the loop.

    Methods:
        __iter__: Returns an iterator over the BrepTopology.Edge objects contained in the loop.
    """
    @property
    def contains(self):
        return [self.brep.topology.edges[i] for i in self.children]

    def __iter__(self):
        return (self.brep.topology.edges[i] for i in self.children)


@dataclass
class BrepFace(BrepBoundedComponent):
    """
    Class BrepFace

    This class represents a face in a Brep (Boundary Representation) object.

    Inherits From:
        - BrepBoundedComponent

    Attributes:
        - bounds: A property that returns a list of the loops that bound the face.

    Methods:
        None

    Example:
        No example code provided.

    """
    @property
    def bounds(self):
        return [self.brep.topology.loops[i] for i in self.bounded_by]


@dataclass
class BrepShell(BrepContainerComponent):
    """A class representing a shell in a Brep object.

    Inherits from BrepContainerComponent. Provides methods and properties
    specific to shells in a Brep object.

    Attributes:
        contains: A list of faces contained in the shell.

    Methods:
        __iter__: Returns an iterator over the faces contained in the shell.

    """
    @property
    def contains(self):
        return [self.brep.topology.faces[i] for i in self.children]

    def __iter__(self):
        return (self.brep.topology.faces[i] for i in self.children)


@dataclass
class BrepTopology:
    """
    This class represents the topology of a Boundary Representation (B-Rep) model.

    Attributes:
        vertices (list): The list of vertices in the B-Rep model.
        edges (list): The list of edges in the B-Rep model.
        loops (list): The list of loops in the B-Rep model.
        faces (list): The list of faces in the B-Rep model.
        shells (list): The list of shells in the B-Rep model.
    """
    vertices: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    loops: list = field(default_factory=list)
    faces: list = field(default_factory=list)
    shells: list = field(default_factory=list)


@dataclass
class BrepGeometry:
    """
    Represents a Brep (Boundary Representation) geometry.

    :param surfaces: A list of surfaces in the Brep. Defaults to an empty list.
    :type surfaces: list
    :param curves: A list of curves in the Brep. Defaults to an empty list.
    :type curves: list
    :param points: A list of points in the Brep. Defaults to an empty list.
    :type points: list
    """
    surfaces: list = field(default_factory=list)
    curves: list = field(default_factory=list)
    points: list = field(default_factory=list)


@dataclass
class Brep:
    """
    Brep

    This class represents a Boundary Representation (Brep) object. It consists of two main components, topology and geometry. The topology component defines the connectivity between different
    * elements of the Brep, such as vertices, edges, loops, faces, and shells. The geometry component stores the actual geometric data, such as points.

    Attributes:
        topology (BrepTopology): The topology component of the Brep.
        geometry (BrepGeometry): The geometry component of the Brep.

    Methods:
        __post_init__()
            Initializes the Brep object and adds it to the global list of Brep objects.

        add_point(xyz)
            Adds a point to the Brep's geometry. If the point already exists, returns the corresponding vertex. Otherwise, creates a new vertex and returns it.

        add_line_edge(a, b)
            Adds a line edge to the Brep. The edge is defined by the given points. Returns the created edge.

        add_line_edge_from_vertices(va, vb)
            Adds a line edge to the Brep. The edge is defined by the given vertices. Returns the created edge.

        add_loop_from_edges(edges)
            Adds a loop to the Brep. The loop is defined by the given edges. Returns the created loop.

        add_loop_from_points(points)
            Adds a loop to the Brep. The loop is defined by the given points. Returns the created loop.

        add_face_from_loops(loops)
            Adds a face to the Brep. The face is defined by the given loops. Returns the created face.

        add_shell_from_faces(faces)
            Adds a shell to the Brep. The shell is defined by the given faces. Returns the created shell.

        add_vertex()
            Adds a vertex to the Brep's topology. Returns the created vertex.

        add_edge()
            Adds an edge to the Brep's topology. Returns the created edge.

        add_loop()
            Adds a loop to the Brep's topology. Returns the created loop.

        add_face()
            Adds a face to the Brep's topology. Returns the created face.

        add_shell()
            Adds a shell to the Brep's topology. Returns the created shell.
    """
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
    """
    :param v: The BrepVertex to extrude.
    :type v: BrepVertex
    :param vec: The vector indicating the direction and magnitude of the extrusion.
    :type vec: tuple[float, float, float]
    :return: The new BrepVertex created by extruding the original vertex along the specified vector.
    :rtype: BrepVertex
    """
    return v.brep.add_line_edge(v.point, tuple(np.array(v.point) + np.array(vec)))


def regions():
    """
    :return: a list of regions
    :rtype: list
    """
    ...
def extrude_line_edge(edge: BrepEdge, vec: tuple[float, float, float]):
    """
    Extrudes a line edge by a given vector.

    :param edge: The line edge to be extruded.
    :type edge: BrepEdge
    :param vec: The vector by which the edge will be extruded.
    :type vec: tuple[float, float, float]
    :return: The resulting loop after extruding the edge.
    :rtype: BrepLoop
    """
    v1, v2 = edge.bounds
    e1 = extrude_vertex(v1, vec)
    e2 = extrude_vertex(v2, vec)
    e3 = edge.brep.add_line_edge_from_vertices(e1.bounds[1], e2.bounds[1])
    return edge.brep.add_loop_from_edges([edge, e1, e2, e3])


def get_loop_vertices(loop: BrepLoop):
    """
    :param loop: The BrepLoop object representing a loop in a Brep topology.
    :type loop: BrepLoop
    :return: A list of vertices that form the loop, sorted according to the loop's edges.
    :rtype: List[BrepVertex]

    """
    return [loop.brep.topology.vertices[i] for i in sort_chain([edge.bounded_by for edge in loop])]


def extrude_polyline_loop(loop: BrepLoop, vec: tuple[float, float, float]):
    """

    :param loop: The polyline loop to be extruded.
    :type loop: BrepLoop

    :param vec: The vector specifying the direction and magnitude of the extrusion.
    :type vec: tuple[float, float, float]

    :return: The shell created from the extruded faces.
    :rtype: BrepShell

    """
    faces = []
    for edge in loop:
        faces.append(loop.brep.add_face_from_loops((extrude_line_edge(edge, vec),)))

    return loop.brep.add_shell_from_faces(faces)


def extrude_polyline_face(face: BrepFace, vec: tuple[float, float, float]):
    """
    :param face: The BrepFace to be extruded.
    :type face: BrepFace
    :param vec: The vector representing the direction and magnitude of the extrusion.
    :type vec: tuple[float, float, float]
    :return: The list of new BrepFaces created by the extrusion.
    :rtype: list[BrepFace]
    """
    new_face_loops = []
    faces = [face]
    for loop in face.bounds:
        fss = list(extrude_polyline_loop(loop, vec))
        new_face_loops.append(face.brep.add_loop_from_edges(list(face.bounds[0])[-1] for face in fss))
        faces.extend(fss)

    faces.append(face.brep.add_face_from_loops(new_face_loops))

    return faces


def sort_chain(ch):
    """
    Sorts a chain of elements based on certain conditions.

    :param ch: A list of tuples representing the chain of elements.
    :type ch: list
    :return: A sorted list of elements from the chain.
    :rtype: list
    """
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
    """
    :param loop: The BrepLoop object representing the loop for which to retrieve the loop points.
    :type loop: BrepLoop
    :return: A list of points representing the loop points.
    :rtype: List[Point]
    """
    return [loop.brep.topology.vertices[i].point for i in sort_chain([edge.bounded_by for edge in loop])]


def offset_polyline_loop(loop: BrepLoop, distance=-0.1):
    """
    :param loop: The BrepLoop object representing the polyline loop to be offset.
    :type loop: BrepLoop
    :param distance: The distance by which the polyline loop should be offset. Defaults to -0.1.
    :type distance: float
    :return: The newly created BrepLoop object representing the offset polyline loop.
    :rtype: BrepLoop
    """
    return loop.brep.add_loop_from_points(offset(get_loop_points(loop), distance))


def face_by_offset_loop(loop, distance=-0.1, offset_func=offset_polyline_loop):
    """
    :param loop: The loop to be used as the base loop for creating the face.
    :type loop: `Loop` object
    :param distance: The distance by which the offset of the loop will be calculated. Negative values indicate inward offset, while positive values indicate outward offset. Default value
    * is -0.1.
    :type distance: `float`
    :param offset_func: The function used for offsetting the loop. Default value is `offset_polyline_loop`.
    :type offset_func: `function`
    :return: The newly created face.
    :rtype: `Face` object

    """
    return loop.brep.add_face_from_loops((loop, offset_func(loop, distance)))


from mmcore.geom.shapes import Shape


def tess_polyline_face(face: BrepFace):
    """
    :param face: The input BrepFace object representing a face in a Brep model.
    :type face: BrepFace
    :return: The mesh data of the face represented by a Shape object.
    :rtype: MeshData

    This method takes a BrepFace object as input and computes the mesh data for the face. It creates a list of 2D points by extracting the point coordinates from the vertices of each loop
    * in the face. If the face has only one loop, a Shape object is created with the list of points as the outer boundary. The mesh data of this Shape object is returned. If the face has
    * multiple loops, a Shape object is created with the first list of points as the outer boundary and the remaining lists of points as holes. The mesh data of this Shape object is returned
    *.

    Example usage:
        face = BrepFace()
        mesh_data = tess_polyline_face(face)
    """
    lst = [[vert.point for vert in get_loop_vertices(loop)] for loop in face.bounds]
    if len(lst) == 1:
        return Shape(lst[0]).mesh_data
    else:
        return Shape(lst[0], holes=lst[1:]).mesh_data


from mmcore.base import AGroup


def tess_polyline_brep(bp: Brep, name="Brep", **kwargs):
    """
    :param bp: The Brep object representing the input polyline.
    :type bp: Brep
    :param name: Optional name for the resulting AGroup object. Default is "Brep".
    :type name: str
    :param kwargs: Additional keyword arguments to be passed to the AGroup constructor.
    :type kwargs: dict
    :return: The AGroup object containing the tessellated polyline faces.
    :rtype: AGroup

    This method takes a Brep object representing a polyline as input and returns an AGroup object containing the tessellated faces of the input polyline. Each face is converted to a mesh
    * using the tess_polyline_face method and added to the AGroup object with a unique UUID and name.
    """
    grp = AGroup(**kwargs, name=name)

    for face in bp.topology.faces:
        grp.add(tess_polyline_face(face).to_mesh(uuid=f'{grp.uuid}-face-{face.ixs}', name=f'{name} face {face.ixs}'))
    return grp
