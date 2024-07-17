import numpy as np
from mmcore.geom.bvh import BVHNode, build_bvh
from mmcore.geom.surfaces import surface_bvh,Surface,CurveOnSurface
from mmcore.geom.implicit.tree import ImplicitTree3D

class Vertex:
    def __init__(self, point):
        self.point = np.array(point)
        self.edges = []

class Edge:
    def __init__(self, start_vertex, end_vertex, curve):
        self.start_vertex = start_vertex
        self.end_vertex = end_vertex
        self.curve = curve
        self.faces = []


class Face:
    def __init__(self, surface):
        self.surface = surface
        self.loops = []
        self.implicit_tree = None

    def build_implicit_tree(self, depth=3):

        self.implicit_tree = ImplicitTree3D(self.surface, depth)

class Loop:
    def __init__(self):
        self.edges = []

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

