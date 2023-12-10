from typing import Any

import numpy as np
from multipledispatch import dispatch

from mmcore.base import ageomdict
from mmcore.common.models.fields import FieldMap
from mmcore.geom.extrusion import extrude_polyline
from mmcore.geom.intersections import intersect
from mmcore.geom.line import Line
from mmcore.geom.mesh import build_mesh_with_buffer, create_mesh_buffer_from_mesh_tuple, union_mesh_simple
from mmcore.geom.plane import Plane, WXY, rotate_plane_around_plane, project
from mmcore.geom.rectangle import Rectangle, rect_to_mesh_vec, rect_to_plane
from mmcore.geom.vec import cross, dist, unit


class Box(Rectangle):
    """A class representing a box.

    Inherits from Rectangle.

    Attributes:
        h (float): The height of the box.
        lock (bool): A flag indicating if the box is locked.
    """

    def __init__(self, *args, h=3.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.h = h
        self.lock = False
        self._mesh = None

        self.field_map = sorted([
            FieldMap('u', 'u1'),
            FieldMap('v', 'v1'),
            FieldMap('h', 'h'),
            FieldMap('x', 'x'),
            FieldMap('y', 'y'),
            FieldMap('z', 'z'),

            FieldMap('area', 'area', backward_support=False),
            FieldMap('lock', 'lock'),

        ])



    @property
    def faces(self):
        return extrude_polyline(super().corners, self.normal * self.h)

    @property
    def uuid(self):
        return self.ecs_rectangle.uuid

    def rotate(self, angle, axis=None, origin=None, inplace=True):
        res = super().rotate(angle=angle, axis=axis, origin=origin, inplace=inplace)
        if inplace:
            self.update_mesh()
        else:
            return res

    def orient(self, plane):
        rect_to_plane(self, plane)
        self.update_mesh()

    def rotate_in_plane(self, angle, plane=WXY):
        # pln=rotate_plane_around_plane(self, plane, angle)
        # self.origin=pln.origin
        # self.xaxis=pln.yaxis
        # self.yaxis = pln.xaxis
        # self.zaxis = pln.zaxis

        rect_to_plane(self, rotate_plane_around_plane(self, plane, angle))
        self.update_mesh()

    def translate(self, translation, inplace=True):
        res = super().translate(translation, inplace=inplace)
        if inplace:
            self.update_mesh()
        else:
            return res

    def get_face(self, i):
        return self.corners[i]

    @property
    def caps(self):
        return self.faces[0], self.faces[-1]

    @property
    def sides(self):
        return self.faces[1:-1]


    def apply_forward(self, data):
        for m in self.field_map:
            m.forward(self, data)

    def apply_backward(self, data):
        for m in self.field_map:
            m.backward(self, data)
        self._dirty = True

    @property
    def x(self):
        return self.origin[0]

    @x.setter
    def x(self, val) -> None:
        self.origin[0] = val

    @property
    def y(self):
        return self.origin[1]

    @y.setter
    def y(self, val) -> None:
        self.origin[1] = val

    @property
    def z(self):
        return self.origin[2]

    @z.setter
    def z(self, val) -> None:
        self.origin[2] = val



    def transpose(self) -> 'Box':
        """transpose the box. Это сделвет продольными противоположные ребра"""
        return Box(float(self.u), float(self.h), h=float(self.v), xaxis=self.xaxis, normal=self.yaxis,
                   origin=self.origin + (self.normal * self.h))

    def elongate_ribs(self):
        return (Line.from_ends(pt, pt + (self.normal * self.h)) for pt in self.corners)

    @dispatch(Plane)
    def thickness_trim(self, plane: Plane) -> 'tuple[Box, bool, Any]':
        """
        Trim with thickness means that the box will be trimmed to the shortest trimmed side.
        In other words, no point of the box will extend beyond the trim plane.

        Подрезка с толщиной означает что бокс будет образан под самую короткую из подрезающихся сторон.
        Иными словами ни одна точка бокса не выйдет за подрезающую плоскость.

        :param plane: Trimming plane
        :type plane: <Plane>
        :return: Trimmed box
        :rtype: <Box>
        """
        intersection = False
        *res, = sorted(((i, r.plane_intersection(plane), r) for i, r in enumerate(self.elongate_ribs())),
                       key=lambda x: x[1][1])
        ixs, (w, t, point), line = res[0]
        if 1. >= t >= 0.:
            intersection = True

        line2 = Line.from_ends(line.start, point)

        return Box(self.ecs_uv.u, self.ecs_uv.v, h=dist(point, line2.start), xaxis=self.xaxis, origin=self.origin,
                   normal=self.normal), intersection, res

    def create_mesh(self):
        _props = dict()
        self.apply_forward(_props)
        fcs = self.faces
        self._mesh = build_mesh_with_buffer(
            union_mesh_simple(rect_to_mesh_vec(np.array(fcs)).reshape((len(fcs), 3)).tolist()),
            uuid=self.uuid,
            props=_props
        )
        self._mesh.owner = self

    def _init_mesh(self):
        self.create_mesh()

    def update_mesh(self, no_back=False):
        if not no_back:
            self.apply_backward(self._mesh.properties)
        fcs = self.faces
        ageomdict[self._mesh._geometry] = create_mesh_buffer_from_mesh_tuple(
            union_mesh_simple(rect_to_mesh_vec(np.array(fcs)).reshape((len(fcs), 3)).tolist()),
            uuid=self._mesh.uuid)

        self.apply_forward(self._mesh.properties)

    def to_mesh(self, **kwargs):
        self.create_mesh() if self._mesh is None else self.update_mesh()
        return self._mesh
    @dispatch(Line)
    def thickness_trim(self, line: Line) -> 'tuple[Box, bool, Any]':
        """
        This version of the method takes a line that will be converted to a vertical trim plane.

        Trim with thickness means that the box will be trimmed to the shortest trimmed side.
        In other words, no point of the box will extend beyond the trim plane.

        Эта версия метода принимает линию, которая будет преобразована в вертикальную плоскость подрезки.

        Подрезка с толщиной означает что бокс будет образан под самую короткую из подрезающихся сторон.
        Иными словами ни одна точка бокса не выйдет за подрезающую плоскость.
        :param line: Cut line
        :type line: <Line>
        :return: Trimmed box
        :rtype: <Box>
        """
        z = np.zeros(3)
        z[:2] = line.direction[:2]

        origin = line.start
        xaxis = unit(z)
        yaxis = np.array([0., 0., 1.])
        zaxis = cross(xaxis, yaxis)

        return self.thickness_trim(Plane(np.array([origin, xaxis, yaxis, zaxis])))

    def intersect_with_other_box(self, other_box: 'Box') -> 'Box':
        """
        Intersect this box with another box.

        Returns the resulting intersection box or None if no intersection found.

        :param other_box: Other box
        :type other_box: <Box>
        :return: Box resulting from intersection or None
        :rtype: <Box>
        """
        # Get the faces (corners) of the current box and the other box
        faces_current = np.array(self.faces)
        faces_other = np.array(other_box.faces)

        # Initialize an empty list to store intersection points
        intersect_points = []

        for face_current in faces_current:
            for face_other in faces_other:
                intersect_pts, _ = intersect(face_current, face_other)
                if intersect_pts is not None:
                    intersect_points.append(intersect_pts)

        # If no intersection points were found, there is no intersection and return None
        if not intersect_points:
            return None

        intersect_points = np.array(intersect_points)

        # Calculate the bounding box of the intersect points and return the corresponding Box

        return intersect_points

    @classmethod
    def from_rectangle(cls, rect: Rectangle, h=3.0):
        return cls(u=rect.u, v=rect.v, h=h, origin=rect.origin, xaxis=rect.xaxis, normal=rect.zaxis)

    @classmethod
    def from_corners(cls, corners):
        corners = np.array(corners)
        if (corners.ndim == 2) and (len(corners) <= 4):
            return cls.from_rectangle(super().from_corners(corners), h=1.0)
        else:
            fc = corners.flatten()
            fc = fc.reshape((len(fc) // 3, 3))
            rect = super().from_corners(fc[:4])
            return cls.from_rectangle(rect, h=rect.distance(fc[-1]).tolist())





def unpack_trim_details(res):
    """
    Unpacks the trim details.

    :param res: A list of tuples containing trim details.
    :type res: list

    :return: A tuple containing the unpacked trim details.
    :rtype: tuple

    """
    w, t, pts = list(zip(*list(zip(*sorted(res, key=lambda x: x[0])))[1]))
    return np.array(w), np.array(t), np.array(pts)
