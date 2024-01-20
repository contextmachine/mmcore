from typing import Any
from uuid import uuid4

import numpy as np

from mmcore.common.models.fields import FieldMap
from mmcore.common.models.mixins import MeshViewSupport
from mmcore.geom.extrusion import extrude_polyline
from mmcore.geom.boundary import Boundary
from mmcore.geom.line import Line
from mmcore.geom.mesh import union_mesh_simple
from mmcore.geom.mesh.shape_mesh import mesh_from_bounds
from mmcore.geom.plane import Plane, WXY, rotate_plane_around_plane
from mmcore.geom.rectangle import Rectangle, rect_to_mesh_vec, rect_to_plane
from mmcore.geom.vec import cross, dist, unit


class Box(Rectangle, MeshViewSupport):
    """A class representing a box.

    Inherits from Rectangle.

    Attributes:
        h (float): The height of the box.
        lock (bool): A flag indicating if the box is locked.
    """

    __field_map__ = (FieldMap("u", "u"), FieldMap("v", "v"), FieldMap("h", "h"), FieldMap("x", "x"), FieldMap("y", "y"),
                     FieldMap("z", "z"), FieldMap("area", "area", backward_support=False),)

    def __init__(self, u, v, h=3.0, color=(0.5, 0.5, 0.5), origin=np.array([0.0, 0.0, 0.0]),
                 xaxis=np.array([1.0, 0.0, 0.0]),
                 normal=np.array([0.0, 0.0, 1.0]), uuid=None, **kwargs):
        super().__init__(u, v, origin=origin, xaxis=xaxis, normal=normal, uuid=uuid)
        #

        self.h = h

        self.__init_support__(self.uuid, color=color, **kwargs)


    @property
    def lock(self):
        return self._lock

    @lock.setter
    def lock(self, val: bool):
        self._lock = val

    @property
    def faces(self):
        return extrude_polyline(super().corners, self.normal * self.h)

    def rotate(self, angle, axis=None, origin=None, inplace=True):
        res = super().rotate(angle=angle, axis=axis, origin=origin, inplace=inplace)
        if inplace:
            self.update_mesh()
        else:
            return res

    def to_mesh(self, **kwargs):
        self.create_mesh() if self._mesh is None else self.update_mesh()
        return self._mesh

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

    # @property
    # def control_points(self):
    #    return np.array([self.origin,
    #     self.origin+self.xaxis*self.u,
    #     self.origin+self.yaxis*self.v]
    #     )
    #
    # @control_points.setter
    # def control_points(self, val):
    #    origin,x,y=np.array(val,float)
    #    xaxis=x-origin
    #    yaxis=y-origin
    #
    #    zaxis=cross(unit(xaxis), unit(yaxis))
    #    yaxis=cross(zaxis,unit(xaxis))
    #
    #
    #    self.xaxis=unit(xaxis)
    #
    #    self.yaxis=unit(yaxis)
    #
    #    self.zaxis=zaxis

    def transpose(self) -> "Box":
        """transpose the box. Это сделвет продольными противоположные ребра"""
        return Box(float(self.u), float(self.h), h=float(self.v), xaxis=self.xaxis, normal=self.yaxis,
                origin=self.origin + (self.normal * self.h), )

    def elongate_ribs(self):
        return (Line.from_ends(pt, pt + (self.normal * self.h)) for pt in self.corners)

    def thickness_trim(self, plane: Plane) -> "tuple[Box, bool, Any]":
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
        (*res,) = sorted(((i, r.plane_intersection(plane), r) for i, r in enumerate(self.elongate_ribs())),
                key=lambda x: x[1][1], )
        ixs, (w, t, point), line = res[0]
        if 1.0 >= t >= 0.0:
            intersection = True

        line2 = Line.from_ends(line.start, point)

        return (Box(self.ecs_uv.u, self.ecs_uv.v, h=dist(point, line2.start), xaxis=self.xaxis, origin=self.origin,
                normal=self.normal, ), intersection, res,)

    def to_mesh_view(self):
        fcs = self.faces
        return union_mesh_simple(rect_to_mesh_vec(np.array(fcs)).reshape((len(fcs), 3)).tolist()
                )

    def thickness_trim_line(self, line: Line) -> "tuple[Box, bool, Any]":
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
        yaxis = np.array([0.0, 0.0, 1.0])
        zaxis = cross(xaxis, yaxis)

        return self.thickness_trim(Plane(np.array([origin, xaxis, yaxis, zaxis])))

    # def intersect_with_other_box(self, other_box: 'Box') -> 'Box':
    #    """
    #    Intersect this box with another box.
    #
    #    Returns the resulting intersection box or None if no intersection found.
    #
    #    :param other_box: Other box
    #    :type other_box: <Box>
    #    :return: Box resulting from intersection or None
    #    :rtype: <Box>
    #    """
    #    # Get the faces (corners) of the current box and the other box
    #    faces_current = np.array(self.faces)
    #    faces_other = np.array(other_box.faces)
    #
    #    # Initialize an empty list to store intersection points
    #    intersect_points = []
    #
    #    for face_current in faces_current:
    #        for face_other in faces_other:
    #            #intersect_pts, _ = intersect(face_current, face_other)
    #            #if intersect_pts is not None:
    #            #    intersect_points.append(intersect_pts)
    #
    #    # If no intersection points were found, there is no intersection and return None
    #    if not intersect_points:
    #        return None
    #
    #    intersect_points = np.array(intersect_points)
    #
    #    # Calculate the bounding box of the intersect points and return the corresponding Box
    #
    #    return intersect_points
    #
    @classmethod
    def from_rectangle(cls, rect: Rectangle, h=3.0):
        return cls(u=rect.u, v=rect.v, h=h, origin=rect.origin, xaxis=rect.xaxis, normal=rect.zaxis, )

    @classmethod
    def from_corners(cls, corners, h=1.0):
        corners = np.array(corners)
        if (corners.ndim == 2) and (len(corners) <= 4):
            return cls.from_rectangle(super().from_corners(corners), h=h)
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


def to_mesh_view(self):
    return mesh_from_bounds(self.boundary.tolist())
