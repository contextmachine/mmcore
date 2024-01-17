from itertools import zip_longest

import more_itertools
import numpy as np

from mmcore.base.models import BaseMaterial
from mmcore.common.models.fields import FieldMap
from mmcore.common.models.mixins import MeshViewSupport
from mmcore.geom.extrusion import extrude_line, extrude_line2, extrude_polyline, polyline_to_lines
from mmcore.geom.mesh import union_mesh_simple
from mmcore.geom.mesh.shape_mesh import mesh_from_bounds, mesh_from_bounds_and_holes
from mmcore.geom.polycurve import PolyCurve
from mmcore.geom.shapes.area import (polygon_area, polygon_area_vectorized, to_polygon_area,
    )
from mmcore.geom.vec import unit
from mmcore.numeric import split_by_parts


class Boundary(MeshViewSupport):
    __field_map__ = ()

    def __init__(self, boundary: np.ndarray = None, count=4, color=(0.3, 0.3, 0.3), **kwargs):
        if boundary is None:
            boundary = np.zeros((count, 3))

        self.boundary = boundary
        self.__init_support__(color=color, **kwargs)

    def to_mesh_view(self):
        return mesh_from_bounds(self.boundary.tolist())

    @property
    def area(self):
        return float(polygon_area(np.array([*self.boundary, self.boundary[0]])))

    @property
    def control_points(self):
        return self.boundary

    @control_points.setter
    def control_points(self, v):
        self.boundary = np.array(v, float)


class BoundaryIterator:
    def __init__(self, boundary):
        self._obj = iter(boundary.boundary)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._obj)
class Face(Boundary):
    __field_map__ = ()

    def __init__(self, boundary: np.ndarray = None, holes=None, count=4, **kwargs):

        self.holes = holes if holes is not None else holes
        super().__init__(boundary, count=count, **kwargs)

    def to_mesh_view(self):
        if self.holes is not None:
            return mesh_from_bounds_and_holes(self.boundary.tolist(), holes=[hole.tolist() for hole in self.holes]
                    )
        else:
            return super().to_mesh_view()

    @property
    def area(self):
        if self.holes is not None:
            return Boundary.area.fget(self) - np.sum([polygon_area(to_polygon_area(h)) for h in self.holes]
                    )
        return Boundary.area.fget(self)


class ShapeFace(Boundary):
    def __init__(self, boundary: np.ndarray = None, h=1, normal=np.array([0., 0., 1]), **kwargs):
        self._structure = [len(i) for i in boundary]
        self.h = h
        self.normal = normal

        super().__init__(np.array([*more_itertools.value_chain(*boundary)]), **kwargs)

    @property
    def holes(self):

        if len(self._structure) > 1:
            return np.vsplit(self.boundary, np.cumsum(self._structure)[:-1])[1:]

    @property
    def loops(self):
        if len(self._structure) > 1:

            return split_by_parts(self.boundary, self._structure[:-1])
        else:
            return [self.boundary]

    @property
    def faces(self):
        lns = []
        for l in self.loops:
            lns.extend(extrude_line2(polyline_to_lines(l), self.normal * self.h))

        return [self.boundary, *lns, self.boundary + self.normal * self.h]

    @property
    def sides(self):
        return np.array(self.faces[1:][:-1])

    @property
    def caps(self):
        f = self.faces
        if self.holes is None:
            return f[0], f[-1]
        else:
            return split_by_parts(f[0], self._structure[:-1]), split_by_parts(f[-1], self._structure[:-1])

    def to_mesh_view(self):
        if self.holes is not None:
            return union_mesh_simple([mesh_from_bounds_and_holes(cap[0].tolist(), cap[1:]) for cap in self.caps] + [
                mesh_from_bounds(side.tolist()) for side in self.sides]
                    )

        else:
            return union_mesh_simple(
                    [mesh_from_bounds(cap.tolist()) for cap in self.caps] + [mesh_from_bounds(side.tolist()) for side in
                                                                             self.sides]
                    )



class PolyCurveBoundary(Boundary):
    __field_map__ = (FieldMap("area", "area", backward_support=False),)

    def __init__(self, boundary: np.ndarray = None, count=4, **kwargs):
        if boundary is not None:
            self._contour = PolyCurve.from_points(boundary)
        else:
            self._contour = PolyCurve()

        super().__init__(boundary, count=count, **kwargs)

    @property
    def boundary(self):
        return self._contour.corners

    @boundary.setter
    def boundary(self, v):
        if v is not None:
            self._contour.corners = (np.array(v, float) if not isinstance(v, np.ndarray) else v)


class PolyCurveFace(Face):
    __field_map__ = (FieldMap("area", "area", backward_support=False),)

    def __init__(self, boundary: np.ndarray = None, holes=None, count=4, **kwargs):
        self._contour = PolyCurve.from_points(boundary)

        self._holes = None

        super().__init__(boundary, holes=holes, count=count, **kwargs)

    def __setitem__(self, key: int, value: PolyCurve) -> None:
        """
        index 0 is the boundary curve
        :param key: index of boundary or hole
        :type key: int
        :param value: PolyCurve
        :type value:  PolyCurve
        :return: PolyCurve object
        :rtype: PolyCurve
        """

        if key == 0:
            self._contour = value
        elif key < 0:
            self._holes[key] = value
        else:
            self._holes[key - 1] = value

    def __getitem__(self, item: int) -> PolyCurve:
        """
        index 0 is the boundary curve
        :param item: index of boundary or hole
        :type item: int
        :return: PolyCurve object
        :rtype: PolyCurve
        """
        if item == 0:
            return self._contour
        elif item < 0:
            return self._holes[item]
        else:
            return self._holes[item - 1]

    @property
    def boundary(self):
        return self._contour.corners

    @boundary.setter
    def boundary(self, v):
        self._contour.corners = (np.array(v, float) if not isinstance(v, np.ndarray) else v)

    @property
    def holes(self):
        if self._holes is None:
            return self._holes
        elif len(self._holes) == 0:
            return None
        else:
            return np.array([hole.corners for hole in self._holes])

    @holes.setter
    def holes(self, v):
        if hasattr(v, "__length__"):
            if len(v) > 0:
                if self._holes is None:
                    self._holes = [PolyCurve.from_points(hole) for hole in v]

                else:
                    for new, current in list(zip_longest(v, self._holes, fillvalue=None)
                            ):
                        if current is None:
                            current = PolyCurve.from_points(np.array(new, float))
                            self._holes.append(current)
                        elif new is None:
                            self._holes.remove(current)
                        else:
                            current.corners = (np.array(new, float) if not isinstance(new, np.ndarray) else new)


class OffsetFace(PolyCurveBoundary):
    __field_map__ = (FieldMap("area", "area", backward_support=False),)

    def __init__(self, boundary: np.ndarray = None, distance=0.1, count=4, **kwargs):
        self._offset_distance = distance
        super().__init__(boundary, count=count, **kwargs)

    def to_mesh_view(self):
        if self.holes is not None:
            return mesh_from_bounds_and_holes(self.boundary.tolist(), holes=self.holes.tolist()
                    )
        else:
            return super().to_mesh_view()

    @property
    def area(self):
        if self.holes is not None:
            return Boundary.area.fget(self) - np.sum(polygon_area_vectorized(to_polygon_area(self.holes))
                    )
        return Boundary.area.fget(self)

    @property
    def holes(self):
        if not hasattr(self, "_contour_hole"):
            self._contour_hole = self._contour.offset(self._offset_distance)
        return np.array([self._contour_hole.corners], float)

    @property
    def distance(self):
        return self._offset_distance

    @distance.setter
    def distance(self, v):
        self._offset_distance = v
        self._contour_hole = self._contour.offset(self._offset_distance)
