import numpy as np

from mmcore.common.models.fields import FieldMap
from mmcore.common.models.mixins import MeshViewSupport
from mmcore.geom.mesh.shape_mesh import mesh_from_bounds, mesh_from_bounds_and_holes
from mmcore.geom.polycurve import PolyCurve
from mmcore.geom.shapes.area import polygon_area, polygon_area_vectorized, to_polygon_area


class Boundary(MeshViewSupport):
    __field_map__ = (FieldMap('area', 'area', backward_support=False),)

    def __init__(self, boundary: np.ndarray = None, count=4, **kwargs):
        if boundary is None:
            boundary = np.zeros((count, 3))

        self.boundary = boundary
        self.__init_support__(**kwargs)

    def to_mesh_view(self):
        return mesh_from_bounds(self.boundary.tolist())

    @property
    def area(self):
        return polygon_area(np.array([*self.boundary, self.boundary[0]]))

    @property
    def control_points(self):
        return self.boundary

    @control_points.setter
    def control_points(self, v):
        self.boundary = np.array(v, float)


class Face(Boundary):
    __field_map__ = (FieldMap('area', 'area', backward_support=False),)
    def __init__(self,boundary: np.ndarray = None, holes=None, count=4, **kwargs):
        self.holes = np.array(holes,float) if holes is not None else holes
        super().__init__(boundary,count=count,**kwargs)
    def to_mesh_view(self):
        if self.holes is not None:
            return mesh_from_bounds_and_holes(self.boundary.tolist(), holes=self.holes.tolist())
        else:
            return super().to_mesh_view()

    @property
    def area(self):
        if self.holes is not None:
            return Boundary.area.fget(self)-np.sum(polygon_area_vectorized(to_polygon_area(self.holes)))
        return Boundary.area.fget(self)

class PolyCurveBoundary(Boundary):
    def __init__(self,boundary: np.ndarray = None, count=4, **kwargs):
        self._contour=PolyCurve.from_points(boundary)
        super().__init__(boundary,count=count,**kwargs)
    @property
    def boundary(self):
        return self._contour.corners

    @boundary.setter
    def boundary(self,v):
        self._contour.corners=np.array(v,float) if not isinstance(v,np.ndarray) else v


class OffsetFace(PolyCurveBoundary):
    __field_map__ = (FieldMap('area', 'area', backward_support=False),)
    def __init__(self,boundary: np.ndarray = None, distance=0.1, count=4, **kwargs):

        self._offset_distance = distance
        super().__init__(boundary,count=count,**kwargs)



    def to_mesh_view(self):
        if self.holes is not None:
            return mesh_from_bounds_and_holes(self.boundary.tolist(), holes=self.holes.tolist())
        else:
            return super().to_mesh_view()

    @property
    def area(self):
        if self.holes is not None:
            return Boundary.area.fget(self)-np.sum(polygon_area_vectorized(to_polygon_area(self.holes)))
        return Boundary.area.fget(self)

    @property
    def holes(self):
        if not hasattr(self,'_contour_hole'):
            self._contour_hole=self._contour.offset(self._offset_distance)
        return np.array([self._contour_hole.corners],float)

    @property
    def distance(self):
        return self._offset_distance
    @distance.setter
    def distance(self, v):
        self._offset_distance=v
        self._contour_hole=self._contour.offset(self._offset_distance)
