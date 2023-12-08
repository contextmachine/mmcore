import numpy as np

from mmcore.base import ageomdict
from mmcore.common.models.fields import FieldMap
from mmcore.geom.extrusion import Extrusion, to_mesh
from mmcore.geom.mesh import create_mesh_buffer_from_mesh_tuple
from mmcore.geom.rectangle import Length, Rectangle, to_mesh

"""self.field_map = sorted([
    FieldMap('u', 'u1'),
    FieldMap('v', 'v1'),
    FieldMap('area', 'floor_area', backward_support=False),
    FieldMap('lock', 'lock'),
    FieldMap('high', 'floor_high'),

])"""


class AnalyticModel:
    def __init_subclass__(cls, field_map=()):
        cls.field_map = sorted(field_map)

    def __init__(self):

        self._mesh = None

    def apply_forward(self, data):
        for m in self.field_map:
            m.forward(self, data)

    def apply_backward(self, data):
        for m in self.field_map:
            m.backward(self, data)

    def update_mesh(self):

        self.apply_forward(self._mesh.properties)
        ageomdict[self._mesh._geometry] = create_mesh_buffer_from_mesh_tuple(to_mesh(self), uuid=self._mesh.uuid)
        self.apply_backward(self._mesh.properties)

    @property
    def uuid(self):
        ...


class RectangleModel(Rectangle, AnalyticModel,
                     field_map=[FieldMap('u', 'u1'),
                                FieldMap('v', 'v1'),
                                FieldMap('area', 'area', backward_support=False),
                                FieldMap('lock', 'lock')]):
    def __init__(self, u: 'float|Length' = 1, v: 'float|Length' = 1, xaxis=np.array((1, 0, 0)),
                 normal=np.array((0, 0, 1)),
                 origin=np.array((0, 0, 0))):

        super().__init__(u, v, xaxis=xaxis, normal=normal, origin=origin)

        self.lock = False

    @property
    def uuid(self):
        return self.ecs_rectangle.uuid

    def to_mesh(self, uuid=None, **kwargs):
        if uuid is None:
            uuid = self.uuid
        if self._mesh is None:
            _props = dict()
            self.apply_forward(_props)
            self._mesh = super().to_mesh(uuid=uuid, props=_props)
            self._mesh.owner = self
            return self._mesh
        else:
            self.update_mesh()
            return self._mesh


class RectangleExtrusionModel(Extrusion, RectangleModel,
                              field_map=[FieldMap('u', 'u1'),
                                         FieldMap('v', 'v1'),
                                         FieldMap('h', 'high'),
                                         FieldMap('area', 'area', backward_support=False),
                                         FieldMap('lock', 'lock')]):

    def __init__(self, u: 'float|Length' = 1, v: 'float|Length' = 1, h=3.0, xaxis=np.array((1, 0, 0)),
                 normal=np.array((0, 0, 1)),
                 origin=np.array((0, 0, 0))):
        RectangleModel.__init__(self, u, v, xaxis=xaxis, normal=normal, origin=origin)

        super().__init__(self, h, vec=self.normal)
