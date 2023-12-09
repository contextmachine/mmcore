from mmcore.base import ageomdict
from mmcore.common.models.fields import FieldMap
from mmcore.geom.extrusion import Extrusion, to_mesh
from mmcore.geom.mesh import build_mesh_with_buffer, create_mesh_buffer_from_mesh_tuple, union_mesh_simple
from mmcore.geom.mesh.shape_mesh import mesh_from_bounds
from mmcore.geom.rectangle import Rectangle


class RectangleModel(Rectangle):
    def __init__(self, *args, high=3.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.high = high
        self.lock = False
        self._mesh = None
        self.field_map = sorted([
            FieldMap('u', 'u1'),
            FieldMap('v', 'v1'),
            FieldMap('area', 'floor_area', backward_support=False),
            FieldMap('lock', 'lock'),
            FieldMap('high', 'floor_high'),

        ])

    @property
    def uuid(self):
        return self.ecs_rectangle.uuid

    def extrusion(self):
        return Extrusion(self, self.high)

    def to_mesh(self, **kwargs):

        if self._mesh is None:
            _props = dict()
            self.apply_forward(_props)
            ext = self.extrusion()

            self._mesh = build_mesh_with_buffer(
                union_mesh_simple([mesh_from_bounds(face, color=(0.3, 0.3, 0.3)) for face in ext.faces]),
                uuid=self.uuid,
                props=_props

            )




        else:
            self.update_mesh()
        self._mesh.owner = self
        return self._mesh

    def apply_forward(self, data):
        for m in self.field_map:
            m.forward(self, data)

    def apply_backward(self, data):
        for m in self.field_map:
            m.backward(self, data)

    def update_mesh(self):

        self.apply_backward(self._mesh.properties)
        print(self._mesh.properties)
        ageomdict[self._mesh._geometry] = create_mesh_buffer_from_mesh_tuple(to_mesh(self.extrusion()),
                                                                             uuid=self._mesh.uuid)
        self.apply_forward(self._mesh.properties)
        print(self._mesh.properties)
