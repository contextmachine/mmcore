import numpy as np

from mmcore.base import ageomdict
from mmcore.common.models.fields import FieldMap
from mmcore.geom.extrusion import extrude_polyline
from mmcore.geom.mesh import build_mesh_with_buffer, create_mesh_buffer_from_mesh_tuple, union_mesh_simple
from mmcore.geom.rectangle import Rectangle, rect_to_mesh_vec


class Box(Rectangle):
    """
>>> from mmcore.base.sharedstate import serve
>>> from mmcore.common.viewer import create_group
>>> from mmcore.geom.vec import unit
>>> group= create_group('fff')
>>> r1=Box(u=10,v=20,h=20)
>>> r2=Box(u=30,v=40,h=60)
>>> r2.translate((50,50,0))
>>> r1.rotate(np.pi/4, axis=unit(np.array([1.,1.,1.])), origin=r2.origin)
>>> group.add(r1.to_mesh())
>>> group.add(r2.to_mesh())
>>> serve.start()
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
            FieldMap('area', 'area', backward_support=False),
            FieldMap('lock', 'lock'),

        ])

        self._init_mesh()

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

    def _init_mesh(self, **kwargs):
        _props = dict()

        self.apply_forward(_props)
        fcs = self.faces
        self._mesh = build_mesh_with_buffer(
            union_mesh_simple(rect_to_mesh_vec(np.array(fcs)).reshape((len(fcs), 3)).tolist()),
            uuid=self.uuid,
            props=_props, **kwargs

        )
        self._mesh.owner = self

    def to_mesh(self, **kwargs):
        self._init_mesh(**kwargs) if self._mesh is None else self.update_mesh()
        return self._mesh

    def apply_forward(self, data):
        for m in self.field_map:
            m.forward(self, data)

    def apply_backward(self, data):
        for m in self.field_map:
            m.backward(self, data)
        self._dirty = True

    def update_mesh(self):

        self.apply_backward(self._mesh.properties)
        fcs = self.faces
        ageomdict[self._mesh._geometry] = create_mesh_buffer_from_mesh_tuple(
            union_mesh_simple(rect_to_mesh_vec(np.array(fcs)).reshape((len(fcs), 3)).tolist()),

            uuid=self._mesh.uuid)

        self.apply_forward(self._mesh.properties)
