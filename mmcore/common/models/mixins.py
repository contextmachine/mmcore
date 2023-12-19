import typing
from abc import ABCMeta, abstractmethod

from mmcore.base import ageomdict, amatdict, AMesh
from mmcore.common.models.fields import FieldMap
from mmcore.geom.mesh import build_mesh_with_buffer, create_mesh_buffer_from_mesh_tuple, MeshTuple, simpleMaterial
from mmcore.base.models.gql import ColorRGB, BaseMaterial
from mmcore.base.userdata.controls import encode_control_points

class ViewSupport(metaclass=ABCMeta):
    __field_map__: tuple[FieldMap] = ()
    field_map: list[FieldMap]
    _uuid = None

    @abstractmethod
    def __init_support__(self, uuid=None, field_map=None):
        self._dirty = True
        self._lock = False
        if uuid is not None:
            self._uuid = uuid
        if not field_map:
            self.field_map = list(sorted(list(self.__class__.__field_map__)))
        else:
            self.field_map = sorted(list(self.__class__.__field_map__) + list(field_map))

    def apply_forward(self, data):
        for m in self.field_map:
            m.forward(self, data)

    def describe(self):
        data = dict()
        self.apply_forward(data)
        return data



    def apply_backward(self, data):
        """

        :param data: dict with updated values
        :type data: dict
        :return: None
        :rtype: None
        """
        for m in self.field_map:
            m.backward(self, data)
        self._dirty = True

    @property
    def uuid(self):
        return self._uuid

    @uuid.setter
    def uuid(self, v):
        self._uuid = v



class MeshViewSupport(ViewSupport, metaclass=ABCMeta):
    _mesh: typing.Optional[AMesh] = None

    def __init_support__(self, uuid=None, field_map=None):
        super().__init_support__(uuid, field_map)
        self._mesh = None

    @abstractmethod
    def to_mesh_view(self) -> MeshTuple:
        ...

    def to_mesh_material(self) -> BaseMaterial:
        """
        Override this method so that it returns vertexColor Material
        if you want to define the color of the geometry with the color attribute.
        """
        return simpleMaterial

    def create_mesh(self, forward=True, controls=None, **kwargs):
        if controls is None:
            controls = dict()

        _props = dict()
        if forward:
            self.apply_forward(_props)
        self._mesh = build_mesh_with_buffer(self.to_mesh_view(), uuid=self.uuid, props=_props, _controls=controls,
                                            **kwargs
                                            )
        self._mesh.owner = self
        if hasattr(self, 'control_points'):

            self._mesh._controls['path'] = dict(points=encode_control_points(self.control_points))

    def _init_mesh(self):
        self.create_mesh(forward=False)

    def update_mesh_geometry(self):
        ageomdict[self._mesh._geometry] = create_mesh_buffer_from_mesh_tuple(self.to_mesh_view(), uuid=self._mesh.uuid
                                                                             )

    def update_mesh_material(self):
        amatdict[self._mesh._material] = self.to_mesh_material()

    def update_mesh(self, no_back=False):
        if self._mesh is not None:
            if not no_back:
                # это плохое место
                self.apply_backward(self._mesh.properties)

            self.update_mesh_geometry()
            self.update_mesh_material()

            self.apply_forward(self._mesh.properties)
        else:
            return ValueError("Mesh not initialized")

    def to_mesh(self, **kwargs):
        self.create_mesh() if self._mesh is None else self.update_mesh()
        return self._mesh
