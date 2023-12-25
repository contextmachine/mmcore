import typing
from abc import ABCMeta, abstractmethod

from mmcore.base import ageomdict, amatdict, AMesh
from mmcore.common.models.fields import FieldMap
from mmcore.geom.mesh import build_mesh_with_buffer, create_mesh_buffer_from_mesh_tuple, MeshTuple, simpleMaterial
from mmcore.base.models.gql import ColorRGB, BaseMaterial
from mmcore.base.userdata.controls import encode_control_points

class ViewSupport(metaclass=ABCMeta):
    """

    The `ViewSupport` class is an abstract base class that provides support for managing and manipulating view data.

    Attributes:
        __field_map__ (tuple[FieldMap]): A tuple of `FieldMap` objects that define the mapping between view fields
        and data fields.
        field_map (list[FieldMap]): A list of `FieldMap` objects that define the mapping between view fields and data
        fields.
        _uuid (Any): A private attribute that holds the UUID of the view.

    Methods:
        __init_support__(self, uuid=None, field_map=None):
            Initializes the `ViewSupport` object.

            Args:
                uuid (Any, optional): The UUID of the view. Defaults to None.
                field_map (list[FieldMap], optional): A list of `FieldMap` objects defining the mapping between view
                fields and data fields. Defaults to None.

            Returns:
                None

        apply_forward(self, data):
            Applies the forward mapping to the given data.

            Args:
                data (dict): A dictionary containing the data to be mapped.

            Returns:
                None

        describe(self):
            Generates a dictionary describing the view data.

            Returns:
                dict: A dictionary containing the view data.

        apply_backward(self, data):
            Applies the backward mapping to the given data.

            Args:
                data (dict): A dictionary containing the updated values.

            Returns:
                None

        uuid(self)
            Gets the UUID of the view.

            Returns:
                Any: The UUID of the view.

        uuid(self, v)
            Sets the UUID of the view.

            Args:
                v (Any): The UUID of the view.

            Returns:
                None

    """
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
    """

    :class: MeshViewSupport

    A class that provides support for creating and updating mesh views.

    Attributes:
    - _mesh (Optional[AMesh]): The mesh object associated with the view.

    Methods:
    - __init_support__(self, uuid=None, field_map=None): Initializes the support object with the specified UUID and
    field map.
    - to_mesh_view(self) -> MeshTuple: Abstract method that must be overridden to return a MeshTuple representing the
    mesh view.
    - to_mesh_material(self) -> BaseMaterial: Returns the material to be used for the mesh.
    - create_mesh(self, forward=True, controls=None, **kwargs): Creates a new mesh using the current mesh view.
    - _init_mesh(self): Initializes the mesh object.
    - update_mesh_geometry(self): Updates the geometry of the mesh object.
    - update_mesh_material(self): Updates the material of the mesh object.
    - update_mesh(self, no_back=False): Updates the mesh object with the current mesh view, geometry, and material.
    - to_mesh(self, **kwargs): Creates or updates the mesh object and returns it.

    Example:

    >>> # Here we define a concrete child class of MeshViewSupport
    ... class ConcreteMeshSupportObject(MeshViewSupport):
    ...     def __init__(self):
    >>>         super().__init_support__()
    ...
    ...     def to_mesh_view(self) -> MeshTuple:
    ...         # Implement this method to return a valid MeshTuple
    ...         return MeshTuple(vertices=[], faces=[], lines=[], color=ColorRGB(r=255, g=255, b=255))
    """
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
                print('Мы ТУТ')
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

