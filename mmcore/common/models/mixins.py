import copy
import typing
from abc import ABCMeta, abstractmethod
from dataclasses import asdict
from uuid import uuid4

from mmcore.base import ageomdict, amatdict, AMesh, AGroup, Props
from mmcore.common.models.fields import FieldMap
from mmcore.geom.mesh import (build_mesh_with_buffer, create_mesh_buffer_from_mesh_tuple, MeshTuple, simpleMaterial,
                              vertexMaterial,
                              )
from mmcore.base.models.gql import (ColorRGB, BaseMaterial, MeshPhongMaterial, MeshStandardMaterial,
                                    create_material_uuid
    )
from mmcore.base.userdata.controls import encode_control_points
from mmcore.common.viewer import (ViewerGroupObserver, ViewerObservableGroup, create_group, group_observer, propsdict
    )


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
    __view_name__ = "base"

    @abstractmethod
    def __init_support__(self, uuid=None, field_map=None, hooks=()):
        self._dirty = True
        self._lock = False
        self.hooks = list(hooks)
        if uuid is not None:
            self._uuid = uuid
        if not field_map:
            self.field_map = list(sorted(list(self.__class__.__field_map__)))
        else:
            self.field_map = sorted(list(self.__class__.__field_map__) + list(field_map)
                    )

    def apply_forward(self, data):
        for m in self.field_map:
            m.forward(self, data)

    @property
    def props(self):
        return propsdict.get(self.uuid, Props(uuid=self.uuid))

    @props.setter
    def props(self, v):
        propsdict.get(self.uuid, Props(uuid=self.uuid)).update(dict(v))


    def describe(self):
        data = dict()
        self.apply_forward(data)
        return data

    def hook(self):
        for h in self.hooks:
            h(self)

    def update_view(self, data, no_back=False):
        if not no_back:
            self.apply_backward(data)
        self.hook()
        self.apply_forward(data)

    def add_field(self, source_field_name, target_field_name=None, backward_support=True, callbacks=(None, None),
                  sort=False):
        self.field_map.append(
            FieldMap(source_field_name, target_field_name if target_field_name is not None else source_field_name,
                     backward_support=backward_support, callbacks=callbacks
                     )
            )
        if sort:
            self.field_map.sort()

    def source_fields_dict(self):
        return {field.source_field_name: field for field in self.field_map}

    def target_fields_dict(self):
        return {field.target_field_name: field for field in self.field_map}

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
    __view_default_material = simpleMaterial
    _material_color = None
    _material_prototype = None
    _material = None
    __view_name__ = "mesh"

    def __init_support__(self, uuid=None, field_map=None, hooks=(), color=None,
            material_prototype: BaseMaterial = None, ):
        super().__init_support__(uuid, field_map, hooks=hooks)
        self._mesh = None
        self._material_prototype = material_prototype
        self._material = copy.copy(self.material_prototype)

        self.material_color = color

    def hook(self):
        super().hook()
        self.update_mesh_geometry()
        self.update_mesh_material()

    @property
    def color(self):
        return self.material_color

    @color.setter
    def color(self, v):
        self.material_color = v

    @property
    def material_color(self):
        return self._material_color

    @material_color.setter
    def material_color(self, v: "tuple | list | np.ndarray | ColorRGB"):
        if v is None:
            v = (0.5, 0.5, 0.5)
        self._material_color = v if isinstance(v, ColorRGB) else ColorRGB(*v)
        self._material.color = self._material_color.decimal
        self._material.uuid = create_material_uuid(self._material, self.material_prototype.uuid
                )

        if self._mesh is not None:
            self.update_mesh_material()

    @property
    def material(self):
        return self._material

    @material.setter
    def material(self, v: BaseMaterial):
        self._material = v
        if self._mesh is not None:
            self.update_mesh_material()

    @property
    def material_prototype(self):
        return (self._material_prototype if self._material_prototype is not None else (
            self.__class__.__view_default_material))

    @material_prototype.setter
    def material_prototype(self, v: BaseMaterial):
        self._material_prototype = v

    @abstractmethod
    def to_mesh_view(self) -> MeshTuple:
        ...

    def to_mesh_material(self) -> BaseMaterial:
        """
        Override this method so that it returns vertexColor Material
        if you want to define the color of the geometry with the color attribute.
        """
        return self._material

    def create_mesh(self, forward=True, controls=None, **kwargs):
        if controls is None:
            controls = dict()

        _props = dict()
        if forward:
            self.apply_forward(_props)
        self._mesh = build_mesh_with_buffer(self.to_mesh_view(), uuid=self.uuid, props=_props, _controls=controls,
                material=self.to_mesh_material(), **kwargs, )
        self._mesh.owner = self

        if hasattr(self, "control_points"):
            self._mesh._controls["path"] = dict(points=encode_control_points(self.control_points)
                    )

    def _init_mesh(self):
        self.create_mesh(forward=False)
        self.update_mesh_material()

    def update_mesh_geometry(self):
        ageomdict[self._mesh._geometry] = create_mesh_buffer_from_mesh_tuple(self.to_mesh_view(), uuid=self._mesh.uuid
                )

    def update_mesh_material(self):
        mat = self.to_mesh_material()

        amatdict[mat.uuid] = mat
        self._mesh._material = mat.uuid

    def update_mesh(self, no_back=False):
        if self._mesh is not None:
            if not no_back:
                self.apply_backward(self._mesh.properties)
            self.hook()
            self.apply_forward(self._mesh.properties)

        else:
            return ValueError("Mesh not initialized")

    def to_mesh(self, **kwargs):
        self.create_mesh() if self._mesh is None else self.update_mesh()
        return self._mesh


class UnionMeshViewSupport(MeshViewSupport):
    def __init_support__(self, *args, material_prototype: BaseMaterial = vertexMaterial, color=(1.0, 1.0, 1.0),
                         **kwargs):
        super().__init_support__(*args, material_prototype=material_prototype, **kwargs)

    def to_mesh_material(self) -> MeshStandardMaterial:
        return MeshStandardMaterial(**{**asdict(self.material_prototype), **dict(color=self.material_color.decimal)})

_group_view_classes = dict()


class GroupViewSupport(ViewSupport):
    __group_class__ = AGroup
    __observer__ = group_observer
    __view_name__ = "group"

    def __class_getitem__(cls, item):
        grp_cls, observer = item
        if _group_view_classes.get((grp_cls, id(observer))) is None:
            _group_view_classes[(grp_cls, id(observer))] = type(
                    f"GroupViewSupport[{item[0].__name__},{observer.__class__.__name__}]", (cls,),
                    dict(__group_class__=grp_cls, __observer__=observer), )
        return _group_view_classes[item]

    def __init_support__(self, uuid=None, field_map=None, items=None):
        super(GroupViewSupport, self).__init_support__(uuid=uuid, field_map=field_map)
        self._group = create_group(self.uuid, obs=self.__observer__, cls=self.__group_class__
                )
        self._items = dict()
        if items is not None:
            self._items |= {item.uuid: item for item in items}

    def update_group(self):
        self._group.children_uuids = self._items.keys()

    def add(self, item: "MeshViewSupport|GroupViewSupport"):
        self._items[item.uuid] = item
        self.update_group()

    def remove(self, item: "MeshViewSupport|GroupViewSupport"):
        del self._items[item.uuid]
        self.update_group()

    def __contains__(self, item: "MeshViewSupport|GroupViewSupport"):
        return self._items.__contains__(item.uuid)

    def update(self, items: "list[MeshViewSupport|GroupViewSupport]"):
        self._items |= {item.uuid: item for item in items}
        self.update_group()

    def index(self, item: "MeshViewSupport|GroupViewSupport"):
        return list(self._items.keys()).index(item)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._items[list(self._items.keys())[item]]
        else:
            return self._items[item]

    def __setitem__(self, item, it):
        if isinstance(item, int):
            self._items[list(self._items.keys())[item]] = it
        else:
            self._items[item] = it
        self.update_group()

