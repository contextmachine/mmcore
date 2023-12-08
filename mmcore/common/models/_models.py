import numpy as np

from mmcore.base import ageomdict
from mmcore.common.models.fields import FieldMap
from mmcore.geom.extrusion import Extrusion, to_mesh
from mmcore.geom.mesh import create_mesh_buffer_from_mesh_tuple
from mmcore.geom.rectangle import Length, Rectangle, to_mesh


class AnalyticModel:
    """
    AnalyticModel - A class representing an analytic model

    Methods:
        __init_subclass__(cls, field_map=())
            Initializes a subclass of AnalyticModel with a field map, a tuple of fields to be sorted in alphabetical order.

        __init__(self)
            Initializes an instance of AnalyticModel.

        apply_forward(self, data)
            Applies the forward method of each field in the field map to the data.

        apply_backward(self, data)
            Applies the backward method of each field in the field map to the data.

        update_mesh(self)
            Updates the mesh by applying the forward method to the mesh properties, creating a mesh buffer, and then applying the backward method to the mesh properties.

    Properties:
        uuid
            The universally unique identifier of the AnalyticModel instance.
    """
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
                     field_map=(FieldMap('u', 'u1'),
                                FieldMap('v', 'v1'),
                                FieldMap('area', 'area', backward_support=False),
                                FieldMap('lock', 'lock'))):
    """
    A class representing a rectangle model.

    Parameters:
    - u: float | Length (optional)
        The length of the rectangle in the u-direction.
    - v: float | Length (optional)
        The length of the rectangle in the v-direction.
    - xaxis: numpy.array (optional)
        The x-axis of the rectangle.
    - normal: numpy.array (optional)
        The normal vector of the rectangle.
    - origin: numpy.array (optional)
        The origin point of the rectangle.

    Attributes:
    - lock: bool
        Specifies if the rectangle is locked.

    Properties:
    - uuid: str
        The unique identifier of the rectangle.

    Methods:
    - to_mesh(uuid=None, **kwargs)
        Converts the rectangle to a mesh representation.

    Example:
    >>> rectangle = RectangleModel(u=2, v=3)
    >>> rectangle.to_mesh()
    """
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
    """
    This class represents a rectangle extrusion model. It is a subclass of the Extrusion and RectangleModel classes.

    Parameters:
    - u: The length of the rectangle along the u-axis. Can be a float or a Length object.
    - v: The length of the rectangle along the v-axis. Can be a float or a Length object.
    - h: The height of the extrusion.
    - xaxis: The x-axis direction of the rectangle. Default is (1, 0, 0).
    - normal: The normal vector of the rectangle. Default is (0, 0, 1).
    - origin: The origin point of the rectangle. Default is (0, 0, 0).

    Examples:
        r = RectangleExtrusionModel(u=2, v=3, h=4)
    """
    def __init__(self, u: 'float|Length' = 1, v: 'float|Length' = 1, h=3.0, xaxis=np.array((1, 0, 0)),
                 normal=np.array((0, 0, 1)),
                 origin=np.array((0, 0, 0))):
        RectangleModel.__init__(self, u, v, xaxis=xaxis, normal=normal, origin=origin)

        super().__init__(self, h, vec=self.normal)
