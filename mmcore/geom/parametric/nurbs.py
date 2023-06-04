import dataclasses
import typing
import warnings
from abc import ABCMeta, abstractmethod

import geomdl
import numpy as np
from geomdl import utilities as geomdl_utils, NURBS
from geomdl.operations import tangent

from mmcore.base import AMesh, AGeometryDescriptor, APointsGeometryDescriptor, ALine, AGeom, \
    APoints
from mmcore.base.geom import MeshData
from mmcore.base.models.gql import BufferGeometryObject, LineBasicMaterial, PointsMaterial, Data1, \
    Attributes1, Position

from mmcore.geom.parametric.base import NormalPoint, ProxyDescriptor, ProxyParametricObject
from mmcore.collections import ElementSequence
from mmcore.geom.materials import ColorRGB

from scipy import optimize


@dataclasses.dataclass
class NurbsCurve(ProxyParametricObject):
    """

    """

    @property
    def __params__(self) -> typing.Any:
        return self.closest_point, self.degree, self.domain, self.knots, self.delta

    control_points: typing.Iterable[typing.Iterable[float]] = ProxyDescriptor(proxy_name="ctrlpts")
    delta: float = 0.01
    degree: int = 3
    dimension: int = ProxyDescriptor(default=2)
    rational: bool = ProxyDescriptor()
    domain: typing.Optional[tuple[float, float]] = ProxyDescriptor()

    bbox: typing.Optional[list[list[float]]] = ProxyDescriptor()
    knots: typing.Optional[list[float]] = None
    slices: int = 32

    def __post_init__(self):

        self.proxy.degree = self.degree

        self.proxy.ctrlpts = self.control_points

        if self.knots is None:
            self.knots = geomdl_utils.generate_knot_vector(self._proxy.degree, len(self._proxy.ctrlpts))
            self.proxy.knotvector = self.knots
        self.proxy.delta = self.delta
        self.proxy.evaluate()

    def resolve(self):
        self.proxy.degree = self.degree
        self.proxy.ctrlpts = self.control_points
        self.knots = geomdl_utils.generate_knot_vector(self._proxy.degree, len(self._proxy.ctrlpts))
        self.proxy.knotvector = self.knots
        self.proxy.delta = self.delta


    def prepare_proxy(self):

        self._proxy = NURBS.Curve()

        self._proxy.degree = self.degree

    def evaluate(self, t):
        if hasattr(t, "__len__"):
            if hasattr(t, "tolist"):
                t = t.tolist()
            else:
                t = list(t)
            return np.asarray(self._proxy.evaluate_list(t))
        else:
            return np.asarray(self._proxy.evaluate_single(t))

    def tan(self, t):
        pt = tangent(self.proxy, t)
        return NormalPoint(*pt)

    def tessellate(self):
        return BufferGeometryObject(**{
            'data': Data1(
                **{'attributes': Attributes1(
                    **{'position': Position(
                        **{'itemSize': 3,
                           'type': 'Float32Array',
                           'array': np.array(
                               [self.evaluate(t) for t in np.linspace(0, 1, self.slices)]).flatten().tolist(),
                           'normalized': False})})})})


@dataclasses.dataclass
class NurbsSurface(ProxyParametricObject):
    _proxy = NURBS.Surface()
    control_points: typing.Iterable[typing.Iterable[float]]
    degree: tuple = ProxyDescriptor(proxy_name="degree", default=(3, 3))
    delta: float = 0.025
    degree_u: int = ProxyDescriptor(proxy_name="degree_u", default=3)
    degree_v: int = ProxyDescriptor(proxy_name="degree_v", default=3)
    size_u: int = ProxyDescriptor(proxy_name="ctrlpts_size_u", default=6)
    size_v: int = ProxyDescriptor(proxy_name="ctrlpts_size_v", default=6)
    dimentions: int = ProxyDescriptor(proxy_name="dimentions", default=3)
    knots_u: typing.Optional[list[list[float]]] = ProxyDescriptor(proxy_name="knotvector_u", no_set=True)
    knots_v: typing.Optional[list[list[float]]] = ProxyDescriptor(proxy_name="knotvector_v", no_set=True)
    knots: typing.Optional[list[list[float]]] = ProxyDescriptor(proxy_name="knotvector", no_set=True)
    domain: typing.Optional[list[list[float]]] = ProxyDescriptor(proxy_name="domain", no_set=True)
    trims: tuple = ()

    @property
    def proxy(self):
        try:
            return self._proxy
        except AttributeError:
            self.prepare_proxy()
            return self._proxy

    def __post_init__(self):

        self.proxy.ctrlpts = self.control_points

        self.proxy.knotvector_u = geomdl_utils.generate_knot_vector(self._proxy.degree[0], self.size_u)
        self.proxy.knotvector_v = geomdl_utils.generate_knot_vector(self._proxy.degree[1], self.size_v)
        self.proxy.delta = self.delta
        self.proxy.evaluate()

        # u,v=self.size_u, self.size_v

    def evaluate(self, t):
        if len(np.array(t).shape) > 2:

            t = np.array(t).tolist()

            return np.asarray(self._proxy.evaluate_list(t))
        else:
            return np.asarray(self._proxy.evaluate_single(t))

    def prepare_proxy(self):

        self._proxy.set_ctrlpts(list(self.control_points), (self.size_u, self.size_v))
        # self._proxy.ctrlpts=self.control_points
        # ##print(self)

        self._proxy.degree_u, self._proxy.degree_v = self.degree

    def normal(self, t):
        return geomdl.operations.normal(self.proxy, t)

    def tangent(self, t):
        pt, tn = tangent(self.proxy, t)
        return NormalPoint(*pt, normal=tn)

    @property
    def mesh_data(self) -> MeshData:
        self.proxy.tessellate()
        vertseq = ElementSequence(self._proxy.vertices)
        faceseq = ElementSequence(self._proxy.faces)
        uv = np.round(np.asarray(vertseq["uv"]), decimals=5)
        normals = [v for p, v in geomdl.operations.normal(self._proxy, uv.tolist())]
        return MeshData(vertices=np.array(vertseq['data'], dtype=float),
                        normals=np.array(normals, dtype=float),
                        indices=np.array(faceseq["vertex_ids"],
                                         dtype=float),
                        uv=uv)

    def tessellate(self) -> BufferGeometryObject:
        return self.mesh_data.create_buffer()

    @property
    def __params__(self) -> typing.Any:
        return self.closest_point, self.degree, self.domain, self.knots, self.delta


class ProxyGeometryDescriptor(AGeometryDescriptor, metaclass=ABCMeta):
    __proxy_dict__ = dict()

    def __init__(self, default=None):
        super().__init__(default=default)

    def __get__(self, instance, owner):
        return super().__get__(instance, owner)

    def __set__(self, instance, value):
        super().__set__(instance, self.solve_proxy(instance, value))

    def __set_name__(self, owner, name):
        super().__set_name__(owner, name)
        owner.__ref__ = property(fget=lambda slf: self,
                                 doc="Access to reference parametric NURBS Surface object. Read only")
        if not (name in self.__proxy_dict__.keys()):
            self.__proxy_dict__[name] = dict()
        owner._ref_extra = dict()
        owner.__ref_extra__ = property(fget=lambda slf: slf._ref_extra,
                                       fset=lambda slf, v: slf._ref_extra.update(v),
                                       doc="Dict with extra NURBS parameters.")

    @abstractmethod
    def solve_geometry(self, instance):
        pass

    @abstractmethod
    def solve_proxy(self, instance, control_points):
        pass

    @property
    def proxy_namespace(self):
        return self.__proxy_dict__[self._name[1:]]


class NurbsProxyDescriptor(ProxyGeometryDescriptor):
    def __set_name__(self, owner, name):
        super().__set_name__(owner, name)
        owner.control_points = property(fget=lambda slf: self.proxy_namespace.get(str(id(slf))).control_points,
                                        doc="NURBS Surface control points.")

    def solve_proxy(self, instance, control_points):
        ...
        arr = np.array(control_points)
        su, sv, b = arr.shape
        degu = su - 1 if su >= 4 else 3
        degv = sv - 1 if sv >= 4 else 3

        self.__proxy_dict__[self._name[1:]][str(id(instance))] = NurbsSurface(
            control_points=arr.reshape((su * sv, b)).tolist(),
            size_u=su,
            size_v=sv,
            degree_u=degu,
            degree_v=degv,
            **instance.__ref_extra__)
        return self.solve_geometry(instance)

    def solve_geometry(self, instance) -> BufferGeometryObject:
        return self.proxy_namespace[str(id(instance))].tessellate().create_buffer()


class NurbsCurveProxyDescriptor(NurbsProxyDescriptor, APointsGeometryDescriptor):
    def solve_proxy(self, instance, control_points):
        self.__proxy_dict__[self._name[1:]][str(id(instance))] = NurbsCurve(control_points, **instance.__ref_extra__)
        return self.solve_geometry(instance)


class NurbsSurfaceProxyDescriptor(NurbsProxyDescriptor):
    def solve_proxy(self, instance, control_points):
        arr = np.array(control_points)
        su, sv, b = arr.shape
        degu = su - 1 if su >= 4 else 3
        degv = sv - 1 if sv >= 4 else 3

        self.__proxy_dict__[self._name[1:]][str(id(instance))] = NurbsSurface(
            control_points=arr.reshape((su * sv, b)).tolist(),
            size_u=su,
            size_v=sv,
            degree_u=degu,
            degree_v=degv,
            **instance.__ref_extra__)
        return self.solve_geometry(instance)


# material=MeshPhongMaterial(color=color.decimal)
class ProxyGeometry(AGeom):
    color = ColorRGB(125, 125, 125).decimal
    geometry = NurbsSurfaceProxyDescriptor()
    receiveShadow: bool = True
    castShadow: bool = True
    __GEOMETRY_WARNING__: bool = True

    @property
    def proxy(self):
        return self.__ref__

    def __new__(cls, control_points=None, *args, color=ColorRGB(0, 255, 40), ref_extra={}, **kwargs):
        if kwargs.get("geometry") and (control_points is not None):
            raise Exception("You can not pass control_points and geometry simultaneously!")
        elif (kwargs.get("geometry") is None) and (control_points is not None):

            inst = super().__new__(cls, geometry=control_points, *args, **kwargs)
            inst.__ref_extra__ = ref_extra

            inst.geometry = control_points
            return inst
        elif (kwargs.get("geometry") is not None) and (control_points is None):
            warnings.warn("ProxyGeometry is used for computable geometry types that take parameters as arguments. "
                          "\nIf you want to pass BufferGeometry at once, "
                          "use the base classes AGeom, APoints, ALine. "
                          "\n\nTo remove this warning, "
                          "set the ProxyGeometry.__GEOMETRY_WARNING__ attribute to False.")
            if isinstance(kwargs.get("material"), PointsMaterial):
                return APoints(**kwargs)
            elif isinstance(kwargs.get("material"), LineBasicMaterial):
                return ALine(**kwargs)

            else:
                return AMesh(**kwargs)

    def __call__(self, control_points=None, **kwargs):
        if control_points is not None:
            self.geometry = control_points

        if kwargs.get('material') is None:
            if kwargs.get('color') is not None:
                self.color = kwargs.get('color').decimal
                self.material = self.material_type(color=kwargs.get('color'))

        return super().__call__(**kwargs)


class NurbsCurveGeometry(ProxyGeometry, ALine):
    geometry = NurbsCurveProxyDescriptor()


class NurbsSurfaceGeometry(ProxyGeometry, AMesh):
    geometry = NurbsSurfaceProxyDescriptor()
