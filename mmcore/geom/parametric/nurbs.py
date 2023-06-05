import copy
import dataclasses
import typing
import warnings
from abc import ABCMeta, abstractmethod

import geomdl
import numpy as np
from geomdl import utilities as geomdl_utils, NURBS
from geomdl.operations import tangent
from more_itertools import flatten

from mmcore.base import AMesh, AGeometryDescriptor, APointsGeometryDescriptor, ALine, AGeom, \
    APoints, ageomdict, AGroup
from mmcore.base.geom import MeshData
from mmcore.base.models.gql import BufferGeometryObject, LineBasicMaterial, PointsMaterial, Data1, \
    Attributes1, Position
from mmcore.base.models.pydantic import BufferGeometry
from mmcore.geom.parametric.base import NormalPoint, ProxyAttributeDescriptor, ProxyParametricObject
from mmcore.collections import ElementSequence, DoublyLinkedList
from mmcore.geom.materials import ColorRGB

from scipy import optimize


@dataclasses.dataclass(eq=True, unsafe_hash=True)
class LoftOptions:
    degree: int = 1

    def asdict(self):
        return dataclasses.asdict(self)


@dataclasses.dataclass
class NurbsCurve(ProxyParametricObject):
    """

    """

    @property
    def __params__(self) -> typing.Any:
        return self.closest_point, self.degree, self.domain, self.knots, self.delta

    control_points: typing.Iterable[typing.Iterable[float]] = ProxyAttributeDescriptor(proxy_name="ctrlpts")
    delta: float = 0.01
    degree: int = 3
    dimension: int = ProxyAttributeDescriptor(default=2)
    rational: bool = ProxyAttributeDescriptor()
    domain: typing.Optional[tuple[float, float]] = ProxyAttributeDescriptor()

    bbox: typing.Optional[list[list[float]]] = ProxyAttributeDescriptor()
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
    cpts: dataclasses.InitVar[typing.Iterable[typing.Iterable[float]]]

    degree_u: int = 3
    degree_v: int = 3,
    delta: float = 0.025

    dimension: int = ProxyAttributeDescriptor(default=3)

    @property
    def proxy(self):
        try:
            return self._proxy
        except AttributeError:
            self.prepare_proxy()
            return self._proxy

    def __post_init__(self, cpts):

        self.prepare_proxy()

        self.proxy._control_points_size[0] = np.array(cpts).shape[0]
        self.proxy._control_points_size[1] = np.array(cpts).shape[1]

        self.proxy.degree_u, self.proxy.degree_v = self.degree

        *fl, = np.array(cpts).reshape((self.proxy._control_points_size[0] * self.proxy._control_points_size[1], 3),
                                      order="C").tolist()

        self.proxy.ctrlpts = fl
        self.proxy.delta = self.delta

        self.proxy.knotvector_u = geomdl_utils.generate_knot_vector(self._proxy.degree[0], self.proxy.ctrlpts_size_u)
        self.proxy.knotvector_v = geomdl_utils.generate_knot_vector(self._proxy.degree[1], self.proxy.ctrlpts_size_v)
        self.control_points = cpts

        # u,v=self.size_u, self.size_v

    @property
    def degree(self):
        return self.degree_u, self.degree_v

    def evaluate(self, t):
        if len(np.array(t).shape) > 2:

            t = np.array(t).tolist()

            return np.asarray(self._proxy.evaluate_list(t))
        else:
            return np.asarray(self._proxy.evaluate_single(t))

    def prepare_proxy(self):

        self._proxy = NURBS.Surface()

    def normal(self, t):
        return geomdl.operations.normal(self.proxy, t)

    def tangent(self, t):
        pt, tn = tangent(self.proxy, t)
        return NormalPoint(*pt, normal=tn)

    @property
    def mesh_data(self) -> MeshData:
        # self.proxy.tessellate()
        vertseq = ElementSequence(self._proxy.vertices)
        faceseq = ElementSequence(self._proxy.faces)
        uv = np.round(np.asarray(vertseq["uv"]), decimals=5)

        normals = [pv[1] for pv in geomdl.operations.normal(self._proxy, uv.tolist())]
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

    @property
    def dim(self) -> int:
        return 2


class ProxyGeometryDescriptor(AGeometryDescriptor):
    __proxy_dict__ = dict()

    def __init__(self, default=None):
        super().__init__(default=default)

    def __get__(self, instance, owner):
        return super().__get__(instance, owner)

    def __set__(self, instance, value):
        geom = self.solve_proxy(instance, value)
        ageomdict[geom.uuid] = geom
        instance._geometry = geom.uuid

    def __set_name__(self, owner, name):
        super().__set_name__(owner, name)
        self._ref_name = "_ref" + self._name
        owner.__ref__ = property(fget=lambda slf: self,
                                 doc="Access to reference parametric NURBS Surface object. Read only")
        setattr(owner, self.refname, property(fget=lambda slf: self.proxy_namespace[str(id(slf))],
                                              doc="Access to reference parametric NURBS Surface object. Read only"))

        if not (name in self.__proxy_dict__.keys()):
            self.__proxy_dict__[name] = dict()
        owner._ref_extra = dict()
        owner.__ref_extra__ = property(fget=lambda slf: slf._ref_extra,
                                       fset=lambda slf, v: slf._ref_extra.update(v),
                                       doc="Dict with extra NURBS parameters.")

    @abstractmethod
    def solve_geometry(self, instance):
        ...

    @property
    def refname(self):
        return f'__ref{self._ref_name}'

    @abstractmethod
    def solve_proxy(self, instance, control_points):
        pass

    @property
    def proxy_namespace(self):
        return self.__proxy_dict__[self._name[1:]]


class NurbsProxyDescriptor(ProxyGeometryDescriptor):
    def __set_name__(self, owner, name):
        super().__set_name__(owner, name)
        owner.control_points = property(fget=lambda slf: self.proxy_namespace.get(str(id(slf)))._ref.cntrlpts,

                                        doc="NURBS Surface control points.")

    def solve_proxy(self, instance, control_points):
        arr = np.array(control_points)
        su, sv, b = arr.shape
        degu = su - 1 if su >= 4 else 3
        degv = sv - 1 if sv >= 4 else 3

        self.__proxy_dict__[self._name[1:]][str(id(instance))] = NurbsSurface(
            control_points=arr.flatten().tolist(),
            degree_u=degu,
            degree_v=degv,
            **instance.__ref_extra__)
        return self.solve_geometry(instance)

    def solve_geometry(self, instance) -> BufferGeometryObject:
        return getattr(instance, self.refname).tessellate()


class NurbsCurveProxyDescriptor(NurbsProxyDescriptor, APointsGeometryDescriptor):
    def solve_proxy(self, instance, control_points):
        self.__proxy_dict__[self._name[1:]][str(id(instance))] = NurbsCurve(control_points, **instance.__ref_extra__)
        return self.solve_geometry(instance)


class NurbsSurfaceProxyDescriptor(NurbsProxyDescriptor):
    def solve_proxy(self, instance, value):
        arr = np.array(value)
        su, sv, b = arr.shape
        print(su, sv)
        degu = su - 1
        degv = sv + 1 if sv <= 2 else 3

        self.__proxy_dict__[self._name[1:]][str(id(instance))] = NurbsSurface(
            control_points=arr.tolist(),
            degree_u=degu,
            degree_v=degv,

            **instance.__ref_extra__)

        return self.solve_geometry(instance)


class NurbsLoftProxyDescriptor(NurbsSurfaceProxyDescriptor):

    def solve_proxy(self, instance, value):
        self.__proxy_dict__[self._name[1:]][str(id(instance))] = self.solve_loft(instance, value)
        return self.solve_geometry(instance)

    def solve_loft(self, instance, value):
        crvspt = value.pop("control_points")
        instance.__ref_extra__ |= value
        crvs = []
        for crv in crvspt:
            crvs.append(NurbsCurve(crv))

        lsp = np.linspace(*np.array(ElementSequence(crvs)["control_points"]), 3)
        prms = copy.deepcopy(instance.__ref_extra__)
        ud, vd = prms.pop("degree")

        return NurbsSurface(lsp, degree_u=ud, degree_v=vd, **prms)


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

    def __new__(cls, *args, control_points=None, color=ColorRGB(0, 255, 40).decimal, ref_extra={}, **kwargs):
        inst = super().__new__(cls, *args, geometry=control_points, material=cls.material_type(color=color), **kwargs)
        inst.__ref_extra__ = ref_extra
        return inst

    def __call__(self, control_points=None, **kwargs):
        if control_points is not None:
            self.geometry = control_points

        return super().__call__(**kwargs)

    def helpers_geometry(self):
        ...
    def include_helpers(self):
        self.helpers_geometry()
        self._children.add(self.uuid + "-helpers")

class NurbsCurveGeometry(ProxyGeometry, ALine):
    geometry = NurbsCurveProxyDescriptor()

    def helpers_geometry(self):
        cpts = self.__ref_ref_geometry.control_points
        return ALine(name="NurbsCurveControlPoints", geometry=cpts, uuid=self.uuid + "-helpers")


class NurbsSurfaceGeometry(ProxyGeometry, AMesh):
    geometry = NurbsSurfaceProxyDescriptor()




class NurbsLoft(ProxyGeometry, AMesh):
    geometry = NurbsLoftProxyDescriptor()