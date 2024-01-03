import abc
import dataclasses
import struct
import typing
import uuid as _uuid

from steputils import p21

from steputils.p21 import DataSection, ParameterList, Entity, SimpleEntityInstance,Keyword, Reference
from string import ascii_uppercase

from mmcore.base import AGroup, ParamGraphNode, TermParamGraphNode
from mmcore.collections import OrderedSet
from mmcore.geom.parametric import PlaneLinear
from mmcore.geom.point import ControlPoint

refs = {}

stepdct = dict()

typenodes = dict()

stepdict={"$":None}
def step_to_nodes(data):
    for k, l in data.instances.items():
        if hasattr(l, "entity"):
            nested = STEPNestedEntity(type=l.entity.name, params=list(l.entity.params))
            if not (l.entity.name in typenodes.keys()):
                typenodes[l.entity.name] = []
            typenodes[l.entity.name].append((k,nested))

            stepdct[k] = STEPNode(ref=k, children=[nested])

        else:
            nitm = []
            for i in l.entities:
                nitm.append(STEPNestedEntity(i.name, i.params))
                if not (i.name in typenodes.keys()):
                    typenodes[i.name] = []
                typenodes[i.name].append(i)
            stepdct[k] = STEPNode(ref=k, children=nitm)
    for ref, params in typenodes["CARTESIAN_POINT"]:
        StepCPt(ref, params=params.params, type=params.type)
    for ref, params in typenodes["DIRECTION"]:
        StepDirection(ref, params=params.params, type=params.type)
    for ref, params in typenodes["AXIS2_PLACEMENT_3D"]:
        StepAxisPlace3D(ref, params=params.params, type=params.type)
    for ref, params in typenodes["VERTEX_POINT"]:
        StepVertexPoint(ref, params=params.params, type=params.type)
    for ref, params in typenodes["CIRCLE"]:
        StepCircle(ref, params=params.params, type=params.type)
def step_refcheck(params, callback=lambda x: x.todict()):
    l = []
    for p in params:
        if isinstance(p, str):
            if p.startswith("#"):

                obj = stepdct[p]
                l.append(callback(obj))
            elif p == "NONE":
                l.append(None)
            elif p == ".T.":
                l.append(True)
            elif p == ".F.":
                l.append(False)
            else:
                l.append(p)
        elif isinstance(p, (tuple, list, p21.ParameterList)):
            if len(p) == 0:
                l.append(p)
            else:
                l.append(step_refcheck(p, callback=callback))

        else:
            l.append(p)
    return l


@dataclasses.dataclass
class STEPNestedEntity:
    type: str
    params: list

    def todict(self):
        return {
            "type": self.type,
            "params": step_refcheck(self.params)
        }

    def typemap(self):
        return {
            "type": self.type,
            "params": step_refcheck(self.params, callback=lambda x: x.typemap())
        }

    def convert(self, uuid_prefix=""):
        return {
            "type": self.type,
            "params": step_refcheck(self.params, callback=lambda x: x.convert(uuid_prefix=uuid_prefix))
        }

    def tostep(self):
        return Entity(name=Keyword(self.type), params=ParameterList(self.params))
@dataclasses.dataclass
class STEPNode:
    ref: str
    children: list[STEPNestedEntity]
    uuid_prefix = ""
    def getchilds(self):
        l=[]
        def _get(obj):
            l.append(obj.ref)

            for child in obj.children:

                for prm in child.params:
                    print(prm)
                    if isinstance(prm, str):
                        if str.startswith(prm, "#"):
                            _get(stepdct[prm])
        _get(self)
        return l

    def todict(self):
        return {
            "ref": self.ref,
            "children": [child.todict() for child in self.children]

        }


    def typemap(self):
        return [child.typemap() for child in self.children]

    def __getitem__(self, item):
        return stepdct[item]

    def convert(self, uuid_prefix=""):
        grp = AGroup(uuid=uuid_prefix+self.ref)
        for child in self.children:
            grp.add(child.convert(uuid_prefix=uuid_prefix))
        return grp
    def tostep(self):
        return SimpleEntityInstance(ref=self.ref, entity=self.children[0].tostep())

class _StepGeometryBase:
    ref: dataclasses.InitVar[str]
    params: dataclasses.InitVar[list]
    type: dataclasses.InitVar[str]
    def __init__(self, ref, params, type):
        self.uuid = ref
        stepdict[ref] = self
        self.step_type = type
        self.name = params[0]
        self._params=params
        self.unpack_params(params[1:])

    @abc.abstractmethod
    def unpack_params(self, params):
        ...


class StepGeometryEntity(_StepGeometryBase):
    ref: dataclasses.InitVar[str]
    params: dataclasses.InitVar[list]
    type: dataclasses.InitVar[str]

    def __init__(self, ref, params, type):
        super().__init__(ref, params, type)

    @abc.abstractmethod
    def unpack_params(self, params):...

    @abc.abstractmethod
    def todict(self, params): ...

    def to_step_ir(self):
        return STEPNode(ref=self.uuid,children=[STEPNestedEntity(params=self.params, type=self.step_type)])



    @property
    def params(self)->list:
        return self._params

    @property
    @abc.abstractmethod
    def resolver(self) -> typing.Any:
        ...

    def native(self):
        return ParamGraphNode(_params= self.__dict__,
        uuid = self.uuid,
        name = self.name,
        resolver = self
        )
    def __call__(self, **kwargs):
        for k,v in kwargs.items():
            if v is not None:
                setattr(self,k,v)

        return self


    def __repr__(self):
        return f'{self.__class__.__name__}(uuid={self.uuid}, name={self.name})'



class StepCPt(StepGeometryEntity):
    ref: dataclasses.InitVar[str]
    params: dataclasses.InitVar[list]
    type: dataclasses.InitVar[str]
    x: float
    y: float
    z: float
    step_type: str ="CARTESIAN_POINT"
    name: typing.Optional[str] = None


    def unpack_params(self, params):

        (self.x, self.y, self.z),=params

    def convert(self):
        return self.x, self.y, self.z

    def native(self):
        return ControlPoint(uuid=self.uuid, name=self.name, x=self.x,y=self.y, z=self.z)
    @property
    def params(self) ->list:
        return [self.name, [self.x, self.y, self.z]]


    def __repr__(self):
        return f'{self.__class__.__name__}(uuid={self.uuid}, name={self.name}, x={self.x}, y={self.y}, z={self.z})'



class StepDirection(StepGeometryEntity):

    x: float
    y: float
    z: float
    step_type: str = "DIRECTION"
    name: typing.Optional[str] = None
    def convert(self):
        return self.x, self.y, self.z
    def native(self):
        return TermParamGraphNode([self.x,self.y, self.z], uuid=self.uuid,name=self.name)
    def unpack_params(self, params):
        (self.x, self.y, self.z), = params
    @property
    def resolver(self):
        return self.x, self.y, self.z
    @property
    def params(self):
       return [self.name, [self.x, self.y, self.z]]
    def todict(self, params):
        return self.params
    def __repr__(self):
        return f'{self.__class__.__name__}(uuid={self.uuid}, name={self.name}, x={self.x}, y={self.y}, z={self.z})'

class StepAxisPlace3D(StepGeometryEntity):


    step_type: str = "AXIS2_PLACEMENT_3D"
    name: typing.Optional[str] = None

    @property
    def params(self):
        return [self.name, [self._origin, self._xaxis, self._yaxis]]

    def unpack_params(self, params):
        self._origin, self._xaxis, self._yaxis = params

    def native(self):
        return ParamGraphNode(dict(origin=self.origin.native(),xaxis=self.origin.native(),zaxis=self.origin.native()), uuid=self.uuid,name=self.name, resolver=self)
    def convert(self):
        if self.yaxis is None:
            return PlaneLinear(origin=self.origin.convert(), xaxis=[1.0,0.0,0.0], normal=self.xaxis.convert())
        return PlaneLinear(origin=self.origin.convert(), normal=self.xaxis.convert(), xaxis=self.yaxis.convert())

    @property
    def xaxis(self):
        return stepdict[self._xaxis]

    @xaxis.setter
    def xaxis(self, v):
        stepdict[self._xaxis](**v)
    def resolver(self, **kwargs):
        self.__dict__|=kwargs
        return self

    @property
    def yaxis(self):
        return stepdict[self._yaxis]

    @yaxis.setter
    def yaxis(self, v):
        stepdict[self._yaxis](**v)
    @property
    def origin(self):
        return stepdict[self._origin]

    @origin.setter
    def origin(self, v):
        stepdict[self._origin](**v)
    def __repr__(self):
        return f'{self.__class__.__name__}(uuid={self.uuid}, name={self.name}, origin={self.origin}, xaxis={self.xaxis}, yaxis={self.yaxis})'

from mmcore.geom.parametric import Circle

class StepVertexPoint(StepGeometryEntity):
    ref: dataclasses.InitVar[str]
    params: dataclasses.InitVar[list]
    type: dataclasses.InitVar[str]
    step_type: str = "VERTEX_POINT"
    def unpack_params(self, params):
        self._point,=params
    @property
    def point(self):
        return stepdict[self._point]
    def native(self):
        return self.point.native()
    def convert(self):
        return stepdict[self._point].convert()


    def __repr__(self):
        return  f'{self.__class__.__name__}(uuid={self.uuid}, name={self.name}, point={self.point})'


class StepCircle(StepGeometryEntity):
    step_type: str = "CIRCLE"
    def unpack_params(self, params):
        self._plane,self.radius=params

    def native(self):
        return ParamGraphNode(dict(radius=self.radius, plane=self.plane.native()))
    def convert(self):
        return Circle(self.radius, self.plane.convert())
    def __repr__(self):
        return  f'{self.__class__.__name__}(uuid={self.uuid}, name={self.name}, radius={self.radius}, plane={self.plane})'

    @property
    def plane(self):
        return stepdict[self._plane]

    @plane.setter
    def plane(self, v):
        stepdict[self._plane](**v)
