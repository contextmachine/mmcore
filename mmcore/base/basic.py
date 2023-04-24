#
import dataclasses
import json
import sys

from collections import namedtuple

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from mmcore.base.registry import geomdict, objdict, matdict
import operator

import typing

import mmcore.base.models.gql as gql_models

from operator import attrgetter, itemgetter
import uuid as _uuid

import copy

import strawberry
from mmcore.services.redis import connect
from mmcore.collections.multi_description import Paginate, ElementSequence

from mmcore.collections import ParamContainer

Link = namedtuple("Link", ["name", "parent", "child"])


@dataclasses.dataclass
class Link:
    parent: str
    child: str
    name: str

    def __post_init__(self):
        ...


@dataclasses.dataclass
class UuidBind:
    value: typing.Any
    uuid: typing.Optional[str] = None

    def __post_init__(self):
        if self.uuid is None:
            self.uuid = _uuid.uuid4()


class Orig:
    def __init__(self, cls):
        super().__init__()
        self.cls = cls

    def get_class(self):
        return self.cls

    def set_class(self, cls):
        self.cls = cls

    def __getitem__(self, item):
        ...

    def __setitem__(self, item):
        ...

    def state(self):
        ...

    def getsetter(self, obj):
        target = self

        class GetSetBind:
            def __getitem__(self, item):
                ...

            def __setitem__(self, k, item):
                target.__getitem__(k, item)
                if target.cls is not None:
                    if isinstance(item, target.cls):
                        itm = UuidBind(uuid=item.uuid, value=item)
                    else:
                        itm = UuidBind(value=item)
                    Link(k, obj.uuid, itm.uuid)
                setattr(obj, "_key_" + k, k)


class RegistryDescriptor:
    class_origin: type

    def __init__(self, origin: type[Orig]):
        super().__init__()
        self._orig = origin

        self.origin = None

    def __set_name__(self, owner, name):
        self.origin = self._orig(owner)
        self.class_origin = owner

        self._name = "_" + name

    def __get__(self, inst, own):
        if inst is None:
            return self.origin.state()
        else:
            return self.origin.getsetter(inst)


import pickle


class ExEncoder(json.JSONEncoder):
    def default(self, o):
        def df(o):

            if isinstance(o, set):
                return list(o)
            elif hasattr(o, "_type_definition"):
                return strawberry.asdict(o)
            elif dataclasses.is_dataclass(o):
                return df(dataclasses.asdict(o))
            else:
                return o

        try:
            return super().default(o)
        except TypeError:
            return df(o)


def hasitemattr(attr):
    def wrp(obj):
        if isinstance(obj, dict):
            return attr in obj.keys()
        else:
            if hasattr(obj, attr):
                a = getattr(obj, attr)
                if a is not None:

                    if hasattr(a, "__len__") and not isinstance(a, str):
                        if len(a) == 0:
                            return False
                        return True
                    return True
                return True
            return False

    return wrp


def graph_from_json(data):
    a = []
    for d in data:
        cls = globals()["type"]
        a.append(cls(uuid=d["uuid"], dump_dict=d))
    return a


from mmcore.base.registry import objdict

objdict = objdict


class GeometrySet(set):

    def __contains__(self, item):
        # TODO: Это нужно переписать на np.allclose по буфферу, т.e. проверять идентичность геометрии честно
        uuids = ElementSequence(list(self))["uuid"]

        return item.uuid in uuids


class MaterialSet(set):
    def __contains__(self, item):
        colors = ElementSequence(list(self))["color"]
        # TODO: Это нужно расширить и проверять не только по цвету,
        #  для этого нужно реально понять какие параметры важные
        return item.color in colors


class Object3D:
    """
    >>> obj2 = Object3D("test2")
    >>> obj3 = Object3D("test3")
    >>> obj = Object3D("test")
    >>> obj.detail_a = obj3
    >>> obj3.profile = obj2
    >>> obj.detail_b = obj2
    """
    __match_args__ = ()
    _matrix = None
    _include_geometries = GeometrySet()
    _include_materials = MaterialSet()
    bind_class = gql_models.GqlObject3D
    _name: str = "Object"
    @property
    def strawberry_properties_input(self):
        return type(list(self.properties.keys()))
    def __new__(cls, *args, name="Object", uuid=None, pkl=None, **kwargs) -> 'Object3D':

        cls.objdoct = objdict
        if pkl:
            obj = pickle.loads(pkl)
            objdict[obj.uuid] = obj
            return obj
        if uuid is not None:
            try:
                return objdict[uuid](*args, **kwargs)
            except KeyError:
                inst = object.__new__(cls)
                inst._uuid = uuid
                kw = kwargs.get("dump_dict")
                if kw.get("properties") is not None:
                    inst.__dict__ |= kw.get("properties")

                inst._children = set(kw["children"])
                inst._parents = set(kw["parents"])
                if "geometry" in kw.keys():
                    inst._geometry = kw["geometry"]
                    inst._material = kw["material"]
                objdict[uuid] = inst

                return inst

        else:
            inst = object.__new__(cls)

            inst._parents = set()
            inst._children = set()
            inst.uuid = _uuid.uuid4()
            objdict[inst.uuid] = inst
            return inst.__call__(*args, **kwargs)

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.__call__(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> 'Object3D':
        if args:
            kwargs |= dict(zip(self.__match_args__[:-len(args)], args))
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, children: {len(self.children)}) at {self.uuid}"

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, v: str):
        self._name = v

    @property
    def matrix(self) -> list[float]:
        if self._matrix is None:
            return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        else:
            return self._matrix

    @matrix.setter
    def matrix(self, v):
        if not (len(v) == 16):
            raise ValueError
        self._matrix = v

    @property
    def children_count(self):
        return len(self._children)

    @property
    def uuid(self):
        return self._uuid

    @uuid.setter
    def uuid(self, v):
        self._uuid = v.__str__()

    @property
    def parents_getter(self):
        if len(self._parents) > 0:
            return itemgetter(*tuple(self._parents))
        else:
            return lambda x: []

    def __set_name__(self, owner, name):
        owner._children.add(self.uuid)
        self._parents.add(owner.uuid)
        owner.__dict__[name] = self.uuid
        self.name = name

    def __get__(self, instance, owner):
        return objdict[instance.__dict__[self.name]]

    def __set__(self, instance, v):
        instance.__dict__[self.name] = v.uuid

    @property
    def child_getter(self):
        if len(self._children) > 0:
            return itemgetter(*tuple(self._children))
        else:
            return lambda x: []

    def __setattr__(self, key, value):
        if isinstance(value, Object3D):
            object.__setattr__(self, key, value.uuid)
            self._children.add(value.uuid)
            value._parents.add(self.uuid)
        else:
            object.__setattr__(self, key, value)

    @property
    def parents(self):
        res = self.parents_getter(objdict)
        if not isinstance(res, (tuple, list)):
            return [res]
        return res

    @property
    def children(self):

        res = self.child_getter(objdict)
        if not isinstance(res, (tuple, list)):
            return [res]
        return res

    @property
    def userData(self):
        return {
            "properties": self.properties,
            "gui": self.gui
        }

    @property
    def properties(self):
        return {
            "name": self.name,
            "children_count": len(self._children)

        }

    @property
    def gui(self):
        return None

    def _add_includes(self, obj):
        if hasattr(obj, "_geometry"):
            if obj._geometry is not None:
                self._include_geometries.add(obj._geometry)
                self._include_materials.add(obj._material)

    def get_child_three(self):
        self._include_materials = set()
        self._include_geometries = set()

        def childthree(obj):
            dct = copy.deepcopy(obj.threejs_repr)

            self._add_includes(obj)

            if len(obj._children) > 0:

                dct["children"] = []
                for chl in obj._children:
                    dct["children"].append(childthree(objdict[chl]))
                if 'children' in dct:
                    if len(dct.get('children')) == 0:
                        del dct['children']
                return obj.bind_class(**dct)
            else:
                print(dct)
                if 'children' in dct:
                    if len(dct.get('children')) == 0:
                        del dct['children']
                return obj.bind_class(**dct)

        return self.threejs_root(childthree(self),
                                 geometries=self._include_geometries,
                                 materials=self._include_materials)

    @property
    def threejs_type(self):
        return self.__class__.__name__

    @property
    def root(self):
        target = self

        class Root:
            object: target.bind_class
            metadata: gql_models.Metadata
            materials: list[typing.Union[gql_models.Material,
            gql_models.MeshPhongMaterial,
            gql_models.PointsMaterial,
            gql_models.LineBasicMaterial,
            None]]

            geometries: list[typing.Union[gql_models.BufferGeometry, None]]

        return strawberry.type(Root)

    @property
    def threejs_type(self):
        return "Object3D"

    def threejs_root(self, dct, geometries=None, materials=None, metadata=None):
        print(materials, geometries)
        return self.root(object=dct,
                         materials=[matdict[mat] for mat in materials] if materials is not None else list(
                             matdict.values()),
                         geometries=[geomdict[geom] for geom in geometries] if geometries is not None else list(
                             geomdict.values()),
                         metadata=metadata if metadata is not None else gql_models.Metadata()
                         )

    @property
    def threejs_repr(self):
        return dict(name=self.name,
                    uuid=self.uuid,
                    userData=gql_models.GqlUserData(**self.userData),
                    matrix=list(self.matrix),
                    layers=1,
                    type=self.threejs_type,
                    castShadow=True,
                    receiveShadow=True

                    )

    @classmethod
    def from_three(cls, obj, *args, **kwargs):
        cls2 = cls.eval_type(obj.get("type"))
        inst = cls2(name=obj.get("name"))
        inst._uuid = obj.get('uuid')

        if obj["userData"].get("properties") is not None:
            inst.__dict__ |= obj["userData"].get("properties")
        for k in inst.threejs_repr.keys():
            inst.__setattr__(k, obj.get(k))

        objdict[obj.get('uuid')] = inst
        return inst

    @staticmethod
    def eval_type(typ):
        try:
            return eval(str(typ))
        except NameError:
            return getattr(sys.modules["mmcore.base.sketch"], str(typ), __default=Object3D)

    def __eq__(self, other):
        return self.uuid == other.uuid

    def __ne__(self, other):
        return self.uuid != other.uuid

    @properties.setter
    def properties(self, v):
        for k, v in v.items():
            self.__setattr__("_" + k, v)

    def strawberry_properties(self, input):
        self.properties = strawberry.asdict(input)

class Group(Object3D):
    _name: str = "Group"
    chart_class = gql_models.GqlChart
    bind_class = gql_models.GqlGroup

    @property
    def threejs_type(self):
        return "Group"

    def __init__(self, children=None, **kwargs):

        super().__init__(**kwargs)
        if children:
            self.update(children)

    def difference(self, other):
        return Group([objdict[ch] for ch in self._children.difference(list(other._children))])

    def symmetric_difference(self, other):
        return Group([objdict[ch] for ch in self._children.symmetric_difference(list(other._children))])

    @property
    def properties(self):
        return {
            "name": self.name,
            "children_count": len(self._children)

        }



    def __len__(self):
        return len(self._children)

    @property
    def children_count(self):
        return len(self)

    def add(self, item):
        self._children.add(item.uuid)
        item._parents.add(self.uuid)

    def update(self, items):
        for item in items:
            self._children.add(item.uuid)
            item._parents.add(self.uuid)

    def to_list(self):
        return list(objdict[child] for child in self._children)

    def paginate(self):
        return Paginate(self.to_list())

    def paginate_userdata(self):
        try:
            return ElementSequence(ElementSequence(ElementSequence(self.to_list())["userData"])["properties"])
        except:
            return dict()

    @property
    def children_keys(self):

        return self.paginate_userdata().keys()

    def __len__(self):
        return self._children.__len__()

    @property
    def threejs_repr(self):
        return dict(name=self.name,
                    uuid=self.uuid,
                    userData=gql_models.GqlUserData(**self.userData),
                    children=[],
                    matrix=list(self.matrix),
                    type=self.threejs_type)

    @property
    def gui(self) -> list[chart_class]:
        return [self.chart_class(key=key) for key in self.children_keys]

    @classmethod
    def from_three(cls, obj, *args, **kwargs):
        def traverse(ob):

            cls2 = cls.eval_type(ob.get("type"))
            inst = cls2.from_three(obj)
            lst = []
            if "children" in ob.keys():
                for child in ob["children"]:
                    inst._children.add(child["uuid"])
                    lst.append(traverse(child))
                return lst
            else:
                return inst

        return traverse(obj)


def gen_rows(attrs=("uuid", "name", "_children", "_parents", "matrix"), return_dict=True):
    getter = attrgetter(*attrs)
    for k in objdict.keys():
        if return_dict:
            yield dict(zip(attrs, getter(objdict[k])))
        else:
            yield getter(objdict[k])


def getattr_(obj):
    baseic = 'uuid', '_children', '_parents', 'properties'
    psw = 'uuid', 'children', 'parents', 'properties'
    getter = operator.attrgetter(*baseic)
    dct = dict(zip(psw, getter(obj)))
    if hasattr(obj, '_geometry'):
        dct['geometry'] = obj._geometry
        dct['material'] = obj._material
    return dct


qschema = """
    


type Metadata {
  generator: String
  type: String
  version: Float
}

type Normal {
  array: [Float]
  itemSize: Int
  normalized: Boolean
  type: String
}
type Uv {
  array: [Float]
  itemSize: Int
  normalized: Boolean
  type: String
}
type Position {
  array: [Float]
  itemSize: Int
  normalized: Boolean
  type: String
}
type Attributes {
  normal: Normal
  position : Position
  uv: Uv
}


type BoundingSphere {
  center: [Float]
  radius: Float
}

type Data {
  attributes: Attributes
  boundingSphere: BoundingSphere
  index: Index
}

type Geometries {
  data: Data
  type: String
  uuid: String
}

type Index {
  array: [Int]
  type: String
}

type Materials {
  color: Int
  colorWrite: Boolean
  depthFunc: Int
  depthTest: Boolean
  depthWrite: Boolean
  emissive: Int
  flatShading: Boolean
  reflectivity: Float
  refractionRatio: Float
  shininess: Int
  side: Int
  specular: Int
  stencilFail: Int
  stencilFunc: Int
  stencilFuncMask: Int
  stencilRef: Int
  stencilWrite: Boolean
  stencilWriteMask: Int
  stencilZFail: Int
  stencilZPass: Int
  type: String
  uuid: String
}


type GeometryObject {
  castShadow: Boolean
  geometry: String
  layers: Int
  material: String
  matrix: [Int]
  name: String
  receiveShadow: Boolean
  type: String
  up: [Int]
  uuid: String
  
}
type Object {
  castShadow: Boolean
  layers: Int
  matrix: [Int]
  name: String
  receiveShadow: Boolean 
  type: String
  up: [Int]
  uuid: String
  children: [Object]
  
}
type Group {
  castShadow: Boolean
  layers: Int
  matrix: [Int]
  name: String
  receiveShadow: Boolean
  type: String
  up: [Int]
  uuid: String
  
}

type SampleOutput {
  geometries: [Geometries]
  materials: [Materials]
  metadata: Metadata
  object: Object
}



"""


def to_camel_case(name: str):
    """
    Ключевая особенность, при преобразовании имени начинающиегося с подчеркивания, подчеркивание будет сохранено.

        foo_bar -> FooBar
        _foo_bar -> _FooBar
    @param name: str
    @return: str
    """
    if not name.startswith("_"):
        return "".join(nm.capitalize() for nm in name.split("_"))

    else:
        return "_" + "".join(nm.capitalize() for nm in name.split("_"))


class DictSchema:
    bind = dataclasses.make_dataclass

    def __init__(self, dict_example):
        self.annotations = dict()

        self.dict_example = dict_example

    def generate_schema(self):
        def wrap(name, obj):
            if isinstance(obj, dict):
                flds = []
                named_f = dict()
                for k, v in obj.items():
                    fld = wrap(k, v)
                    flds.append(flds)
                    named_f[k] = fld
                print(name, named_f)
                dcls = self.bind("Generic" + to_camel_case(name),
                                 named_f.values())

                init = copy.deepcopy(dcls.__init__)

                def _init(slf, **kwargs):

                    kws = dict()
                    for nm in named_f.keys():

                        _name, tp, dflt = named_f[nm]
                        if nm in kwargs.keys():
                            if isinstance(kwargs[nm], dict):
                                kws[nm] = tp(**kwargs[nm])

                            else:
                                kws[nm] = tp(kwargs[nm])

                        else:
                            kws[nm] = tp(dflt)
                    init(slf, **kws)

                dcls.__init__ = _init
                return name, dcls, lambda: dcls(**obj)

            else:
                print(name, type(obj), obj)
                return name, type(obj), lambda: obj

        return wrap("root", self.dict_example)


from strawberry.tools import create_type


class DictGqlSchema(DictSchema):
    bind = create_type


class Object3DWithChildren(Object3D):
    bind_class = gql_models.GqlObject3D

    @property
    def threejs_repr(self):
        dct = super().threejs_repr
        dct |= {
            "children": []
        }
        return dct
