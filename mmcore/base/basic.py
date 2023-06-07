#
import warnings

import itertools

import functools

import inspect
import os
import pprint
import types
import uuid

import numpy as np
import ujson
from scipy.spatial.distance import euclidean
from strawberry.scalars import JSON

NAMESPACE_MMCOREBASE = uuid.UUID('5901d0eb-61fb-4e8c-8fd3-a7ed8c7b3981')
import dataclasses
import hashlib
import json
import sys
from collections import namedtuple
from mmcore.base.registry import geomdict, objdict, matdict
import operator

import typing

import mmcore.base.models.gql as gql_models

from operator import attrgetter, itemgetter
import uuid as _uuid

import copy

import strawberry

from mmcore.collections.multi_description import Paginate, ElementSequence

Link = namedtuple("Link", ["name", "parent", "child"])
LOG_UUIDS = False

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
            elif isinstance(o, np.int64):
                return int(o)
            else:
                return o

        try:
            return json.JSONEncoder.default(self, o)
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


from mmcore.utils.termtools import ColorStr, TermColors, TermAttrs, MMColorStr


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


parenthashmap = dict()


class UUIDMissPermException(AttributeError):
    ...


from functools import total_ordering

ShaSub = namedtuple("ShaSub", ["int", "hex"])


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
    _uuid = None
    _is_uuid_set = False
    _state = dict()

    @property
    def strawberry_properties_input(self):
        return type(list(self.properties.keys()))

    def __new__(cls, *args, name="Object", uuid=None, pkl=None, **kwargs) -> 'Object3D':

        cls.objdict = objdict
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

                inst.__init__(*args, **kwargs)
                objdict[uuid] = inst
                return inst

        else:
            inst = object.__new__(cls)

            inst._parents = set()
            inst._children = set()
            inst._uuid = _uuid.uuid4().__str__()
            objdict[inst.uuid] = inst

            inst.__init__(*args, name=name, **kwargs)
            return inst

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.__call__(*args, **kwargs, trav=False)

    def __call__(self, *args, trav=True, **kwargs) -> 'Object3D':

        if args:
            kwargs |= dict(zip(self.__match_args__[:-len(args)], args))

        for k, v in kwargs.items():
            setattr(self, k, v)

        if not trav:
            return self
        else:
            return self.get_child_three()

    def __str__(self):
        mm = "[mmcore]: "

        t = " " * (len("[mmcore]: ") + 1)
        aaa = f", \n{t}".join(f'{k}={self.properties[k]}' for k in self.properties.keys())

        return mm + f"{self.__class__.__name__}({aaa}) at {self._uuid}"

    def __repr__(self):

        if int(os.getenv("INRHINO")) == 1:
            mm = "[mmcore]: "

            t = " " * (len("[mmcore]: ") + 1)
            aaa = f", \n{t}".join(f'{k}={self.properties[k]}' for k in self.properties.keys())

            return mm + f"{self.__class__.__name__}({aaa}) at {self._uuid}"

        mm, head = MMColorStr(": "), ColorStr(self.__class__.__name__, color=TermColors.blue,
                                              term_attrs=(TermAttrs.blink, TermAttrs.bold))
        t = " " * (mm.real_len + len(head) + 1)
        aaa = f", \n{t}".join(
            f'{ColorStr(k, color=TermColors.yellow, term_attrs=[TermAttrs.blink, TermAttrs.bold])}={self.properties[k]}'
            for k in self.properties.keys())

        return mm + head + f"({aaa}) at {ColorStr(self._uuid, color=TermColors.cyan, term_attrs=[TermAttrs.blink])}"

    def ToJSON(self, cls=ExEncoder, **kwargs):
        return json.dumps(self.get_child_three(), cls=cls, check_circular=False, **kwargs)

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
        additional = {
            'priority': 1.0,  # Because its code generated object
            "children_count": len(self._children)

        }

        return additional

    @property
    def gui(self):
        return None

    def _add_includes(self, obj):
        if hasattr(obj, "_geometry"):
            if obj._geometry is not None:
                self._include_geometries.add(obj._geometry)
                self._include_materials.add(obj._material)

    def get_child_three(self):
        """
        get_child_three() is a function that takes an object and returns a threejs_root object with all of its children.

        Parameters:
            self (object): The object to be processed.

        Returns:
            threejs_root: A threejs_root object with all the object's children.

        This function takes an object and creates a deep copy of its threejs_repr. It then adds all the object's
        children to the threejs_root object. Finally, it binds the class to the threejs_root object and returns it.
        @return:
        """
        self._include_materials = set()
        self._include_geometries = set()

        def childthree(obj):
            dct = obj.threejs_repr

            self._add_includes(obj)

            if len(obj._children) > 0:

                dct["children"] = []
                for chl in obj._children:
                    try:
                        dct["children"].append(childthree(objdict[chl]))
                    except KeyError:
                        pass
                if 'children' in dct:
                    if len(dct.get('children')) == 0:
                        del dct['children']
                return obj.bind_class(**dct)
            else:
                # #(dct)
                if 'children' in dct:
                    if len(dct.get('children')) == 0:
                        del dct['children']
                return obj.bind_class(**dct)

        return self.threejs_root(childthree(self),
                                 geometries=self._include_geometries,
                                 materials=self._include_materials)

    def _hashdict(self):
        return dict(name=self.name)

    def __hash__(self):

        return int(self.uuid, 16)

    @dataclasses.dataclass
    class Root:
        object: typing.Any
        metadata: gql_models.Metadata
        materials: list[typing.Union[gql_models.Material,
        gql_models.MeshPhongMaterial,
        gql_models.PointsMaterial,
        gql_models.LineBasicMaterial, None]]
        geometries: list[typing.Union[gql_models.BufferGeometry, None]]

    @property
    def _root(self):
        # #(self.Root.__annotations__)
        self.Root.__annotations__['object'] = self.bind_class
        self.Root.__name__ = f"GenericRoot{id(self)}"

        return self.Root

    @property
    def _root_(self):
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

        Root.__name__ = f"GenericRoot{id(self)}"
        return strawberry.type(self._root)

    @property
    def threejs_type(self) -> str:
        return "Object3D"

    def threejs_root(self, dct, geometries=None, materials=None, metadata=None,
                     root_callback=lambda x: x):
        # #(materials, geometries)
        return root_callback(self._root)(object=dct,
                                         materials=[matdict.get(mat) for mat in
                                                    materials] if materials is not None else list(
                                             matdict.values()),
                                         geometries=[geomdict.get(geom) for geom in
                                                     geometries] if geometries is not None else list(
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

    def matrix_to_square_form(self) -> np.ndarray:
        return np.array(self.matrix, dtype=float).reshape((4, 4)).T

    def transform(self, matrix):
        """
        Этот метод применяет трансформацию к уже существующей матрице,
        если вы просто хотите заменить матрицу трансформации используйте `self.matrix = <matrix>`.
        @param matrix:
        @return:
        """
        self.matrix = (
                self.matrix_to_square_form() @ np.array(matrix, dtype=float).reshape((4, 4))).T.flatten().tolist()

    def rotate(self, angle: float, axis: tuple[float, float, float] = (0, 0, 1)):
        """

        @param axis:
        @param angle: radians
        @return:
        """
        q = Quaternion(axis=unit(axis), angle=angle)
        self.transform(q.transformation_matrix)

    def translate(self, vector: tuple[float, float, float]):
        matrix = np.array([[1, 0, 0, vector[0]],
                           [0, 1, 0, vector[1]],
                           [0, 0, 1, vector[2]],
                           [0, 0, 0, 1]], dtype=float)
        self.transform(matrix)

    def reset_transform(self):
        self.matrix = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


class GroupIterator(typing.Iterator):
    def __init__(self, seq: typing.Iterable = ()):
        self._seq = iter(seq)

    def __next__(self):
        return self._seq.__next__()


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

    def __iter__(self):
        return GroupIterator(self._to_list())

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

    def _to_list(self):
        return list(objdict[child] for child in self._children)

    def _to_ptrlist(self):
        return list(self._children)

    def paginate(self):
        return Paginate(self._to_list())

    def paginate_userdata(self):
        try:
            return ElementSequence(
                ElementSequence(
                    ElementSequence(
                        self._to_list())["userData"])["properties"])
        except:
            return dict()

    @property
    def children_keys(self):

        return self.paginate_userdata().keys()

    def __getitem__(self, item):

        return objdict[self._to_ptrlist()[item]]

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
from strawberry.tools.merge_types import merge_types


def to_camel_case(name: str):
    """
    Ключевая особенность, при преобразовании имени начинающиегося с подчеркивания, подчеркивание будет сохранено.

        foo_bar -> FooBar
        _foo_bar -> _FooBar
    @param name: str
    @return: str
    """
    if not name.startswith("_"):

        return "".join(nm[0].capitalize() + nm[1:] for nm in name.split("_"))

    else:
        return "_" + "".join(nm[0].capitalize() + nm[1:] for nm in name.split("_"))


class GenericList(list):
    def __class_getitem__(cls, item):

        _name = "Generic" + cls.__base__.__name__.capitalize() + "[" + item.__name__ + "]"

        def __new__(cls, l):

            if l == []:
                return []
            elif l is None:
                return []
            else:
                ll = []
                for i in l:
                    if i is not None:
                        try:
                            ll.append(item(**i))
                        except TypeError:
                            # #print(item, i)
                            ll.append(item(i))
                return ll

        __ann = typing.Optional[list[item]]

        return type(f'{__ann}', (list,), {"__new__": __new__, "__origin__": list[item]})


ROOT_DOC = """"""


@strawberry.interface(description=ROOT_DOC)
class RootInterface:
    metadata: gql_models.Metadata
    materials: gql_models.AnyMaterial
    geometries: gql_models.BufferGeometry
    object: gql_models.AnyObject3D

    # shapes: typing.Optional[JSON] = None

    @strawberry.field
    def all(self) -> JSON:
        return strawberry.asdict(self)


class DictSchema:
    """
    >>> import strawberry
    >>> from dataclasses import is_dataclass, asdict
    >>> A=Object3D(name="A")
    >>> B = Group(name="B")
    >>> B.add(A)
    >>> dct = strawberry.asdict(B.get_child_three())
    >>> ##print(dct)
    {'object': {'name': 'B', 'uuid': 'bcd5e328-c5e5-4a8f-8381-bb97cb022708', 'userData': {'properties': {'name': 'B', 'children_count': 1}, 'gui': [{'key': 'name', 'id': 'name_chart_linechart_piechart', 'name': 'Name Chart', 'colors': 'default', 'require': ('linechart', 'piechart')}, {'key': 'children_count', 'id': 'children_count_chart_linechart_piechart', 'name': 'Children_count Chart', 'colors': 'default', 'require': ('linechart', 'piechart')}], 'params': None}, 'matrix': [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], 'layers': 1, 'type': 'Group', 'castShadow': True, 'receiveShadow': True, 'children': [{'name': 'A', 'uuid': 'c4864663-67f6-44bb-888a-5f1a1a72e974', 'userData': {'properties': {'name': 'A', 'children_count': 0}, 'gui': None, 'params': None}, 'matrix': [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], 'layers': 1, 'type': 'Object3D', 'castShadow': True, 'receiveShadow': True, 'children': []}]}, 'metadata': {'version': 4.5, 'type': 'Object', 'generator': 'Object3D.toJSON'}, 'materials': [], 'geometries': []}
    >>> ds=DictSchema(dct)
    >>> tp=ds.get_init_default()
    >>> tp.object
    GenericObject(name='B', uuid='bcd5e328-c5e5-4a8f-8381-bb97cb022708',
    userData=GenericUserdata(properties=GenericProperties(name='B', children_count=1),
    gui=[GenericGui(key='name', id='name_chart_linechart_piechart', name='Name Chart', colors='default',
    require=('linechart', 'piechart')), GenericGui(key='children_count', id='children_count_chart_linechart_piechart',
    name='Children_count Chart', colors='default', require=('linechart', 'piechart'))], params=None),
    matrix=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], layers=1, type='Group', castShadow=True, receiveShadow=True,
    children=[GenericChildren(name='A', uuid='c4864663-67f6-44bb-888a-5f1a1a72e974',
    userData=GenericUserdata(properties=GenericProperties(name='A', children_count=0), gui=None, params=None),
    matrix=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], layers=1, type='Object3D', castShadow=True,
    receiveShadow=True, children=())])
    >>> tp.object.children
    [GenericChildren(name='A', uuid='c4864663-67f6-44bb-888a-5f1a1a72e974',
    userData=GenericUserdata(properties=GenericProperties(name='A', children_count=0), gui=None, params=None),
    matrix=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], layers=1, type='Object3D', castShadow=True,
    receiveShadow=True, children=())]
    """

    def bind(self, cls_name: str,
             fields: typing.Iterable,
             *args, **kwargs):
        ncls = dataclasses.make_dataclass(cls_name, fields, *args, **kwargs)
        self.classes.add(ncls)
        return ncls

    def __init__(self, dict_example):
        self.annotations = dict()
        self.cls = object
        self.dict_example = DeepDict(dict_example)
        self.classes = set()

    def generate_schema(self, callback=lambda x: x):
        def wrap(name, obj):
            if obj is None:
                return name, typing.Optional[list[dict]], None
            elif isinstance(obj, dict):
                flds = []
                named_f = dict()
                for k, v in obj.items():
                    fld = wrap(k, v)

                    named_f[k] = fld

                # ##print(name, named_f)
                # ##print("Generic" + to_camel_case(name),

                dcls = callback(self.bind("Generic" + to_camel_case(name),
                                          list(named_f.values())))

                init = copy.deepcopy(dcls.__init__)

                def _init(slf, **kwargs):

                    kws = dict()
                    for nm in named_f.keys():

                        _name, tp, dflt = named_f[nm]

                        # #print(_name, tp, dflt)
                        if nm in kwargs.keys():
                            if isinstance(kwargs[nm], dict):
                                kws[nm] = tp(**kwargs[nm])
                            elif isinstance(kwargs[nm], (GenericList, tuple)):
                                kws[nm] = tp(kwargs[nm])
                            else:
                                try:
                                    kws[nm] = tp(kwargs[nm])
                                except TypeError:
                                    kws[nm] = kwargs[nm]

                        else:
                            kws[nm] = tp(dflt)
                    init(slf, **kws)

                dcls.__init__ = _init
                return name, dcls, lambda: dcls(**obj)
            elif isinstance(obj, list):
                # ##print(name, type(obj), obj)
                *nt, = zip(*[wrap(name, o) for o in obj])
                # #print(nt)
                if len(nt) == 0:
                    return name, tuple, lambda: []
                else:
                    g = GenericList[nt[1][0]]
                    if len(nt) == 3:
                        # #print(g)
                        return name, g, lambda: nt[-1]
                    else:
                        return name, g, lambda: []
            elif obj is None:
                return name, typing.Optional[typing.Any], None
            else:
                return name, type(obj), lambda: obj

        return wrap(self.cls.__name__, self.dict_example)[1]

    @property
    def schema(self):
        return self.generate_schema()

    def get_init_default(self):

        return self.schema(**self.dict_example)

    def get_init_default_strawberry(self):

        return self.get_strawberry()(**self.dict_example)

    def get_strawberry(self):

        return strawberry.type(self.schema)

    def init_partial(self, **kwargs):
        dct = copy.deepcopy(self.dict_example)
        dct |= kwargs
        return self.schema(**dct)

    def _get_new(self, kwargs):
        cp = DeepDict(copy.copy(self.dict_example))
        cp |= kwargs
        inst = self.schema(**cp)
        inst.__schema__ = self.schema
        inst.__classes__ = list(self.classes)
        return inst

    def __call__(self, **kwargs):

        return self._get_new(kwargs)

    def decorate(self, cls):
        self.cls = cls
        return self


class Delegate:
    def __init__(self, delegate):
        self._delegate = delegate

    def __call__(self, owner):
        self._owner = owner

        def _getattr_(inst, item):

            if hasattr(self._owner, item):

                return self._owner.__getattribute__(inst, item)

            else:

                return getattr(inst._ref, item)

        self._owner.__getattr__ = _getattr_
        d = set(dir(self._delegate))
        d1 = set(dir(self._owner))
        d.update(d1)

        def dr(dlf):
            r = object.__dir__(dlf._ref)
            rr = set(object.__dir__(dlf))
            rr.update(set(r))
            return rr

        self._owner.__dir__ = dr

        return self._owner


class DictGqlSchema(DictSchema):
    bind = strawberry.type


class Object3DWithChildren(Object3D):
    bind_class = gql_models.GqlObject3D

    @property
    def threejs_repr(self):
        dct = super().threejs_repr
        dct |= {
            "children": []
        }
        return dct


br = Group(name="base_root")


def iscollection(item):
    return isinstance(item, typing.MutableSequence) or isinstance(item, (tuple, set, frozenset))


objdict["_"] = br
from mmcore.collections import traversal

DEFAULT_MATRIX = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def remove_empty(data, remove_empty_collections=True):
    if isinstance(data, dict):
        d = {}
        for k, v in data.items():
            res = remove_empty(v)
            if res is not None or not (res == []):
                d[k] = res

        return d
    elif iscollection(data):

        if len(data) == [] and remove_empty_collections:
            return None

        else:
            l = []
            for i, v in enumerate(data):
                res = remove_empty(v)
                if res is not None:
                    l.append(res)
            return l
    else:
        return data


def sumdicts(*dicts):
    d = dict(dicts[0])
    for dct in dicts[1:]:
        d |= dct
    return d


from mmcore.base.registry import adict, ageomdict, amatdict, idict
from mmcore.geom.vectors import unit
from pyquaternion import Quaternion
from mmcore.gql.client import GQLReducedQuery


class GQLPropertyDescriptor:
    def __init__(self, query, mutate=None, vars=None):
        self._query = GQLReducedQuery(query)
        self._mutate = GQLReducedQuery(mutate)
        self._vars = vars if vars is not None else set()

    def __set_name__(self, owner, name):
        self.name = name
        owner.properties_keys.add(name)

    def __get__(self, inst, owner):
        if inst:
            variables = dict()
            if len(self._vars) > 0:

                for v in self._vars:
                    variables[v] = getattr(inst, v)
            return self._query(variables=variables)
        else:
            return self

    def __set__(self, inst, v):

        variables = dict()
        if len(self._vars) > 0:

            for k in self._vars:
                variables[k] = getattr(inst, k)
        variables |= v
        self._mutate(variables=variables)


class A:
    idict = idict
    args_keys = ["name"]
    _uuid: str = "no-uuid"
    name: str = "A"
    _state_keys = {
        "uuid",
        "name",
        "matrix"
    }
    priority = 1.0

    properties_keys = {
        "priority",
        "name"

    }
    _matrix = list(DEFAULT_MATRIX)
    _include_geometries = GeometrySet()
    _include_materials = MaterialSet()
    _properties = dict()

    def __copy__(self):

        dct = {}
        for k in dir(self):
            if not k.startswith("_"):
                if not (k in ["properties", "adict", 'threejs_type']):
                    dct[k] = getattr(self, k)
        dct["uuid"] = _uuid.uuid4().hex
        obj = self.__class__.__new__(self.__class__, **dct)

        return obj

    def traverse_child_three(self):
        """
        get_child_three() is a function that takes an object and returns a threejs_root object with all of its children.

        Parameters:
            self (object): The object to be processed.

        Returns:
            threejs_root: A threejs_root object with all of the object's children.

        This function takes an object and creates a deep copy of its threejs_repr. It then adds all of the object's children to the threejs_root object. Finally, it binds the class to the threejs_root object and returns it.
        @return:
        """
        self._include_materials = MaterialSet()
        self._include_geometries = GeometrySet()

        def childthree(obj):
            dct = dict()
            self._add_includes(obj)
            if len(obj.children) > 0:
                dct["children"] = []
                for chl in obj.children:
                    try:
                        dct["children"].append(childthree(chl))
                    except KeyError:
                        pass
                return dct
            else:
                # #print(dct)
                if 'children' in dct:
                    if len(dct.get('children')) == 0:
                        del dct['children']
                return dct

        return childthree(self)

    def property_interface(self):
        return

    @property
    def state_keys(self):
        return set(list(self.args_keys + list(self.child_keys) + list(self._state_keys)))

    @state_keys.setter
    def state_keys(self, v):
        self._state_keys = set(v)

    @property
    def properties(self):

        return self._properties

    @properties.setter
    def properties(self, v):

        self._properties |= v

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, v):
        self._matrix = v

    @matrix.deleter
    def matrix(self):
        self._matrix = list(DEFAULT_MATRIX)

    def gql_input_type(self):
        ds = DictSchema(self.properties)
        ds.get_init_default_strawberry()

    @property
    def threejs_type(self):
        return "Object3D"

    @property
    def uuid(self) -> str:
        return self._uuid

    @uuid.setter
    def uuid(self, v):
        try:

            val = adict[self._uuid]
            adict[str(v)] = val
            del adict[self._uuid]
            self._uuid = str(v)
        except KeyError:
            self._uuid = str(v)
            adict[self._uuid] = self

    @classmethod
    def get_object(cls, uuid: str):
        return adict[uuid]

    @property
    def children(self):
        l = []
        for child in self._children:
            try:
                l.append(adict[child])
            except KeyError as err:

                pass
        return l

    def __new__(cls, *args, **kwargs):
        inst = object.__new__(cls)

        inst.child_keys = set()
        inst._children = set()

        inst.set_state(*args, **kwargs)
        if inst.uuid == "no-uuid":
            inst.uuid = uuid.uuid4().hex
        adict[inst.uuid] = inst

        return inst

    def render(self):
        l = []
        for w in self.children:
            l.append(w())
            self._add_includes(w)

    def get_state(self):
        dct = {}
        for k in self.state_keys:
            val = self.__getattr__(k)
            if isinstance(val, A):
                dct[k] = val.get_state()
            else:
                dct[k] = val
        return dct

    def set_state(self, *args, **kwargs):
        _kwargs = dict(zip(self.args_keys, args[:len(self.args_keys)]))
        _kwargs |= kwargs

        for k, v in _kwargs.items():
            self.__setattr__(k, v)
        self.traverse_child_three()


    @children.setter
    def children(self, v):
        for child in v:
            self._children.add(child.uuid)

    def root(self, shapes=None):
        self.traverse_child_three()
        if shapes:

            return {
                "metadata": {
                    "version": 4.5,
                    "type": "Object",
                    "generator": "Object3D.toJSON"
                },
                "object": self(),
                "shapes": shapes,
                "geometries": [strawberry.asdict(ageomdict[uid]) for uid in self._include_geometries],
                "materials": [strawberry.asdict(amatdict[uid]) for uid in self._include_materials]
            }
        else:
            return {
                "metadata": {
                    "version": 4.5,
                    "type": "Object",
                    "generator": "Object3D.toJSON"
                },
                "object": self(),
                "geometries": [strawberry.asdict(ageomdict[uid]) for uid in self._include_geometries],
                "materials": [strawberry.asdict(amatdict[uid]) for uid in self._include_materials]
            }

    def __call__(self, *args, callback=lambda x: x, **kwargs):
        self.set_state(*args, **kwargs)
        return callback(
            {
                "name": self.name,
                "uuid": self.uuid,
                "children": [ch() for ch in self.children],
                "type": self.threejs_type,
                "layers": 1,
                "matrix": self.matrix,
                "castShadow": True,
                "receiveShadow": True,
                "userData": {

                    "properties": sumdicts({
                        "name": self.name
                    },
                        self.properties,
                    )
                }
            }
        )

    _parents = set()

    def __getattr__(self, key):
        try:
            if (key, self.uuid) in idict.keys():

                return adict[idict[(key, self.uuid)]]
            else:
                return getattr(self, key)
        except RecursionError as err:
            return self.__getattribute__(key)

    def set_state_attr(self, key, value):
        self.__setattr__(key, value)
        self._state_keys.add(key)

    def __setattr__(self, key, v):

        if isinstance(v, A):
            if (key, self.uuid) in idict:
                idict[(key, self.uuid)] = v.uuid
            else:
                self.child_keys.add(key)
                self._children.add(v.uuid)
                idict[(key, self.uuid)] = v.uuid

        else:

            object.__setattr__(self, key, v)

    def __delattr__(self, key):
        if (key, self.uuid) in idict:
            del idict[(key, self.uuid)]
            self.child_keys.remove(key)

        else:
            super().__delattr__(key)

    def _add_includes(self, obj):
        if hasattr(obj, "_geometry"):
            if obj._geometry is not None:
                self._include_geometries.add(obj._geometry)
                self._include_materials.add(obj._material)

    def matrix_to_square_form(self) -> np.ndarray:
        return np.array(self.matrix, dtype=float).reshape((4, 4)).T

    def transform(self, matrix):
        """
        Этот метод применяет трансформацию к уже существующей матрице,
        если вы просто хотите заменить матрицу трансформации используйте `self.matrix = <matrix>`.
        @param matrix:
        @return:
        """
        self.matrix = (
                self.matrix_to_square_form() @ np.array(matrix, dtype=float).reshape((4, 4))).T.flatten().tolist()

    def rotate(self, angle: float, axis: tuple[float, float, float] = (0, 0, 1)):
        """

        @param axis:
        @param angle: radians
        @return:
        """
        q = Quaternion(axis=unit(axis), angle=angle)
        self.transform(q.transformation_matrix)

    def translate(self, vector: tuple[float, float, float]):
        matrix = np.array([[1, 0, 0, vector[0]],
                           [0, 1, 0, vector[1]],
                           [0, 0, 1, vector[2]],
                           [0, 0, 0, 1]], dtype=float)
        self.transform(matrix)

    def scale(self, x: float = 1, y: float = 1, z: float = 1):
        matrix = np.array([[x, 0, 0, 0],
                           [0, y, 0, 0],
                           [0, 0, z, 0],
                           [0, 0, 0, 1]], dtype=float)
        self.transform(matrix)

    def reset_transform(self):
        self.matrix = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

    def dispose(self):

        for k, v in idict.items():
            uid, name = k
            if v == self.uuid:
                parent = self.get_object(uid)
                parent._children.remove(self.uuid)
                parent.__delattr__(name)
                idict.__delitem__((uid, name))

        adict.__delitem__(self.uuid)
        del self

    def dump(self, filename):
        with open(filename, "w") as f:
            ujson.dump(self.root(), f)
class AGroup(A):

    @property
    def threejs_type(self):
        return "Group"

    def add(self, obj):
        self._children.add(obj.uuid)

    def root(self, shapes=None):
        if shapes:

            return {
                "metadata": {
                    "version": 4.5,
                    "type": "Object",
                    "generator": "Object3D.toJSON"
                },
                "object": self(),
                "shapes": shapes,
                "geometries": [strawberry.asdict(ageomdict[uid]) for uid in self._include_geometries],
                "materials": [strawberry.asdict(amatdict[uid]) for uid in self._include_materials]
            }
        else:
            return {
                "metadata": {
                    "version": 4.5,
                    "type": "Object",
                    "generator": "Object3D.toJSON"
                },
                "object": self(),
                "geometries": [strawberry.asdict(ageomdict.get(uid)) for uid in self._include_geometries],
                "materials": [strawberry.asdict(amatdict.get(uid)) for uid in self._include_materials]
            }


class RootForm(A):
    def __call__(self, res=None, *args, **kwargs):
        _ = A.__call__(self, *args, **kwargs)
        return {
            "metadata": {
                "version": 4.5,
                "type": "Object",
                "generator": "Object3D.toJSON"
            },
            "object": res,
            "geometries": list(self._include_geometries),
            "materials": list(self._include_materials)
        }


class AGeometryDescriptor:
    adict = dict()

    def __init__(self, default=None):
        super().__init__()
        if default is not None:
            ageomdict[default.uuid] = default
            self._default = default.uuid
        else:
            self._default = default

    def __set_name__(self, owner, name):
        if not (name == "geometry"):
            raise
        self._name = "_" + name

    def __get__(self, instance, owner):
        if instance is None:

            return ageomdict.get(self._default)
        else:
            return ageomdict.get(getattr(instance, self._name))

    def __set__(self, instance, value):
        ageomdict[value.uuid] = value
        setattr(instance, self._name, value.uuid)


from mmcore.geom.materials import ColorRGB


class AMaterialDescriptor:
    adict = dict()

    def __init__(self, default=None):
        super().__init__()
        if default is not None:
            amatdict[default.uuid] = default
            self._default = default.uuid
        else:
            self._default = default

    def __set_name__(self, owner, name):
        if not (name == "material"):
            raise
        self._name = "_" + name

    def __get__(self, instance, owner):
        if instance is None:

            return amatdict.get(self._default)
        else:

            return amatdict[getattr(instance, self._name)]

    def __set__(self, instance, value):

        amatdict[value.uuid] = value
        setattr(instance, self._name, value.uuid)


class AGeom(A):
    material_type = gql_models.Material
    geometry = AGeometryDescriptor(default=None)
    material = AMaterialDescriptor(default=None)

    @property
    def threejs_type(self):
        return "Geometry"

    def __call__(self, *args, **kwargs):
        res = super().__call__(*args, **kwargs)
        res |= {
            "geometry": self.geometry.uuid if self.geometry else None,
            "material": self._material if self._material else None,
        }
        return res


class AMesh(AGeom):
    material_type = gql_models.MeshPhongMaterial
    geometry = AGeometryDescriptor(default=None)
    material = AMaterialDescriptor(default=gql_models.MeshPhongMaterial(color=ColorRGB(120, 200, 40).decimal))

    @property
    def threejs_type(self):
        return "Mesh"


def position_hash(points):
    return hashlib.sha1(ujson.dumps(np.array(points, dtype=float).flatten().tolist()).encode()).hexdigest()


class APointsGeometryDescriptor(AGeometryDescriptor):
    def __get__(self, instance, owner):
        if instance is None:
            return self._default
        else:
            if hasattr(instance, self._name):
                return gql_models.BufferGeometryObject(**{
                    'data': gql_models.Data1(
                        **{'attributes': gql_models.Attributes1(
                            **{'position': gql_models.Position(
                                **{'itemSize': 3,
                                   'type': 'Float32Array',
                                   'array': np.array(
                                       instance.points).flatten().tolist(),
                                   'normalized': False})})})})
            else:
                return None

    def __set__(self, instance, value):

        instance.points = value

        uid = position_hash(value)
        setattr(instance, self._name, uid)
        ageomdict[uid] = gql_models.BufferGeometryObject(**{
            'data': gql_models.Data1(
                **{'attributes': gql_models.Attributes1(
                    **{'position': gql_models.Position(
                        **{'itemSize': 3,
                           'type': 'Float32Array',
                           'array': np.array(value).flatten().tolist(),
                           'normalized': False})})})})


class APoints(AGeom):
    material_type = gql_models.PointsMaterial
    geometry = APointsGeometryDescriptor(default=None)

    material = AMaterialDescriptor(default=gql_models.PointsMaterial(color=ColorRGB(120, 200, 40).decimal))
    _material = None
    _points = []
    kd = None

    def solve_kd(self):
        self.kd = KDTree(self.points)
        return self.kd

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, v):
        self._points = v

    @property
    def threejs_type(self):
        return "Points"


from scipy.spatial import KDTree


class APoint(APoints):

    def __new__(cls, x, y, z, *args, **kwargs):
        return super().__new__(points=[x, y, z], *args, **kwargs)

    @property
    def x(self):
        return self.points[0]

    @property
    def y(self):
        return self.points[1]

    @property
    def z(self):
        return self.points[2]

    @z.setter
    def z(self, v):
        self.points[2] = v

    @y.setter
    def y(self, v):
        self.points[1] = v

    @x.setter
    def x(self, v):
        self.points[0] = v

    def distance(self, other: 'APoint'):
        return euclidean(self.points, other.points)


class ALine(APoints):
    material_type = gql_models.LineBasicMaterial
    geometry = APointsGeometryDescriptor(default=None)
    material = AMaterialDescriptor(
        default=gql_models.LineBasicMaterial(color=ColorRGB(120, 200, 40).decimal, uuid="line-12020040"))

    @property
    def start(self):
        return self.points[0]

    @start.setter
    def start(self, value):
        self.points[0] = value

    @property
    def end(self):
        return self.points[-1]

    @end.setter
    def end(self, value):
        self.points[-1] = value

    @property
    def threejs_type(self):
        return "Line"


grp = AGroup(name="base_root", uuid="_")


class TestException(Exception): ...


def generate_object3d_dict(**kwargs):
    dct = {
        "uuid": uuid.uuid4().hex,
        "type": "Object3D",
        "layers": 1,
        "matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        "up": [0, 1, 0]
    }


def deep_merge(dct, dct2):
    for k, v in dct.items():
        vv = dct2.get(k)
        if vv is not None:
            if isinstance(vv, dict):
                if isinstance(v, dict):
                    deep_merge(v, vv)
                else:
                    dct[k] = vv
            else:
                dct[k] = vv
    for k, v2 in dct2.items():
        if not (k in dct.keys()):
            dct[k] = v2
    return dct


class DeepDict(dict):
    def __ior__(self, other):
        deep_merge(self, other)
        return self


class ObjectThree:
    def __init__(self, root: A):
        self.root = root

    def get_obj(self):
        return adict[self.root]

    def all(self):
        return adict[self.root].root()

    def walk(self):
        three = {}
        obj = self.get_obj()
        if len(obj.child_keys) > 0:
            for k in obj.idict.keys():
                name, key = k
                if key == obj.uuid:
                    three[name] = self.__class__(obj.idict[(name, key)])
        return three

    def __getitem__(self, item):
        if item == "all":
            return self.all()
        else:
            return self.walk()[item]

    def keys(self):
        return self.walk().keys()

    def to_dict(self):

        dct = {"all": self.all()}
        for k, v in self.walk().items():
            if isinstance(v, ObjectThree):
                dct[k] = v.to_dict()
        return dct


class GqlObjectThree(ObjectThree):

    def __getitem__(self, item):
        if item != "all":

            return super().__getitem__(item)
        else:
            return super().__getitem__(item)


class Three:
    all: JSON


def new_three(origin: GqlObjectThree = None):
    attrs = dict(itertools.zip_longest(origin.keys(), ['GenericThree'], fillvalue='GenericThree'))
    define = f"""
    
@strawberry.type
class GenericThree:
    __annotations__ = {attrs}
    
    @property
    def origin(self):
        return self._origin
      
    @strawberry.field
    def all(self) -> JSON:
        return self.origin.all()
        
"""
    for k in origin.keys():
        define += f"""   
    @strawberry.field 
    def {k}(self)->'GenericThree':
        return new_three(self.origin["{k}"])"""

    cd = compile(define, "_i", "exec")

    exec(cd, globals(), locals())
    e = eval('GenericThree')()
    e._origin = origin
    return e
