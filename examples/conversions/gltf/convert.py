import functools
import functools
import json
from collections import Counter
from functools import reduce
from uuid import uuid4

import requests
import time
from itertools import count

from mmcore.compat.gltf.components import (
    GLTFAccessor,
    GLTFBuffer,
    GLTFBufferView,
    GLTFDocument,
    GLTFMaterial,
    GLTFNode,
    GLTFPbrMetallicRoughness,
    GLTFScene,
)
from mmcore.compat.gltf.utils import appendBufferView, byte_stride


def fromnp(val):
    if isinstance(val, np.ndarray):
        return val.tolist()
    else:
        return list(val)


from mmcore.base.geom import MeshData


def sum_md(md, md2):
    ixs1, ixs2 = np.array(md.indices), np.array(md2.indices)
    ixs3 = ixs1.tolist() + (ixs2 + np.max(ixs1 + 1)).tolist()
    dct = dict(indices=ixs3)
    for attr in ["vertices", "normals", "uv"]:
        v1, v2 = getattr(md, attr), getattr(md2, attr)
        if all([v1 is not None, v2 is not None]):
            dct[attr] = fromnp(v1) + fromnp(v2)
    return MeshData(**dct)


def sum_mds(*mds):
    return functools.reduce(sum_md, mds)


class GLTFColor:
    def __init__(self, r=120, g=120, b=120, a=255):
        self._data = np.array([r, g, b, a]) * (1 / 255)

    def __iter__(self):
        return iter(self._data.tolist())


DEFAULT_MATERIAL = GLTFMaterial(
    name="mmcore_default",
    doubleSided=True,
    pbrMetallicRoughness=GLTFPbrMetallicRoughness(baseColorFactor=tuple(GLTFColor())),
)

views = []
accessors = []
buffers = []
materials = [DEFAULT_MATERIAL]
buffers_to_views = dict()
views_to_accessors = dict()
nodes = []
from mmcore.compat.gltf.consts import *

from typing import Any, TypedDict

from mmcore.collections.basic import IndexOrderedSet

call_count_dict = dict()


def call_counter(fun):
    call_count = count()
    call_count_dict[fun] = next(call_count)

    @functools.wraps(fun)
    def wrapper(*args, **kwargs):
        nonlocal call_count

        call_count_dict[fun] = next(call_count)

        return fun(*args, **kwargs)

    return wrapper


component_registry = dict()
component_instance_stack = []
component_registry_counters = dict()
component_extras_map = dict()


def from_rgba(cls, r=0.4, g=0.4, b=0.4, a=1.0):
    return cls(
        name=f"simple-mat-{hex(int(r * 255))}-{hex(int(g * 255))}-{hex(int(b * 255))}-{hex(int(a * 100))}",
        doubleSided=True,
        pbrMetallicRoughness=GLTFPbrMetallicRoughness(baseColorFactor=(r, g, b, a)),
    )


def random(cls, random_opacity=False):
    res = np.random.random(4)
    if not random_opacity:
        res[-1] = 1.0

    return from_rgba(cls, *res)


def component(key=None):
    def wrapper(cls):
        nonlocal key
        if key is None:
            key = cls.__name__.lower()
        cls.__component_key__ = key
        component_registry[key] = []
        component_registry_counters[key] = count()
        cls.global_registry = property(
            fget=lambda self: component_registry[self.__class__.__component_key__],
            fset=lambda self, v: component_registry.__setitem__(
                self.__class__.__component_key__, v
            ),
        )
        if not hasattr(cls, "__hash__"):
            cls.__hash__ = lambda slf: id(slf)

        @functools.wraps(cls)
        def initwrapper(*args, name=None, **kwargs):
            if name is None:
                name = uuid4().hex

            self = cls(*args, **kwargs)

            component_registry[key].append(self)
            self.name = name
            self._ixs = next(component_registry_counters[key])

            self.global_index = self._ixs

            return self

        initwrapper.wrapped_cls = cls
        return initwrapper

    return wrapper


class HashableDict(dict):
    def __hash__(self):
        return id(self)


# @functools.lru_cache(None)
def relative_index(comp, index_map=None):
    if not index_map:
        return comp.global_index

    return index_map[comp.__component_key__][comp._ixs]


def enm(lst):
    cnt = count()
    for i in lst:
        yield i, next(cnt)


def asscene(node: "SceneNode", name=None, buffer=None, **kwargs):
    if buffer is None:
        buffer = bytearray()
    s = time.time()
    _deps = node.deps()
    #
    # print(index_map)
    print("\tsolve deps", divmod(time.time() - s, 60))
    local_registry = dict.fromkeys(component_registry.keys())
    s = time.time()
    # print(_deps)
    _deps["bufferViews"] = []
    local_registry["bufferViews"] = []
    for vv in component_registry["bufferViews"]:
        if not vv.isempty:
            _deps["bufferViews"].append(vv.global_index)
            local_registry["bufferViews"].append(vv.pack(buffer))

    print("\tcreate buffer views", divmod(time.time() - s, 60))
    s = time.time()
    index_map = {k: dict(enm(v)) for k, v in _deps.items()}
    print("\tcreate index_map", divmod(time.time() - s, 60))
    s = time.time()
    for k, v in index_map.items():
        s1 = time.time()
        if k != "bufferViews":
            local_registry[k] = [
                component_registry[k][i].togltf(index_map=index_map) for i in v
            ]
        print(f"\t\t{k} step", divmod(time.time() - s1, 60))
    print("\tmain loop", divmod(time.time() - s, 60))
    s = time.time()
    if name is None:
        name = f"{node.name}-scene"

    local_registry["buffers"] = [GLTFBuffer.from_bytes(buffer)]
    local_registry["scenes"] = [GLTFScene(nodes=[0], name=name, **kwargs)]
    _doc = GLTFDocument(**local_registry)
    print("\tcreating doc", divmod(time.time() - s, 60))
    return _doc


@component(key="nodes")
class SceneNode:
    def __init__(self, children=None, mesh=None, name=None, matrix=None, extras=None):
        self.children = []
        if children:
            self.children.extend(children)
        self.mesh = mesh
        self.name = name
        self.matrix = matrix
        self.extras = extras

    @call_counter
    def deps(self):
        mesh = getattr(self, "mesh", None)

        dps = dict(nodes=[self.global_index])
        if mesh:
            dps |= dict(
                merge_list_values(dict(meshes=[mesh.global_index]), mesh.deps())
            )

        return merge_dict_list_values_sequence(
            [dps] + [child.deps() for child in self.children]
        )

    def add_child(self, child):
        if child not in self.children:
            self.children.append(child)
        return self.children.index(child)

    def remove_child(self, child):
        self.children.remove(child)

    def tocomponent(self):
        ...

    def togltf(self, index_map=None):
        return GLTFNode(
            children=[
                relative_index(child, index_map=index_map) for child in self.children
            ],
            mesh=None
            if self.mesh is None
            else relative_index(self.mesh, index_map=index_map),
            name=self.name,
            matrix=self.matrix,
            extras=self.extras,
        )

    def child_tree(self):
        dct = [self.global_index]
        for child in self.children:
            dct.extend(child.child_tree())
        return dct


@component(key="materials")
class MeshMaterial:
    def __init__(self, material: GLTFMaterial = None):
        self._mat = material

    def __hash__(self):
        return hash(self._mat.name)

    def __eq__(self, other):
        return self._mat.name == other._mat.name

    def togltf(self, index_map=None):
        return self._mat

    @call_counter
    def deps(self):
        return dict(materials=[self.global_index])


class AccessorData(TypedDict):
    view: int
    name: str
    data: np.ndarray


@component(key="accessors")
class AccessorNode:
    def __init__(self, data: AccessorData):
        self.data = data
        self.next = None
        self.size = self.count * byte_stride(self.view.gltf_type, self.view.dtype)
        self._buffofset = None
        self._end = None

    @property
    def byteOffset(self):
        if self.next is None:
            return 0

        else:
            if self._buffofset is None:
                self._buffofset = self.next.byteOffset + self.next.size
            return self._buffofset

    @property
    def min(self):
        return np.min(self.buffer_data, axis=0).tolist()

    @property
    def max(self):
        return np.max(self.buffer_data, axis=0).tolist()

    @property
    def view(self):
        return component_registry["bufferViews"][self.data["view"]]

    @property
    def count(self):
        return len(self.buffer_data)

    @property
    def prev_count(self):
        if self.next is None:
            return 0
        else:
            return self.next.count

    @property
    def start(self):
        return self.prev_count

    @property
    def end(self):
        if self.next is None:
            return self.count
        elif self._end is None:
            self._end = self.next.end + self.count
        return self._end

    @property
    def buffer_data(self):
        return self.data["data"]

    @buffer_data.setter
    def buffer_data(self, v):
        if len(v) != len(self.data["data"]):
            self.data["data"] = np.array(v, dtype=self.view.np_dtype)
            self._end = None
            self._buffofset = None
        else:
            self.data["data"] = np.array(v, dtype=self.view.np_dtype)

    def togltf(self, index_map=None):
        res = {
            "componentType": self.view.dtype,
            "count": self.count,
            "max": self.max,
            "min": self.min,
            "type": self.view.gltf_type,
        }
        if self.byteOffset > 0:
            res["byteOffset"] = self.byteOffset

        relative_index(self.view, index_map=index_map)
        return GLTFAccessor(
            bufferView=relative_index(self.view, index_map=index_map), **res
        )

    def todict(self):
        res = {
            "bufferView": self.view.doc_index,
            "componentType": self.view.dtype,
            "count": self.count,
            "max": self.max,
            "min": self.min,
            "type": self.view.gltf_type,
        }
        if self.byteOffset > 0:
            res["byteOffset"] = self.byteOffset
        return res

    @call_counter
    def deps(self):
        return dict(bufferViews=[self.view.global_index])


def merge_accessors(*others):
    self = others[0]
    if len(others) > 1:
        for other in others[1:]:
            if other is not None:
                self.buffer_data = np.c_[self.buffer_data, other.buffer_data]
                self.view.accessors.remove_node(other)
                accessors.remove(other)
                del other

    return self


def merge_indices_accessors(*others):
    self = others[0]
    if len(others) > 1:
        for other in others[1:]:
            other_buff = other.buffer_data + np.max(self.buffer_data.flatten())
            self.buffer_data = np.c_[self.buffer_data, other_buff]
            self.view.accessors.remove_node(other)
            accessors.remove(other)
            del other

    return self


# Create a LinkedList class


class AccessorList:
    def __init__(self):
        self.head = None

    # Method to add a node at begin of LL
    def insertAtBegin(self, data):
        new_node = AccessorNode(data)
        if self.head is None:
            self.head = new_node
            return new_node
        else:
            new_node.next = self.head
            self.head = new_node
        return new_node

    # Method to add a node at any index
    # Indexing starts from 0.
    def insertAtIndex(self, data: AccessorData, index: int):
        new_node = AccessorNode(data)
        current_node = self.head
        position = 0
        if position == index:
            self.insertAtBegin(data)
        else:
            while current_node != None and position + 1 != index:
                position = position + 1
                current_node = current_node.next

            if current_node != None:
                new_node.next = current_node.next
                current_node.next = new_node
            else:
                print("Index not present")
        return new_node

    # Method to add a node at the end of LL

    def insertAtEnd(self, data: AccessorData):
        new_node = AccessorNode(data)
        if self.head is None:
            self.head = new_node
            return new_node

        current_node = self.head
        while current_node.next:
            current_node = current_node.next

        current_node.next = new_node
        return new_node

    # Update node of a linked list
    # at given position
    def updateNode(self, val, index):
        current_node = self.head
        position = 0
        if position == index:
            current_node.data = val
        else:
            while current_node != None and position != index:
                position = position + 1
                current_node = current_node.next

            if current_node != None:
                current_node.data = val
            else:
                print("Index not present")

    # Method to remove first node of linked list

    def remove_first_node(self):
        if self.head == None:
            return

        self.head = self.head.next

    # Method to remove last node of linked list
    def remove_last_node(self):
        if self.head is None:
            return

        current_node = self.head
        while current_node.next.next:
            current_node = current_node.next

        current_node.next = None

    # Method to remove at given index
    def remove_at_index(self, index):
        if self.head == None:
            return

        current_node = self.head
        position = 0
        if position == index:
            self.remove_first_node()
        else:
            while current_node != None and position + 1 != index:
                position = position + 1
                current_node = current_node.next

            if current_node != None:
                current_node.next = current_node.next.next
            else:
                print("Index not present")

    # Method to remove a node from linked list
    def remove_node(self, node):
        current_node = self.head

        while (
            current_node != None and current_node.next.global_index != node.global_index
        ):
            current_node = current_node.next

        if current_node == None:
            return
        else:
            current_node.next = current_node.next.next

    def index(self, node):
        current_node = self.head
        i = 0
        while (
            current_node != None and current_node.next.global_index != node.global_index
        ):
            current_node = current_node.next
            i += 1
        if current_node == None:
            return
        else:
            return i

    def get(self, ixs, __default=None):
        current_node = self.head
        i = 0
        while True:
            if i == ixs:
                break
            elif current_node.next is None:
                break
            current_node = current_node.next
            i += 1
        if current_node == None:
            return __default
        else:
            return current_node

    # Print the size of linked list
    def sizeOfLL(self):
        size = 0
        if self.head:
            current_node = self.head
            while current_node:
                size = size + 1
                current_node = current_node.next
            return size
        else:
            return 0

    # print method for the linked list
    def printLL(self):
        current_node = self.head
        while current_node:
            print(current_node.data)
            current_node = current_node.next

    def __getitem__(self, item):
        return self.get(item)

    def __setitem__(self, item, v):
        return self.insertAtIndex(item, v)

    def __contains__(self, item: AccessorNode):
        return self.index(item) is not None

    def __iter__(self):
        return AccessorListIterator(self)


class AccessorListIterator:
    def __init__(self, ll):
        self.curnode = ll.head

    def __iter__(self):
        return self

    def __next__(self):
        if self.curnode is not None:
            node = self.curnode
            self.curnode = self.curnode.next
            return node
        else:
            raise StopIteration()


@component(key="bufferViews")
class BufferView:
    def __init__(self, gltf_type, dtype, name="view"):
        self.buffer = bytearray()
        self.name = name
        self._docindex = 0
        self.gltf_type, self.dtype = gltf_type, dtype
        # self.data =
        self.isempty = True
        self.accessors = AccessorList()

    @property
    def np_dtype(self):
        return componentTypeCodeTable[self.dtype]["numpy"]

    def new_accessor(self, data=(), name=None):
        self.isempty = False
        return self.accessors.insertAtBegin(
            dict(
                view=self.global_index,
                data=np.array(data, dtype=self.np_dtype),
                name=name,
            )
        )

    @property
    def doc_index(self):
        return self._docindex

    @property
    def buffer_data(self):
        r = [
            np.array(
                accessor.buffer_data, dtype=componentTypeCodeTable[self.dtype]["numpy"]
            )
            for accessor in self.accessors
        ]
        r.reverse()
        return np.concatenate(r)

    def to_bytes(self):
        ba = bytearray()
        for part in self.buffer_data:
            ba.extend(part.to_bytes())
        return ba

    def pack(self, buffer):
        if len(list(self.accessors)) > 1:
            return GLTFBufferView(
                **appendBufferView(
                    self.buffer_data,
                    buffer,
                    self.gltf_type,
                    self.dtype,
                    name=self.name,
                    use_stride=True,
                )
            )
        else:
            return GLTFBufferView(
                **appendBufferView(
                    self.buffer_data,
                    buffer,
                    self.gltf_type,
                    self.dtype,
                    name=self.name,
                    use_stride=False,
                )
            )


v1 = BufferView("VEC3", 5126, name="vec3view")
v2 = BufferView("VEC2", 5126, name="vec2view")
v3 = BufferView("SCALAR", 5125, name="indices_view")

view_typemap = {"VEC3": v1, "VEC2": v2, "SCALAR": v3}
from mmcore.geom.shapes import Shape, ShapeInterface

from mmcore.compat.gltf.components import GLTFPrimitive, GLTFMesh


def reshape_indices(indices):
    ixs = np.array(indices).flatten()
    return ixs.reshape((len(ixs), 1))


def path_to_offset(path):
    return Shape([list(p.values()) for p in path["points"].values()])


def _merge_list_values(d1, d2=None):
    """

    Parameters
    ----------
    d1
    d2

    -------

    Examples
    ----------

    >>> d1 = dict(a=[2], c=[7])
    >>> d2 = dict(b=[5], c=[77])
    >>> merge_list_values(d1, d2)
    {'c': [7, 77], 'a': [2], 'b': [5]}
    """

    for k in d1.keys() | d2.keys():
        yield k, d1.get(k, []) + d2.get(k, [])


def merge_list_values(d1, d2=None):
    """

    Parameters
    ----------
    d1
    d2

    -------

    Examples
    ----------

    >>> d1 = dict(a=[2], c=[7])
    >>> d2 = dict(b=[5], c=[77])
    >>> merge_list_values(d1, d2)
    {'c': [7, 77], 'a': [2], 'b': [5]}
    """

    for k in d1.keys() | d2.keys():
        _v2 = d2.get(k, None)
        if k not in d1:
            d1[k] = _v2
        else:
            targ = d1[k]
            if _v2 is not None:
                for c in _v2:
                    if c not in targ:
                        targ.append(c)

    return d1


def merge_dict_list_values_sequence(dicts):
    return reduce(lambda d1, d2: dict(merge_list_values(d1, d2)), dicts)


class MeshAttributes:
    __annotations__ = {
        "position": list,
        "normal": list,
        "tangent": list,
        "uv": list,
        "color": list,
        "joints": list,
        "weights": list,
    }

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            view = view_typemap[attrTable[k]["gltf_type"]]

            self.__dict__[k] = view.new_accessor(v)

    def gltf_mesh_primitive_attributes(self, index_map=None):
        for k in self.__dict__.keys():
            yield attrTable[k]["gltf"], relative_index(
                self.__dict__[k], index_map=index_map
            )

    @call_counter
    def deps(self) -> dict[str, list[Any]]:
        dct = dict(
            accessors=[self.__dict__[k].global_index for k in self.__dict__.keys()]
        )

        return dct

    def togltf(self, index_map=None):
        return dict(self.gltf_mesh_primitive_attributes(index_map=index_map))

    def accessors(self):
        return {attrTable[k]["gltf"]: self.__dict__[k] for k in self.__dict__.keys()}

    def merge(self, other):
        acc1 = self.accessors()
        acc2 = other.accessors()
        keys = acc1.keys() | acc2.keys()

        for k in keys:
            self.__dict__[k] = merge_accessors(acc1.get(k), acc2.get(k))

        return self


class MeshPart:
    def __init__(self, attrs, indices=None, material: MeshMaterial = None, **kwargs):
        self._attrs = MeshAttributes(**attrs)
        self._material = material
        self._indices = None
        if indices is not None:
            self._indices = view_typemap["SCALAR"].new_accessor(indices)
        self.extras = kwargs

    @call_counter
    def deps(self) -> dict[Any, IndexOrderedSet]:
        _deps = self._attrs.deps()
        if self._indices is not None:
            _deps["accessors"].append(self._indices.global_index)
        if self._material is not None:
            _deps["materials"] = [self._material.global_index]
        return _deps

    def togltf(self, index_map=None):
        return GLTFPrimitive(
            attributes=self._attrs.togltf(index_map=index_map),
            indices=relative_index(self._indices, index_map=index_map),
            material=relative_index(self._material, index_map=index_map),
        )

    def merge(self, other: "MeshPart"):
        self._attrs = self._attrs.merge(other._attrs)
        if self._indices:
            self._indices = merge_indices_accessors(self._indices, other._indices)

        return self


meshes = []


@component(key="meshes")
class Mesh:
    def __init__(self, parts: list[MeshPart] = (), name=None, extras=None):
        self._parts = list(parts)
        self._name = name
        self.extras = extras

    @call_counter
    def deps(self) -> dict[Any, IndexOrderedSet]:
        dct = dict()
        for part in self._parts:
            for k, v in part.deps().items():
                if k not in dct:
                    dct[k] = v
                elif k == "materials":
                    for item in v:
                        if item not in dct[k]:
                            dct[k].append(item)
                else:
                    dct[k].extend(v)

        return dct

    def merge_inplace(self, other: "Mesh"):
        self._parts.extend(other._parts)

        return self

    def merge_parts(self):
        """
        Always in place!
        Returns
        -------

        """

        self._parts = [reduce(MeshPart.merge, self._parts)]

    def merge_shallow(self, other: "Mesh"):
        return Mesh(self._parts + other._parts, name=f"{self._name}_{other._name}")

    def deep_merge(self, other):
        """
        Always in place!
        Returns
        -------

        """
        self.merge_inplace(other).merge_parts()
        return self

    __merge_modes__ = dict(
        shallow=merge_shallow, inplace=merge_inplace, deep=deep_merge
    )

    def merge(self, other: "Mesh", mode="shallow"):
        """

        Parameters
        ----------
        other : Mesh
        mode : str
            One of shallow, inplace, deep.
            If shallow:
                Return a new mesh consisting of primitives of the original meshes.
                The original meshes themselves will remain untouched.
            If inplace:
                The second mash merges with the first. Return second mesh.
            If deep:
                The second meshes merge with the first, and all primitives merge into one. Return second mesh.

        Returns Mesh
        -------

        """
        return self.__class__.__merge_modes__[mode](self, other)

    def togltf(self, index_map=None, **kwargs):
        return GLTFMesh(
            primitives=[part.togltf(index_map=index_map) for part in self._parts],
            name=self._name,
            extras=self.extras,
            **kwargs,
        )


myshape = Shape(
    [
        (-12.0, 7.0, 0.0),
        (-2.0, 19.0, 0.0),
        (18.0, 14.0, 0.0),
        (16.0, 1.0, 0.0),
        (4.0, -17.0, 0.0),
        (-3.0, -13.0, 0.0),
        (4.0, -2.0, 0.0),
        (-3.0, 2.0, 0.0),
        (-5.0, -1.0, 0.0),
        (-2.0, -3.0, 0.0),
        (-4.0, -6.0, 0.0),
        (-20.0, -2.0, 0.0),
    ],
    holes=[[(4.0, 7.0, 0.0), (4.0, 13.0, 0.0), (11.0, 11.0, 0.0), (11.0, 7.0, 0.0)]],
)
DEFAULT_MATERIAL_COMPONENT = MeshMaterial(DEFAULT_MATERIAL)


def mesh_part_from_shape(shape, material=DEFAULT_MATERIAL_COMPONENT, **kwargs):
    return MeshPart(
        shape.earcut.attributes,
        indices=reshape_indices(shape.earcut.indices),
        material=material,
        **kwargs,
    )


# res = go(myshape)
# with open('test3.gltf', 'w') as f:
#    json.dump(res, f, indent=2)
def case1():
    data = [
        {
            "bounds": [
                (7.0, 9.0, 15.0),
                (7.0, 9.0, 0.0),
                (24.0, 9.0, 0.0),
                (24.0, 9.0, 15.0),
                (7.0, 9.0, 15.0),
            ],
            "holes": None,
        },
        {
            "bounds": [
                (24.0, 24.0, 15.0),
                (24.0, 9.0, 15.0),
                (24.0, 9.0, 0.0),
                (24.0, 24.0, 0.0),
                (24.0, 24.0, 15.0),
            ],
            "holes": None,
        },
        {
            "bounds": [
                (7.0, 24.0, 15.0),
                (24.0, 24.0, 15.0),
                (24.0, 24.0, 0.0),
                (7.0, 24.0, 0.0),
                (7.0, 24.0, 15.0),
            ],
            "holes": None,
        },
        {
            "bounds": [
                (7.0, 24.0, 15.0),
                (7.0, 24.0, 0.0),
                (7.0, 9.0, 0.0),
                (7.0, 9.0, 15.0),
                (7.0, 24.0, 15.0),
            ],
            "holes": None,
        },
        {
            "bounds": [
                (7.0, 24.0, 0.0),
                (7.0, 9.0, 0.0),
                (24.0, 9.0, 0.0),
                (24.0, 24.0, 0.0),
                (7.0, 24.0, 0.0),
            ],
            "holes": None,
        },
        {
            "bounds": [
                (7.0, 24.0, 15.0),
                (7.0, 9.0, 15.0),
                (24.0, 9.0, 15.0),
                (24.0, 24.0, 15.0),
                (7.0, 24.0, 15.0),
            ],
            "holes": [
                [
                    (20.0, 16.0, 15.0),
                    (13.0, 16.0, 15.0),
                    (13.0, 22.0, 15.0),
                    (20.0, 22.0, 15.0),
                    (20.0, 16.0, 15.0),
                ]
            ],
        },
        {
            "bounds": [
                (13.0, 16.0, 15.0),
                (13.0, 16.0, 8.0),
                (13.0, 22.0, 8.0),
                (13.0, 22.0, 15.0),
                (13.0, 16.0, 15.0),
            ],
            "holes": None,
        },
        {
            "bounds": [
                (20.0, 22.0, 15.0),
                (13.0, 22.0, 15.0),
                (13.0, 22.0, 8.0),
                (20.0, 22.0, 8.0),
                (20.0, 22.0, 15.0),
            ],
            "holes": None,
        },
        {
            "bounds": [
                (20.0, 16.0, 15.0),
                (20.0, 22.0, 15.0),
                (20.0, 22.0, 8.0),
                (20.0, 16.0, 8.0),
                (20.0, 16.0, 15.0),
            ],
            "holes": None,
        },
        {
            "bounds": [
                (20.0, 16.0, 15.0),
                (20.0, 16.0, 8.0),
                (13.0, 16.0, 8.0),
                (13.0, 16.0, 15.0),
                (20.0, 16.0, 15.0),
            ],
            "holes": None,
        },
        {
            "bounds": [
                (20.0, 16.0, 8.0),
                (13.0, 16.0, 8.0),
                (13.0, 22.0, 8.0),
                (20.0, 22.0, 8.0),
                (20.0, 16.0, 8.0),
            ],
            "holes": None,
        },
    ]

    data2 = [
        {
            "bounds": [
                (-2.4414586181062736, 6.643768432937762, 18.475367792037417),
                (-2.4414586181062736, 1.6940209646319282, 13.525620323731589),
                (2.4734536476276787, -0.7394577614039015, 15.95909904976742),
                (2.4734536476276787, 4.210289706901932, 20.90884651807325),
                (-2.4414586181062736, 6.643768432937762, 18.475367792037417),
            ],
            "holes": None,
        },
        {
            "bounds": [
                (5.914912265733953, 3.0, 12.74782122709649),
                (4.799810136790509, 3.0, 15.0),
                (5.914912265733953, 5.252178772903514, 15.0),
                (5.914912265733953, 7.685657498939348, 17.43347872603583),
                (2.4734536476276787, 4.210289706901932, 20.90884651807325),
                (2.4734536476276787, -0.7394577614039015, 15.95909904976742),
                (5.914912265733953, 2.735910030633514, 12.483731257730003),
                (5.914912265733953, 3.0, 12.74782122709649),
            ],
            "holes": None,
        },
        {
            "bounds": [
                (5.914912265733953, 3.0, 12.74782122709649),
                (5.914912265733953, 2.735910030633514, 12.483731257730003),
                (5.381528096063782, 3.0, 12.219641288363519),
                (5.914912265733953, 3.0, 12.74782122709649),
            ],
            "holes": None,
        },
        {
            "bounds": [
                (1.0, 10.119136224975179, 15.0),
                (1.0, 5.169388756669344, 10.05025253169417),
                (-2.4414586181062736, 1.6940209646319282, 13.525620323731589),
                (-2.4414586181062736, 6.643768432937762, 18.475367792037417),
                (1.0, 10.119136224975179, 15.0),
            ],
            "holes": None,
        },
        {
            "bounds": [
                (1.0, 2.9999999999999982, 12.219641288363519),
                (1.0, 5.169388756669344, 10.05025253169417),
                (-2.4414586181062736, 1.6940209646319282, 13.525620323731589),
                (2.4734536476276787, -0.7394577614039015, 15.95909904976742),
                (5.914912265733953, 2.735910030633514, 12.483731257730003),
                (5.381528096063782, 3.0, 12.219641288363519),
                (1.0, 2.9999999999999982, 12.219641288363519),
            ],
            "holes": None,
        },
        {
            "bounds": [
                (1.0, 10.119136224975179, 15.0),
                (-2.4414586181062736, 6.643768432937762, 18.475367792037417),
                (2.4734536476276787, 4.210289706901932, 20.90884651807325),
                (5.914912265733953, 7.685657498939348, 17.43347872603583),
                (1.0, 10.119136224975179, 15.0),
            ],
            "holes": None,
        },
        {
            "bounds": [
                (4.799810136790509, 3.0, 15.0),
                (5.914912265733953, 3.0, 12.74782122709649),
                (5.381528096063782, 3.0, 12.219641288363519),
                (1.0, 2.9999999999999982, 12.219641288363519),
                (1.0, 3.0, 15.0),
                (4.799810136790509, 3.0, 15.0),
            ],
            "holes": None,
        },
        {
            "bounds": [
                (4.799810136790509, 3.0, 15.0),
                (5.914912265733953, 5.252178772903514, 15.0),
                (1.0, 10.119136224975179, 15.0),
                (1.0, 3.0, 15.0),
                (4.799810136790509, 3.0, 15.0),
            ],
            "holes": None,
        },
        {
            "bounds": [
                (1.0, 10.119136224975179, 15.0),
                (5.914912265733953, 5.252178772903514, 15.0),
                (5.914912265733953, 7.685657498939348, 17.43347872603583),
                (1.0, 10.119136224975179, 15.0),
            ],
            "holes": None,
        },
        {
            "bounds": [
                (1.0, 5.169388756669344, 10.05025253169417),
                (1.0, 10.119136224975179, 15.0),
                (1.0, 3.0, 15.0),
                (1.0, 2.9999999999999982, 12.219641288363519),
                (1.0, 5.169388756669344, 10.05025253169417),
            ],
            "holes": None,
        },
    ]

    shapes = [Shape.from_shape_interface_dict(**item) for item in data]
    shapes2 = [Shape.from_shape_interface_dict(**item) for item in data2]
    shape1mesh = Mesh(
        [mesh_part_from_shape(shp, material=random(GLTFMaterial)) for shp in shapes],
        name="shape1mesh",
    )
    shape2mesh = Mesh(
        [mesh_part_from_shape(shp, material=DEFAULT_MATERIAL) for shp in shapes2],
        name="shape2mesh",
    )

    node1 = SceneNode(mesh=shape1mesh, name="shape1")
    node2 = SceneNode(mesh=shape2mesh, name="shape2")

    node0 = SceneNode(children=[node1, node2], name="shapes")
    _data = asscene(node0)
    with open("test44.gltf", "w") as f:
        json.dump(_data.todict(), f, indent=2)
    return shapes, shapes2, shape1mesh, shape2mesh, node1, node2, node0, _data


import ujson


def _gen(shapes, names, mask, tags):
    cols = {"A-0": (0.5, 0.5, 0.5)}
    for i, t in enumerate(shapes):
        if mask[i] != 2:
            name = names[i]
            if name in tags:
                tag = tags[name]
                if tag not in cols:
                    # print(tag)

                    cols[tag] = tuple(np.random.random(3))
                col = cols[tag]

            else:
                col = (0.5, 0.5, 0.5)

            for tt in t:
                if len(tt) > 3:
                    if tt[0] == tt[-1]:
                        tt = tt[:-1]
                    yield ShapeInterface(tt), col
                elif len(tt) == 3:
                    yield ShapeInterface(tt), col


from mmcore.geom.mesh import MeshTuple, union_mesh_old, mesh_from_shapes


def case2(
    parts=[
        "w1",
        "w2",
        "w3",
        "w4",
        "l2",
        "f1",
        "f2",
        "f3",
        "f5",
        "sl1",
        "sl3",
        "sl1b",
        "sl2b",
        "sl3b",
        "sl4b",
    ],
    random_mat=False,
):
    # print(parts, "random mterial:", random_mat)
    # s = time.time()

    # with open('test_data.json', 'r') as f:
    #    _ = ujson.load(f)
    #    coldata = _['coldata']
    #    ress = _['ress']
    resp = requests.post(
        "https://viewer.contextmachine.online/cxm/api/v2/mfb_contour_server/sw/contours-merged",
        json=dict(names=parts),
    )
    resp2 = requests.get(
        "https://viewer.contextmachine.online/cxm/api/v2/mfb_sw_w/stats"
    )
    resp3 = requests.get(
        "https://viewer.contextmachine.online/cxm/api/v2/mfb_sw_l2/stats"
    )
    resp4 = requests.get(
        "https://viewer.contextmachine.online/cxm/api/v2/mfb_sw_f/stats"
    )
    coldata = {
        rr["name"]: rr["tag"] for rr in resp2.json() + resp3.json() + resp4.json()
    }
    mats = {"A-0": DEFAULT_MATERIAL_COMPONENT}
    cols = {"A-0": (0.5, 0.5, 0.5)}
    ress = resp.json()
    # print("request", divmod(time.time() - s, 60))
    parts = []
    s = time.time()

    shps, cols = list(zip(*_gen(ress["shapes"], ress["names"], ress["mask"], coldata)))
    print(shps[:3], cols[:3])
    m = union_mesh_old(mesh_from_shapes(shps, cols), ks=["position", "color"])

    print("creating parts", divmod(time.time() - s, 60))
    s = time.time()

    allshp = Mesh(
        [
            MeshPart(
                {k: v.reshape((len(v) // 3, 3)) for k, v in m.attributes.items()},
                indices=m.indices.reshape((len(m.indices), 1)),
                material=DEFAULT_MATERIAL_COMPONENT,
            )
        ],
        name="allshapes",
        extras={"parts": m.extras["parts"].tolist()},
    )
    print("create mesh", divmod(time.time() - s, 60))
    s = time.time()
    node_test = SceneNode(mesh=allshp, name="sw_node")
    print("create nodes", divmod(time.time() - s, 60))
    s = time.time()

    scene2 = asscene(node_test)
    print("create scene", divmod(time.time() - s, 60))
    s = time.time()
    scene_dct = scene2.todict()
    print("create dict", divmod(time.time() - s, 60))
    s = time.time()
    with open("testsw.gltf", "w") as f:
        ujson.dump(scene_dct, f, indent=2)
    print("dump json", divmod(time.time() - s, 60))  # print(call_count_dict)


def create_union_mesh_node(m: MeshTuple, name="mesh"):
    if m.indices:
        gltf_mesh = Mesh(
            [
                MeshPart(
                    {k: v.reshape((len(v) // 3, 3)) for k, v in m.attributes.items()},
                    indices=m.indices.reshape((len(m.indices), 1)),
                    material=DEFAULT_MATERIAL_COMPONENT,
                )
            ],
            name=name,
            extras={"parts": m.extras["parts"].tolist()},
        )
    else:
        gltf_mesh = Mesh(
            [
                MeshPart(
                    {k: v.reshape((len(v) // 3, 3)) for k, v in m.attributes.items()},
                    material=DEFAULT_MATERIAL_COMPONENT,
                )
            ],
            name=name,
            extras={"parts": m.extras["parts"].tolist()},
        )

    return SceneNode(mesh=gltf_mesh, name=f"{name}_node")


def get_mesh_wrap(mesh):
    parts = mesh.extras["parts"]
    l = np.arange(len(parts))
    u = np.unique(parts, return_index=True, return_inverse=True)
    pc = Counter(parts)
    pos = np.array(mesh.primitives[0].attributes.position)
    cols = np.array(mesh.primitives[0].attributes.color)
    ixsa = len(mesh.primitives[0].indices)
    faces = np.array(mesh.primitives[0].indices).reshape((ixsa // 3, 3))

    def get_mesh(i):
        s = u[1][i]
        fcs = faces[l[s : s + pc[i]]]
        uniq, rv = np.unique(fcs.flatten(), return_inverse=True)

        return {
            "attributes": {"position": pos[uniq], "color": cols[uniq]},
            "indices": rv,
        }

    return get_mesh


case2()

# case2(random_mat=True)
