import typing
from abc import abstractmethod
from collections import namedtuple
DEBUG_GRAPH = False
STARTUP = False
objdict = dict()
geomdict = dict()
matdict = dict()
adict = dict()
ageomdict = dict()
amatdict = dict()
idict = dict()
propsdict = dict()
# Usage example:
# from mmcore.base.registry.fcpickle import FSDB
# from mmcore.base.basic import Object3D
# c= Object3D(name="A")
# FSDB['obj']= obj
# ...
# shell:
# python -m pickle .pkl/obj
# [mmcore] : Object3D(priority=1.0,
#                    children_count=0,
#                    name=A,
#                    part=NE) at cf3d55d7-677e-4f96-9e31-b628c3962520
#

T = typing.TypeVar("T")
N = typing.TypeVar("N")
IndexKeyValue = namedtuple('IndexKeyValue', ["i", "k", "v"])


class OrderedKeysDict(dict):
    _keys = []
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self._keys = []

    def __setitem__(self, key, value):
        if not (key in self._keys):
            self._keys.append(key)
        dict.__setitem__(self, key, value)

    def to_order(self):
        l = []
        for i, k in enumerate(self._keys):
            l.append(IndexKeyValue(i, k, self[k]))
        return l

    def index_from_key(self, key):
        return self._keys.index(key)

    def getitem_from_index(self, i: int):
        return self[self._keys[i]]

    def last(self):
        return self.getitem_from_index(-1)


class AGraph(typing.Generic[N]):
    def __init__(self):
        self.item_table: OrderedKeysDict[str, typing.Union[T, N]] = OrderedKeysDict()
        self.relay_table: dict[str, dict] = dict()

    def __setitem__(self, k: str, v: typing.Union[T, N]):
        self.item_table[k] = v

    def __getitem__(self, k: str) -> T:
        return self.item_table[k]

    def __delitem__(self, key):
        self.item_table[key].del_from_graph(self)

    def nodes(self):
        return self.item_table.values()

    def relays(self):
        return self.relay_table.values()

    def get_from_name(self, name):
        nodes = []
        for i in self.item_table.values():
            if i.name == name:
                nodes.append(i)
        return nodes

    def get_from_startswith(self, name):
        nodes = []
        for i in self.keys():

            if str(i.name).startswith(str(name)):
                nodes.append(i)
        return nodes

    def get_from_callback(self, callback=lambda node: True):
        nodes = []
        for i in self.item_table.values():
            if callback(i):
                nodes.append(i)

        return nodes

    def get(self, k):
        return self.item_table.get(k)

    def get_relays(self, node):
        return dict((k, self.get_relay(node, k)) for k in self.relay_table[node.uuid].keys())
    def keys(self):
        return self.relay_table.keys()
    def get_relay(self, node, name):
        return self.item_table[self.relay_table[node.uuid][name]]

    @abstractmethod
    def set_relay(self, node: T, name: str, v: typing.Any):
        ...

    def __repr__(self):
        return self.__class__.__name__ + f'({self.item_table.__repr__()}, {self.relay_table.__repr__()})'

    def __contains__(self, item: typing.Union[T, N]):
        return item.uuid in self.item_table
