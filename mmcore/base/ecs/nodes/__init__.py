import typing
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import Enum
from uuid import uuid4

import ujson

DB = dict(
    nodes=dict(),
    links=dict()
)

ContainerLinkType = (list, tuple)
MappingLinkType = (dict,)


class NodeType(str, Enum):
    LEAF = 'LEAF'
    MAPPING = 'MAPPING'
    CONTAINER = 'CONTAINER'


class Visitable:
    def accept(self, visitor):
        lookup = "visit_" + type(self).__qualname__.replace(".", "_").lower()
        return getattr(visitor, lookup)(self)


def node_from_data(data, db=None):
    if isinstance(data, MappingLinkType):
        return node_from_dict(data, db=db)
    elif isinstance(data, ContainerLinkType):
        return node_from_list(data, db=db)
    else:
        return leaf_from_value(data, db=db)


def leaf_from_value(data, db=None):
    return Node(data=data, db=db)


def node_from_list(lst, db=None):
    return NodeContainer(data=[node_from_data(l, db=db) for l in lst], db=db)


def node_from_dict(dct, db=None):
    n = Node(db=db)
    for k, v in dct.items():
        if isinstance(v, MappingLinkType):
            n.add_child(k, node_from_dict(v, db=db))
        elif isinstance(v, ContainerLinkType):
            n.add_child(k, node_from_list(v, db=db))
        else:
            n.add_child(k, node_from_data(v, db=db))

    return n


def update_node(node, data):
    if isinstance(data, MappingLinkType):
        return update_node_from_dict(node, data)
    elif isinstance(data, ContainerLinkType):
        return update_node_from_list(node, data)
    else:
        return update_leaf_from_value(node, data)


def update_leaf_from_value(node, data):
    node._data = data


def update_node_from_list(node, lst):
    for n, l in zip(node.children, lst):
        update_node(n, l)


def update_node_from_dict(node, dct):
    for k, v in dct.items():
        if isinstance(v, MappingLinkType):

            update_node_from_dict(node[k], v)
        elif isinstance(v, ContainerLinkType):
            update_node_from_list(node[k], v)
        else:
            update_node(node[k], v)
    return node


@dataclass(slots=True)
class NodeSpec:
    uuid: str
    leaf: bool
    data: typing.Any


class ResponsibilityChain:
    def __init__(self, fun):
        """change or increase the local variable using nxt"""
        self._fun = fun
        self._next = None

    def next(self, nxt):
        self._next = ResponsibilityChain(nxt)
        return self._next

    def __call__(self, *args, **kwargs):
        return self.handle(*args, **kwargs)

    def handle(self, *args, **kwargs):
        """It calls the processRequest through given request"""

        handled = self._fun(*args, **kwargs)

        """case when it is not handled"""

        if handled is None:
            return self._next.handle(*args, **kwargs)
        return handled


@ResponsibilityChain
def node_from_spec(spec, db=None):
    if spec.leaf:
        return Node(uuid=spec.uuid,
                    data=spec.data,
                    db=db)


@node_from_spec.next
def node_from_spec_container(spec: NodeSpec, db=None):
    if isinstance(spec.data, ContainerLinkType):
        return NodeContainer(uuid=spec.uuid,
                             data=[node_from_spec(l, db=db) for l in spec.data],
                             db=db)


@node_from_spec_container.next
def node_from_spec_mapping(spec: NodeSpec, db=None) -> 'Node':
    if isinstance(spec.data, MappingLinkType):
        n = Node(uuid=spec.uuid, db=db)
        for k, v in spec.data.items():
            n.add_child(k, node_from_spec(spec=v, db=db))
        return n


def node_to_spec_leaf(node: 'Node') -> NodeSpec:
    return NodeSpec(uuid=node.uuid, leaf=node.leaf, data=node.value)


def node_to_spec_container(node: 'Node') -> NodeSpec:
    return NodeSpec(uuid=node.uuid,
                    leaf=node.leaf,
                    data=[node_to_spec(node.get_node(v)) for v in node.links])


def node_to_spec_mapping(node: 'Node') -> NodeSpec:
    return NodeSpec(uuid=node.uuid,
                    leaf=node.leaf,
                    data=dict((k, node_to_spec(node.get_node(v))) for k, v in node.links.items()))


def node_to_spec(node: 'Node') -> NodeSpec:
    if NodePredicates.leaf(node):
        return node_to_spec_leaf(node)
    elif NodePredicates.container(node):
        return node_to_spec_container(node)
    elif NodePredicates.mapping(node):
        return node_to_spec_mapping(node)
    else:
        TypeError(f'Unknown type for {node.__repr__()} node!)')


class NodePredicates:
    @classmethod
    def leaf(cls, node: 'Node'):
        return len(node.links) < 1

    @classmethod
    def container(cls, node: 'Node'):
        return all([not cls.leaf(node), isinstance(node.links, ContainerLinkType)])

    @classmethod
    def mapping(cls, node: 'Node'):
        return all([not cls.leaf(node), isinstance(node.links, MappingLinkType)])


def tospec(self):
    if self.leaf:
        return dict(uuid=self.uuid, leaf=self.leaf, data=self._data)
    return dict(uuid=self.uuid, leaf=self.leaf,
                data=dict((k, self.get_node(v).tospec()) for k, v in self.links.items()))


class NodeFactory:
    def __init__(self, db=DB):
        self.db = db

    def __call__(self, data=None, uuid=None):
        if uuid in self.db['nodes']:
            return self.from_uuid_unsafe(uuid=uuid)
        else:
            return self.from_data(data)

    def from_data(self, data):
        return node_from_data(data, db=self.db)

    def from_uuid_unsafe(self, uuid):
        return self.db['nodes'][uuid]

    def overwrite_from_uuid(self, uuid, data):

        return self.db['nodes'][uuid]

    def from_uuid(self, uuid):
        return Node(uuid=uuid, db=self.db)

    def from_spec(self, spec):
        if len(spec) == 1:
            return self.from_uuid_unsafe(uuid=spec['uuid'])
        if spec['leaf']:
            return Node(uuid=spec['uuid'], data=spec['data'], db=self.db)
        elif isinstance(spec['data'], list):
            return NodeContainer(uuid=spec['uuid'], data=[self.from_spec(v) for v in spec['data']], db=self.db)
        else:
            return Node(
                uuid=spec['uuid'],
                data={k: self.from_spec(v) for k, v in spec['data'].items()}, db=self.db)




class AbstractNode(metaclass=ABCMeta):
    db = DB

    def __new__(cls, uuid=None, db=DB):
        if uuid is None:
            uuid = uuid4().hex
        if uuid in db['nodes']:
            return db['nodes'][uuid]
        else:
            obj = super().__new__(cls)
            obj._uuid, obj._db = uuid, db
            obj._db['nodes'][obj._uuid] = obj
            return obj

    @abstractmethod
    def todict(self):
        pass

    def update(self, dct):
        update_node(self, dct)

    @classmethod
    def get_node(cls, uuid):
        return cls.db['nodes'][uuid]

    @abstractmethod
    def tospec(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @property
    def uuid(self):
        return self._uuid


class Node(AbstractNode):

    def __new__(cls, uuid=None, data=None, db=DB):
        obj = super().__new__(cls, uuid=uuid, db=db)
        obj._data = data

        obj._db['links'][obj._uuid] = dict()
        if isinstance(data, MappingLinkType):
            obj._db['links'][obj._uuid] |= dict((k, v.uuid) for k, v in data.items())
        if isinstance(data, MappingLinkType):
            obj._db['links'][obj._uuid] |= dict((k, v.uuid) for k, v in data.items())
        return obj

    @property
    def links(self):
        return self._db['links'][self.uuid]

    @property
    def leaf(self):
        return len(self.links) < 1

    @property
    def container(self):

        return all([not self.leaf, isinstance(self.links, ContainerLinkType)])

    @property
    def mapping(self):
        return all([not self.leaf, isinstance(self.links, MappingLinkType)])

    @property
    def value(self):
        if self.leaf:
            return self._data
        else:
            return self

    @property
    def children(self):
        return dict(map(lambda x: (x[0], self._db['nodes'][x[1]]), self.links.items()))

    def __getitem__(self, k):
        return self._db['nodes'][self.links[k]]

    @property
    def node_type(self):
        if self.leaf:
            return NodeType.LEAF

        else:
            return type(self)

    def todict(self):
        if self.leaf:
            return self._data
        else:
            return dict((k, self.get_node(v).todict()) for k, v in self.links.items())

    def add_child(self, k, v: 'Node', *args, **kwargs):
        if isinstance(v, Node):
            self.links[k] = v.uuid
        else:
            raise TypeError(k, v)

    def tospec(self):
        if self.leaf:
            return dict(uuid=self.uuid, leaf=self.leaf, data=self._data)
        return dict(uuid=self.uuid, leaf=self.leaf,
                    data=dict((k, self.get_node(v).tospec()) for k, v in self.links.items()))

    def __repr__(self):
        if self.leaf:
            return f'( {self.value} )'

        return ujson.dumps(self.todict(), indent=2)


class NodeContainer(Node):
    def __new__(cls, data=(), uuid=None, db=DB):
        self = AbstractNode.__new__(cls, uuid=uuid, db=db)
        self._db['links'][self._uuid] = list(s.uuid for s in data)
        return self

    def value(self):
        if self.leaf:
            return []
        else:
            return self

    def update(self, seq):
        update_node_from_list(self, seq)

    def tospec(self):
        return dict(uuid=self.uuid, leaf=self.leaf, data=[self.get_node(v).tospec() for v in self.links])

    def todict(self):

        return [self.get_node(v).todict() for v in self.links]

    def add_child(self, v: Node = None, *args, **kwargs):

        if isinstance(v, Node) and v.uuid not in self.links:
            self._db['links'][self._uuid].append(v.uuid)
        else:
            raise TypeError(v)

    def __repr__(self):

        return ujson.dumps(self.todict(), indent=2)


node_factory = NodeFactory(db=DB)

"""
node_factory = NodeFactory(db=DB)

res = node_factory(data=dict(foo=dict(bar=dict(
    x=7,
    y=-45,
    z=-7
)
)
    , baz=dict(
        x=87,
        y=5,
        z=0,
        w=[dict(x=4, y=0), dict(p=8, g=7)]
    )
))

res.update({
    "baz": {
        "x": 7,
        "y": -65},
    "foo": {
        "bar": {
            "x": 17}}})
"""
