import dataclasses
import datetime
import typing
import uuid
import uuid as _uuid

import dill
import networkx as nx
import pyvis
import ujson
from graphql import GraphQLScalarType, GraphQLSchema

from mmcore.base.registry import AGraph

paramtable = dict()
relaytable = dict()
DEBUG_GRAPH = False

import string
from graphql.type import GraphQLObjectType, \
    GraphQLField, \
    GraphQLUnionType, \
    GraphQLString, GraphQLNullableType, GraphQLInputObjectType, GraphQLFieldMap

Schema = GraphQLSchema()


@dataclasses.dataclass
class FlowEdge:
    id: str
    source: str
    target: str
    type: str = 'smoothstep'
    animated: bool = False

    def todict(self):
        return dataclasses.asdict(self)


def query_resolver(uid):
    return pgraph.item_table[uid].togql()


class ParamGraphNodeInput(GraphQLInputObjectType):
    name = 'ParamGraphNodeInput'
    fields = lambda: {
        'name': GraphQLField(GraphQLString()),
        'uuid': GraphQLField(GraphQLString()),
        'params': GraphQLFieldMap(dict[str, GraphQLField(
            GraphQLUnionType("ParamUnion", [ParamGraphNodeInput(), GraphQLScalarType(), JSONParamMap])
        )])
    }


class ParamGraph(AGraph):
    _scenegraph = None

    def __init__(self):
        super().__init__()
        self.scene_relay_table = dict()

    def keys(self):
        return self.item_table.keys()

    @property
    def scenegraph(self):
        return self._scenegraph

    @scenegraph.setter
    def scenegraph(self, v: AGraph):
        self._scenegraph = v
        self.scene_relay_table = dict()

    def draw(self):
        self.mappingtable, self.ajtable, self.leafs = draw_aj(self)
        return self.ajtable

    def get_terms(self) -> 'list[TermParamGraphNode]':
        nodes = []
        for node in self.item_table.values():
            if isinstance(node, TermParamGraphNode):
                nodes.append(node)
        return nodes

    def set_relay(self, node, name, v):
        if DEBUG_GRAPH:
            print(f"[{self.__class__.__name__}] SET RELAY: node:{node}\n\t{name}:{v}")
        if name in self.relay_table[node.uuid].keys():
            if isinstance(self.item_table[self.relay_table[node.uuid][name]], TermParamGraphNode):
                self.item_table[self.relay_table[node.uuid][name]](v)

            else:
                for k in self.relay_table[self.relay_table[node.uuid][name]]:
                    if v.get(k) is not None:
                        self.set_relay(self.item_table[self.relay_table[node.uuid][name]], k, v.get(k))

        else:
            if isinstance(v, (ParamGraphNode, TermParamGraphNode)):
                self.relay_table[node.uuid][name] = v.uuid
            else:
                self.relay_table[node.uuid][name] = TermParamGraphNode(v, name=f'{node.name}:{name}').uuid

    def toflow(self):
        nodes = []
        edges = []
        i = -1
        for k, v in self.relay_table.items():
            i += 1
            nodes.append(self.item_table[k].toflow(i, i))
            if isinstance(v, dict):
                for kk, vv in v.items():
                    print(vv, kk, k)
                    edges.append(FlowEdge(id=f'e{k}-{kk}', source=k, target=vv).todict())
                else:
                    pass
        return {"nodes": nodes, "edges": edges}


pgraph = ParamGraph()
JSONParamMap = GraphQLScalarType(name="JSONParamMap", serialize=lambda value: ujson.dumps(value),
                                 parse_value=lambda value: ujson.loads(value))


@dataclasses.dataclass(unsafe_hash=True)
class TermParamGraphNode:
    data: dataclasses.InitVar[typing.Any]
    name: typing.Optional[str] = None
    uuid: typing.Optional[str] = None

    def __post_init__(self, data):
        self.graph = pgraph
        if self.uuid is None:
            self.uuid = _uuid.uuid4().hex

        if self.name is None:
            self.name = f'untitled{len(self.graph.get_from_startswith("untitled"))}'
        if isinstance(data, TermParamGraphNode):
            data = data.solve()
        self.graph.item_table[self.uuid] = self
        self.graph.relay_table[self.uuid] = data

    def solve(self):
        return self.graph.relay_table[self.uuid]

    def __call__(self, data=None):

        if data is not None:
            if DEBUG_GRAPH:
                print(
                    f"[{self.__class__.__name__}] TERM PARAM UPDATE: node: {self.uuid}\n\t{self.graph.relay_table[self.uuid]} to {data}")
            if isinstance(data, TermParamGraphNode):
                data = data.solve()
            self.graph.relay_table[self.uuid] = data

        return self.solve()

    @property
    def index(self):
        return self.graph.item_table.index_from_key(self.uuid)

    def togql(self):
        return GraphQLObjectType('ParamGraphTerm', {
            "name": GraphQLField(GraphQLString, resolve=lambda: self.name),
            "uuid": GraphQLField(GraphQLString, resolve=lambda: self.uuid),
            "data": GraphQLField(GraphQLNullableType, resolve=lambda: self.solve()),

        })

    def toflow(self, x=0, y=0):
        return {
            "id": self.uuid,
            "data": {"label": f'Node {self.uuid}'},
            "position": {"x": x, "y": y}
        }


'''
@dataclasses.dataclass(unsafe_hash=True)
class RedisTermNode(TermParamGraphNode):
    graph:RedisParamGraph=rpgraph

    def __post_init__(self, data):

        if self.uuid is None:
            self.uuid = _uuid.uuid4().hex

        if self.name is None:
            self.name = f'untitled{len(self.graph.get_from_startswith("untitled"))}'
        if isinstance(data, TermParamGraphNode):
            data = data.solve()
        self.graph.item_table[self.uuid] = self
        self.graph.relay_table[self.uuid, "__data__"] = data

    def solve(self):
        return self.graph.get_relay(self, "__data__")
    def __call__(self, data=None):
        if data is not None:
            if DEBUG_GRAPH:
                print(
                    f"[{self.__class__.__name__}] TERM PARAM UPDATE: node: {self.uuid}\n\t{self.graph.relay_table[self.uuid]} to {data}")
            if isinstance(data, TermParamGraphNode):
                data = data.solve()
            self.graph.set_relay(self, "__data__", data)
        return self.graph.get_relay(self, "__data__")'''


@dataclasses.dataclass(unsafe_hash=True)
class ParamGraphNode:
    _params: dataclasses.InitVar['dict[str, typing.Union[ ParamGraphNode, typing.Any]]']
    name: typing.Optional[str] = None
    uuid: typing.Optional[str] = None
    resolver: typing.Optional[typing.Callable] = None
    graph: ParamGraph = pgraph

    def togql(self):
        def chr(**kwargs):
            if len(kwargs) > 0:
                self(**kwargs)
            return self.todict()

        def children():
            dct2 = dict()
            for k in self.keys():
                tp = self.graph.get_relay(self, k)

                dct2[k] = GraphQLField(ParamGraphNodeType, resolve=tp.togql())
            return dct2

        ParamGraphNodeType = GraphQLObjectType('GqlParamGraphNode', lambda: dict(
            name=GraphQLField(GraphQLString, resolve=lambda: self.name),
            uuid=GraphQLField(GraphQLString, resolve=lambda: self.uuid),
            params=GraphQLField(
                GraphQLObjectType(f"{self.name.capitalize()}NodeParams", children),
                resolve=lambda **kwargs: chr(**kwargs)
            )))

        return ParamGraphNodeType

    @property
    def index(self):
        return self.graph.item_table.index_from_key(self.uuid)

    def __post_init__(self, _params):

        if self.uuid is None:
            self.uuid = _uuid.uuid4().hex
        if self.name is None:
            self.name = "_"
            self.name = f'untitled{len(self.graph.get_from_startswith("untitled"))}'
        if self.resolver is None:
            self.resolver = lambda **kwargs: kwargs
        self.graph.item_table[self.uuid] = self
        self.graph.relay_table[self.uuid] = dict()

        if _params is not None:
            for k, v in _params.items():
                self.graph.set_relay(self, k, v)

    def neighbours(self):
        dct = list()

        def search(obj):

            for k in obj.keys():
                r = self.graph.get_relay(obj, k)
                if isinstance(r, ParamGraphNode):
                    dct.append(r)
                    search(r)
                else:
                    dct.append(r)

        search(self)
        return dct

    def subgraph(self):
        subgraph = ParamGraph()

        def search(obj):
            subgraph.item_table[obj.uuid] = obj
            subgraph.relay_table[obj.uuid] = dict()
            for k in obj.keys():
                r = self.graph.item_table[self.graph.relay_table[obj.uuid][k]]
                subgraph.relay_table[obj.uuid][k] = r.uuid
                if isinstance(r, ParamGraphNode):
                    search(r)

                else:
                    subgraph.item_table[r.uuid] = r
                    subgraph.relay_table[r.uuid] = self.graph.relay_table[r.uuid]
                    r.graph = subgraph

            obj.graph = subgraph

        search(self)

        return subgraph

    def get(self, k):
        try:
            return self.graph.get_relay(self, k)
        except KeyError:
            return None

    def leafs(self):
        nodes = list()

        def search(obj):

            for k in obj.keys():
                r = self.graph.get_relay(obj, k)
                if isinstance(r, TermParamGraphNode):
                    nodes.append(r)
                else:
                    search(r)

        search(self)
        return nodes

    def keys(self):
        return self.graph.relay_table[self.uuid].keys()

    def todict(self, no_attrs=True):
        dct = {}
        for k in self.keys():

            if isinstance(self.params[k], ParamGraphNode):
                dct[k] = self.graph.get_relay(self, k).todict(no_attrs)
            else:
                dct[k] = self.graph.get_relay(self, k).solve()

        if no_attrs:
            return dct

        def vl():
            res = self.solve()
            if hasattr(res, "_repr3d"):
                return res._repr3d.root()
            elif hasattr(res, "root"):
                return res._repr3d.root()
            else:
                return res

        return {
            "kind": self.__class__.__name__,
            "name": self.name,
            "uuid": self.uuid,
            "value": vl(),
            "params": dct
        }

    def solve(self):
        dct = {}
        for k in self.keys():
            dct[k] = self.graph.get_relay(self, k).solve()

        return self.resolver(**dct)

    def schema(self):
        return self.__dct_schema__.schema

    def inputs(self):
        return self.__dct_schema__.init_partial(**self.todict())

    @property
    def params(self):
        dct = dict()
        for i in self.graph.relay_table[self.uuid].keys():
            dct[i] = self.graph.item_table[self.graph.relay_table[self.uuid][i]]
        return dct

    def __setitem__(self, key, value):

        self.graph.set_relay(self, key, value)

    def __getitem__(self, key):
        return self.graph.get_relay(self, key)

    def __call__(self, *args, **params):
        if len(args) > 0:
            params |= dict(zip(list(self.keys())[:len(args)], args))
        if params is not None:

            for k, v in params.items():
                self.graph.set_relay(self, k, v)
            # self.__dct_schema__ = DictSchema(self.todict(no_attrs=True))
        return self.solve()

    def dispose(self):
        del self.graph.item_table[self.uuid]
        for k, v in self.graph.relay_table[self.uuid].items():
            self.graph.item_table[v].dispose()

        del self.graph.relay_table[self.uuid]

    def del_from_graph(self, othergraph):
        """

        @param graph:
        @return:
        >>> from mmcore.base.params import ParamGraphNode, pgraph
        >>> node = ParamGraphNode(...)
        >>> subgraph = node.subgraph()
        >>> node in subgraph
        True
        >>> node in pgraph
        True
        >>> node.del_from_graph(pgraph)
        >>> node in pgraph
        False
        """

        for i in othergraph.relay_table.get(self.uuid).values():
            othergraph.item_table[i].del_from_graph(othergraph)

        othergraph.relay_table.pop(self.uuid)
        othergraph.item_table.pop(self.uuid)

    def toflow(self, x=0, y=0):
        return {
            "id": self.uuid,
            "data": {"label": f'Node {self.uuid}'},
            "position": {"x": x, "y": y}
        }


def param_graph_node(params: dict):
    def inner(fn):
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        name = fn.__name__

        return ParamGraphNode(params, name=name, resolver=wrapper)

    return inner


def param_graph_node_native(fn):
    """
    Вся разница в том что здесь params берется из kwargs функции а не передается отдельно,
    как следствие обязательно указывать значения для параметров по умолчанию. Сейчас не ясно какой из подходов удобнее,
    поэтому они существуют оба.
    @param fn:
    @return:
    """

    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    name = fn.__name__

    return ParamGraphNode(dict(zip(fn.__code__.co_varnames, fn.__defaults__)), name=name, resolver=wrapper)


@dataclasses.dataclass
class RedisPgraphNode(ParamGraphNode):
    def __post_init__(self, _params):
        if self.uuid is None:
            self.uuid = _uuid.uuid4().hex
        if self.name is None:
            self.name = "_"
            self.name = f'untitled{len(self.graph.get_from_startswith("untitled"))}'
        if self.resolver is None:
            self.resolver = lambda **kwargs: kwargs
        self.graph.item_table[self.uuid] = self

        if _params is not None:
            for k, v in _params.items():
                self.graph.set_relay(self, k, v)


import itertools


def draw_aj(graph):
    import pandas as pd

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    l = len(graph.item_table.keys())

    dct = dict()
    for i, k in enumerate(graph.item_table.keys()):
        n, m = divmod(i, len(string.ascii_uppercase))

        dct[k] = string.ascii_uppercase[m] * (n + 1), i
    dct2 = dict()
    dct3 = dict()
    for k, v in dct.items():
        name, i = v
        *r, = itertools.repeat(' ', l)
        print(k, v)
        if isinstance(graph.relay_table[k], dict):
            for kk, vv in graph.relay_table[k].items():
                r[list(dct.keys()).index(vv)] = f'{kk} : {name}->{dct[vv][0]}'

        else:
            dct3[name] = {"value": graph.relay_table[k], "uuid": v, "name": name}
        dct2[name] = r

    print(dct2)
    *keys, = map(lambda x: x[0], dct.values())

    return pd.DataFrame(dct), pd.DataFrame(dct2, index=list(keys)), pd.DataFrame(dct3, index=['value', 'uuid'])


def restore_pgraph(s, graph: ParamGraph = None):
    global pgraph
    _graph = dill.loads(s)
    if graph is not None:

        graph.item_table |= _graph.item_table
        graph.relay_table |= _graph.relay_table

    else:
        pgraph.item_table |= _graph.item_table
        pgraph.relay_table |= _graph.relay_table


def store_pgraph(graph=None):
    if graph is None:
        global pgraph

        return dill.dumps(pgraph, byref=True, recurse=True)
    else:
        return dill.dumps(graph, byref=True, recurse=True)


from collections import namedtuple

ParentsResponse = namedtuple('ParentsResponse', ["roots", "tree"])
import copy


class MultiGraph:
    def __init__(self, uuid="api_mmcore_params"):
        self.table = dict()
        self.edges = dict()
        self.pedges = dict()
        self.uuid = uuid

    def add_edge(self, parent, name, child):
        self.edges[parent][name] = child
        self.pedges[child][parent] = name


MG = MultiGraph()


class Graph:
    def __init__(self, uuid=None, graph=MG):
        if uuid is None:
            uuid = _uuid.uuid4().hex
        self.uuid = uuid
        self.graph = graph
        self.graph.table[self.uuid] = self
        self.graph.edges[self.uuid] = dict()
        self.table = dict()
        self.edges = dict()
        self.pedges = dict()
        self.vis_config_table = dict()
        self.update_query = []

        self.attaches = dict()
        self.attaches_by_node = dict()

    def add_edge(self, parent, name, child):
        self.edges[parent][name] = child
        self.pedges[child][parent] = name


    def __copy__(self):
        dpc = copy.deepcopy(self.graph)
        dpc.uuid = uuid.uuid4().hex
        dpc.table = copy.deepcopy(self.table)
        dpc.edges = copy.deepcopy(self.edges)
        dpc.pedges = copy.deepcopy(self.pedges)

        self.graph.table[dpc.uuid] = dpc
        self.graph.edges[dpc.uuid] = dict()
        return dpc

    def to_nx(self) -> nx.DiGraph:
        G = nx.DiGraph()
        for i in self.table.keys():
            G.add_node(i)
        for k, v in self.edges.items():
            if len(v) != 0:
                for val in v.values():
                    G.add_edge(k, val)
        return G

    def to_pyvis(self, *args, **kwargs) -> pyvis.network.Network:
        net = pyvis.network.Network(*args, directed=True, **kwargs)
        net.from_nx(self.to_nx())
        return net

    def write_html(self, path, local=True, notebook: bool = False, open_browser=True, **kwargs):
        net = self.to_pyvis(**kwargs)
        return net.write_html(path, local=local, notebook=notebook, open_browser=open_browser)

    def toflow(self):
        nodes = []
        edges = []
        i = -1
        for k, v in self.edges.items():
            i += 1
            nodes.append(self.table[k].toflow(i * 100, i * 100))
            if isinstance(v, dict):
                for kk, vv in v.items():
                    print(vv, kk, k)
                    edges.append(FlowEdge(id=f'e{k}-{kk}', source=k, target=vv).todict())
                else:
                    pass
        return {"nodes": nodes, "edges": edges}
from collections import namedtuple

ParentsResponse = namedtuple('ParentsResponse', ["roots", "tree"])


class CallGraphEvent:
    def __init__(self, node, request):
        self.node = node
        self.graph = node.graph
        self.request = request
        self.changes = []
        self.compiled = False
        self.paths = dict()
        self.roots = []
        self.changes = dict()

    def all_roots(self, node):
        self.paths[node.uuid] = dict()
        if node.is_root:
            self.roots.append(node.uuid)


        else:

            for parent in node.parents():
                self.paths[node.uuid][node.graph.pedges[node.uuid][parent.uuid]] = parent.uuid

                self.all_roots(
                    parent)

    def build(self, node, request):

        self.changes[node.uuid] = dict()
        if node.is_scalar:
            print(f"change {node.uuid} value: {node.value} -> {request}")
            self.changes[node.uuid]["value"] = {"old": copy.deepcopy(node.value), "new": request}

            node.value = request

            self.all_roots(node)

        else:

            for k, v in request.items():
                self.changes[node.uuid][k] = node.edges()[k]

                self.build(node.graph.table[node.edges()[k]], v)
        self.compiled = True

    def __call__(self):

        self.build(self.node, self.request)


NG = Graph()


@dataclasses.dataclass(unsafe_hash=True)
class NodeVisConfigsPosition:
    __x: dataclasses.InitVar[float]
    __y: dataclasses.InitVar[float]
    _x = 0.0
    _y = 0.0

    def __post_init__(self, __x: float = 0.0, __y: float = 0.0):
        self.x = __x
        self.y = __y

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, v):
        self._x = v

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, v):
        self._y = v


@dataclasses.dataclass(unsafe_hash=True)
class NodeVisConfigs:
    position: NodeVisConfigsPosition



class Node:
    uuid = None

    def __init__(self, graph=NG, uuid=None, value=None, vis_config=None, **kwargs):
        self.uuid = uuid
        if uuid is None:
            self.uuid = _uuid.uuid4().hex
        self.graph = graph
        self.is_scalar = False

        if self.uuid in self.graph.table:
            raise KeyError(f"UUID {self.uuid} is exist (to write exist uuid use restore option)")
        self.graph.table[self.uuid] = self
        if vis_config is None:
            vis_config = NodeVisConfigs(position=NodeVisConfigsPosition(0.0, 0.0))
            self.graph.vis_config_table[self.uuid] = vis_config
        self.graph.pedges[self.uuid] = dict()
        self.graph.edges[self.uuid] = dict()
        self.graph.attaches_by_node[self.uuid] = dict()

        if value is not None:
            self.is_scalar = True
            self.value = value
        if len(kwargs) > 0:
            self.add_edges(**kwargs)
        self._wrapped = lambda **kwargs: kwargs
        self._resolver = lambda kwargs: self._wrapped(**kwargs)

    def keys(self):
        return self.graph.edges[self.uuid].keys()

    def edges(self):
        return self.graph.edges[self.uuid]

    def todict(self):
        if self.is_scalar:
            return self.value
        dct = {}
        for k in self.keys():
            dct[k] = self.graph.table[self.edges()[k]].todict()
        return dct

    def to(self):
        if self.is_scalar:
            return self.value
        dct = {}
        for k in self.keys():
            dct[k] = self.graph.table[self.edges()[k]]
        return dct

    def resolve_list(self, n, itm):
        l = []
        for i, j in zip(n, itm):
            if isinstance(i, (list, tuple)) and isinstance(j, (list, tuple)):
                l.extend(self.resolve_list(i, j))

            elif i in self.graph.table and isinstance(j, dict):

                l.extend(self.graph.table[i](**j))
            else:
                l.extend(self.graph.table[i](value=j))
        return l

    def parents(self):

        return [self.graph.table[uid] for uid in self.graph.pedges[self.uuid].keys()]

    def children(self):

        return [self.graph.table[uid] for uid in self.graph.edges[self.uuid].values()]

    @property
    def is_root(self):
        return len(self.parents()) == 0

    def all_connected_roots(self):
        roots = []
        if self.is_root:
            return [self.uuid]
        else:
            for parent in self.parents():
                roots.extend(parent.all_connected_roots())
            return roots

    def __getattr__(self, v):
        if v in object.__getattribute__(self, "graph").edges[object.__getattribute__(self, "uuid")].keys():
            graph = object.__getattribute__(self, "graph")
            uuid = object.__getattribute__(self, "uuid")
            node = graph.table[graph.edges[uuid][v]]
            if node.is_scalar:
                return node.value
            return node
        else:
            return object.__getattribute__(self, v)

    def __call__(self, value=None, **params):

        if self.is_scalar:
            if value is not None:
                self.value = value

            self.graph.update_query.append(self.all_connected_roots())


        else:

            if params is not None:
                changes = []
                for k, v in params.items():
                    if (k in self.edges()) and (v is not None):
                        if isinstance(v, dict):
                            changes.extend(self.graph.table[self.edges()[k]](**v))
                        elif isinstance(v, (list, tuple)) and isinstance(self.edges()[k], (list, tuple)):
                            print(v, self.edges()[k])
                            changes.extend(self.resolve_list(self.edges()[k], v))
                        else:
                            changes.extend(self.graph.table[self.edges()[k]](value=v))

                self.graph.update_query.append(changes)
        return self

    def add_edge(self, k, v):
        if not self.is_scalar:
            if isinstance(v, Node):

                self.graph.add_edge(self.uuid, k, v.uuid)

            elif isinstance(v, dict):
                node = Node(self.graph, uuid=f'{self.uuid}_{k}', **v)
                self.graph.add_edge(self.uuid, k, node.uuid)
            elif isinstance(v, (list, tuple)):
                if isinstance(v[0], (list, tuple)):
                    node = Node(self.graph, uuid=f'{self.uuid}_{k}', **dict(enumerate(v)))
                    self.graph.add_edge(self.uuid, k, node.uuid)
                else:
                    node = Node(self.graph, uuid=f'{self.uuid}_{k}', value=v)
                    self.graph.add_edge(self.uuid, k, node.uuid)


            else:
                node = Node(self.graph, uuid=f'{self.uuid}_{k}', value=v)
                self.graph.add_edge(self.uuid, k, node.uuid)
        else:
            raise

    def set_edge(self, k, v):
        if not self.is_scalar:
            self(**{k: v})

    def __repr__(self):
        if self.is_scalar:
            return f"Scalar{self.__class__.__qualname__}(value={self.value}) at {self.uuid}"
        return f"{self.__class__.__qualname__}({', '.join(f'{k}={getattr(self, k)}' for k in self.keys())}) at {self.uuid}"

    def add_edges(self, **kwargs):
        if not self.is_scalar:
            for k, v in kwargs.items():
                self.add_edge(k, v)
        else:
            raise

    def bind(self, cls):
        self._wrapped = cls
        self._resolver = lambda kwargs: self._wrapped(**kwargs)
        return self

    def solve(self):
        if self.is_scalar:
            return self.value
        return self.resolver(dict((k, self[k].solve()) for k in self.keys()))

    @property
    def resolver(self):
        return self._resolver

    def get_edge(self, k):
        return self.graph.table[self.graph.edges[self.uuid][k]]

    def __iter__(self):
        if not self.is_scalar:
            return iter(self.to().items())
        return iter([self.value])

    def __getitem__(self, item):
        res = self.graph.table[self.graph.edges[self.uuid][item]]
        if isinstance(res, Node):
            if res.is_scalar:
                return res
        return res

    def __setitem__(self, item, v):
        res = self.graph.table[self.graph.edges[self.uuid][item]]
        if isinstance(res, Node):
            if res.is_scalar:
                res(value=v)
        else:
            self(**{item: v})

    def __copy__(self):
        if self.is_scalar:
            new_node = Node(uuid=uuid.uuid4().hex, graph=self.graph, value=self.value)
            new_node.bind(self._wrapped)

        else:
            new_node = Node(uuid=uuid.uuid4().hex, graph=self.graph,
                            **dict((k, v.__copy__()) for k, v in dict(self).items()))
            new_node.bind(self._wrapped)

        return new_node

    def copy_with_graph(self):

        return self.graph.__copy__()

    @property
    def global_uuid(self):
        return self.graph.graph.uuid + "_" + self.graph.uuid + "_" + self.uuid

    @property
    def address(self):

        keys = list(self.graph.pedges[self.uuid].keys())
        if len(keys) == 0:
            return "rootnode"
        else:
            k = keys[0]
            return self.graph.table[self.graph.pedges[k]].address + "_" + k

    def dump(self, f):
        if isinstance(f, str):
            with open(f, "wb") as fl:
                dill.dump(self, fl)
        else:
            dill.dump(self, f)

    def dumps(self):
        return dill.dumps(self)

    @classmethod
    def loads(cls, data: bytes):

        return dill.loads(data)

    @classmethod
    def load(cls, f):
        if isinstance(f, str):
            with open(f, "rb") as fl:
                return dill.load(fl)
        else:
            return dill.load(f)

    _metadata: dict = dict()

    @property
    def vis_config(self):
        return self.graph.vis_config_table[self.uuid]

    @vis_config.setter
    def vis_config(self, v):
        self.graph.vis_config_table[self.uuid] = v

    @vis_config.deleter
    def vis_config(self):
        del self.graph.vis_config_table[self.uuid]

    def toflow(self, x=0.0, y=0.0):

        data = {
            "id": self.uuid,
            "data": {
                "label": f'Node {self.uuid}'},
            "position": {"x": x, "y": y}
        }
        if self.is_scalar:
            data['data']['label'] = f'{self.uuid.split("_")[-1]} = {self.value}'

        return data





class BufferNode(Node):
    def __init__(self, *args, value=None, buffer=None, **kwargs):

        self._value = value
        super().__init__(*args, value=value, **kwargs)
        self.is_scalar = True
        self.buffer = buffer

    @property
    def value(self):
        return self.buffer[self._value]

    @value.setter
    def value(self, v):
        if isinstance(v, int):
            self._value = v
        else:
            self.buffer[self._value] = v


class AssetType(type):
    def __new__(mcs, name, bases=(object,), attrs=None):
        def new(cls, *args, **kwargs):
            copied = cls.__blueprint__.__copy__()
            return copied(*args, **kwargs)

        attrs["__new__"] = new
        return type(name, bases, attrs)


import zlib
from collections import ChainMap
from uuid import uuid4

__g__ = dict()
__s__ = dict()


def _dt(): t, b = datetime.datetime.now().isoformat().replace("T", " ").split(" ");return f'{t} {b[:8]}'


class GG:
    """
    >>> from mmcore.geom.parametric import Linear
    >>> import numpy as np
    
    >>> def fffff(ff, tbl):
    ... return Linear.from_two_points(Linear.from_two_points(tbl[__g__[ff].left.start],
    ...                                                     tbl[__g__[ff].left.end]).evaluate(__g__[ff].t),
    ...                              Linear.from_two_points(tbl[__g__[ff].right.start],
    ...                                                     tbl[__g__[ff].right.end]).evaluate(__g__[ff].t)
    ...                                                     )
    ...

    >>> g = GG()(**{"left": {"start": 11, "end": 12}, "right": {"start": 112, "end": 122}, "t": 0.5})
    >>> g(left=dict(start=9))(right=dict(start=95))(t=0.99)
GG({'left': GG({'start': 9, 'end': 12}) 1427244694, 'right': GG({'start': 95, 'end': 122}) 1644168957, 't': 0.99}) 1898519619
    >>> g0 = g.iget(3)
    >>> g0.select()
    >>> lst=np.random.random((1000, 3))
    >>> r1 = fffff(g0.uuid, lst)
    >>> r1
    Linear(x0=0.6889861472482821, y0=0.8909674342106825, z0=0.8568261420024046, a=0.2950900739957746, b=-0.8777288437677975, c=-0.7182311178474678)
    >>> g0.iget(0).select()
    >>> r2 = fffff(g0.uuid, lst)
    >>> r2
    Linear(x0=0.7859260789013973, y0=0.6308696186785778, z0=0.9266336005225433, a=-0.2076829377633831, b=-0.45996186338381867, c=-0.5864401684515325)
    >>> g0.rollback().select()
    Linear(x0=0.3932193728273126, y0=0.471692721813888, z0=0.44143473016794593, a=0.19407916587504676, b=0.015702012813977373, c=-0.1713995877225078)
    >>> g0.rollback().select()
    Linear(x0=0.6889861472482821, y0=0.8909674342106825, z0=0.8568261420024046, a=0.2950900739957746, b=-0.8777288437677975, c=-0.7182311178474678)

    Example 2:

    >>> import pickle
    >>> gg=b'\x80\x04\x95Z\x02\x00\x00\x00\x00\x00\x00\x8c\x12mmcore.base.params\x94\x8c\x02GG\x94\x93\x94)\x81\x94}\x94(\x8c\x04uuid\x94\x8c faf7d499b3c046faa01bd34ce0c32a27\x94\x8c\x06__maps\x94]\x94(}\x94(\x8c\x01t\x94G?\xbcj~\xf9\xdb"\xd1\x8c\tupdate_at\x94\x8c\x132023-09-05 07:06:54\x94u}\x94(\x8c\x01t\x94G?\xef\xae\x14z\xe1G\xae\x8c\tupdate_at\x94\x8c\x132023-09-05 07:05:00\x94u}\x94(\x8c\x05right\x94h\x02)\x81\x94}\x94(h\x05\x8c 1ce94b43e0734c4592f86c915ae41260\x94h\x07]\x94(}\x94(\x8c\x05start\x94K_h\x0f\x8c\x132023-09-05 07:05:00\x94u}\x94(h\x18Kp\x8c\x03end\x94Kzh\x0f\x8c\x132023-09-05 07:05:00\x94u}\x94eubh\x0f\x8c\x132023-09-05 07:05:00\x94u}\x94(\x8c\x04left\x94h\x02)\x81\x94}\x94(h\x05\x8c 0054a7299e4d40e69f8a2ac33a8c221c\x94h\x07]\x94(}\x94(h\x18K\th\x0f\x8c\x132023-09-05 07:05:00\x94u}\x94(h\x18K\x0bh\x1bK\x0ch\x0f\x8c\x132023-09-05 07:05:00\x94uh\x1deubh\x0f\x8c\x132023-09-05 07:05:00\x94u}\x94(h h\x02)\x81\x94}\x94(h\x05h#h\x07h$ubh\x12h\x02)\x81\x94}\x94(h\x05h\x15h\x07h\x16ubh\x0eG?\xe0\x00\x00\x00\x00\x00\x00h\x0f\x8c\x132023-09-05 07:05:00\x94uh\x1deub.'
    >>> p=pickle.loads(gg)
    >>> p.iget(0)
    Out[5]: GG({'t': 0.111, 'update_at': '2023-09-05 07:06:54'}) 725945272
    >>> p.iget(4)
    Out[6]: GG({'left': {'start': 9, 'end': 12, 'update_at': '2023-09-05 07:05:00'}, 'right': {'start': 95, 'end': 122, 'update_at': '2023-09-05 07:05:00'}, 't': 0.111, 'update_at': '2023-09-05 07:06:54'}) 3053531210
    >>> p.iget(5)
    Out[7]: GG({'left': {'start': 9, 'end': 12, 'update_at': '2023-09-05 07:05:00'}, 'right': {'start': 95, 'end': 122, 'update_at': '2023-09-05 07:05:00'}, 't': 0.111, 'update_at': '2023-09-05 07:06:54'}) 3053531210
    >>> p.i
    Out[8]: 5
    >>> p.rollback()
    Out[9]: GG({'left': {'start': 9, 'end': 12, 'update_at': '2023-09-05 07:05:00'}, 'right': {'start': 95, 'end': 122, 'update_at': '2023-09-05 07:05:00'}, 't': 0.99, 'update_at': '2023-09-05 07:05:00'}) 2316513311
    >>> p.rollout()
    Out[10]: GG({'left': {'start': 9, 'end': 12, 'update_at': '2023-09-05 07:05:00'}, 'right': {'start': 95, 'end': 122, 'update_at': '2023-09-05 07:05:00'}, 't': 0.111, 'update_at': '2023-09-05 07:06:54'}) 3053531210
    """
    _state = ChainMap()

    def __init__(self, uuid=None, select=True):
        super().__init__()

        self.uuid = uuid4().hex if uuid is None else uuid
        if select:
            self.select()

    def __setstate__(self, state):
        self.uuid = state["uuid"]

        self._state = ChainMap()
        __s__[self.uuid] = state["__maps"]
        self._state.maps = state["__maps"]
        __g__[self.uuid] = self

    def __getstate__(self):
        state = dict(uuid=self.uuid)

        state["__maps"] = __s__[self.uuid]
        return state

    def __getattr__(self, k):
        if k.startswith("_"):
            return super().__getattribute__(k)
        elif k in self._state:
            return self._state[k]
        else:
            return super().__getattribute__(k)

    # _state=ChainMap()
    def __hash__(self):
        return zlib.adler32(repr(self.todict()).encode())

    def __eq__(self, o):
        return hash(self) == hash(o)

    @property
    def i(self):
        return len(self._state.maps) - 1

    def rollout(self, steps=1, select=True):
        gg = self.iget(self.i + steps)
        if select:
            gg.select()
        return gg

    def iget(self, i):
        gg = GG(uuid=self.uuid, select=False)
        gg._state = ChainMap()

        gg._state.maps = __s__[self.uuid][:i + 1]

        return gg

    def select(self):
        __g__[self.uuid] = self

    def rollback(self, steps=1, select=True):
        def rb(o):
            g = GG(uuid=self.uuid)
            g._state = o._state.parents
            return g

        first = self
        for i in range(steps):
            first = rb(first)
        if select:
            first.select()
        return first

    def todict(self):
        dct = {}
        for k, v in dict(self._state).items():
            if isinstance(v, GG):
                dct[k] = v.todict()
            elif isinstance(v, ChainMap):

                dct[k] = dict(v)
            else:
                dct[k] = v
        return dct

    def __call__(self, **state):
        ss = {}
        for k in state.keys():
            if state[k] is not None:
                p = self._state.get(k)

                if isinstance(state[k], dict):
                    if isinstance(p, GG):

                        ss[k] = p(**state[k])
                    else:
                        ss[k] = GG()(**state[k])





                else:

                    if state[k] != p:
                        ss[k] = state[k]
        if len(ss) > 0:
            ss["update_at"] = _dt()
        G = GG(uuid=self.uuid)
        G._state = self._state.new_child(ss)
        __s__[self.uuid] = G._state.maps

        return G

    def __repr__(self):
        return f'{self.__class__.__qualname__}({self.todict()}) {self.__hash__()}'
