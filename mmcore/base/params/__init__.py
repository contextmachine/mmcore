import dataclasses
import typing
import uuid as _uuid

from mmcore.base.basic import DictSchema
from mmcore.base.registry import AGraph

paramtable = dict()
relaytable = dict()
DEBUG_GRAPH = False


import string


class ParamGraph(AGraph):
    _scenegraph = None

    def __init__(self):
        super().__init__()
        self.scene_relay_table = dict()

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


pgraph = ParamGraph()


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

        self.graph.item_table[self.uuid] = self
        self.graph.relay_table[self.uuid] = data

    def solve(self):
        return self.graph.relay_table[self.uuid]

    def __call__(self, data=None) :

        if data is not None:
            if DEBUG_GRAPH:
                print(
                    f"[{self.__class__.__name__}] TERM PARAM UPDATE: node: {self.uuid}\n\t{self.graph.relay_table[self.uuid]} to {data}")
            self.graph.relay_table[self.uuid] = data

        return self.solve()

    def dispose(self):
        del self.graph.item_table[self.uuid]
        del self.graph.relay_table[self.uuid]

    def del_from_graph(self, graph):
        graph.item_table.pop(self.uuid)
        graph.relay_table.pop(self.uuid)

    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.name}, value={self.solve()}, uuid={self.uuid})'

    @property
    def index(self):
        return self.graph.item_table.index_from_key(self.uuid)


@dataclasses.dataclass(unsafe_hash=True)
class ParamGraphNode:
    _params: dataclasses.InitVar['dict[str, typing.Union[ ParamGraphNode, typing.Any]]']
    name: typing.Optional[str] = None
    uuid: typing.Optional[str] = None
    resolver: typing.Optional[typing.Callable] = None

    @property
    def index(self):
        return self.graph.item_table.index_from_key(self.uuid)

    def __post_init__(self, _params):
        self.graph = pgraph
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



    def __call__(self, *args, **params) :
        if len(args) > 0:
            params |= dict(zip(list(self.keys())[:len(args)], args))
        if params is not None:

            for k, v in params.items():
                self.graph.set_relay(self, k, v)
            self.__dct_schema__ = DictSchema(self.todict(no_attrs=True))
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
