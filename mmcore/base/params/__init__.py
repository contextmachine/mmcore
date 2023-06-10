import dataclasses, json, abc, typing
import uuid as _uuid
from mmcore.base.basic import deep_merge, ALine, AGroup
from mmcore.base.models.gql import LineBasicMaterial

paramtable = dict()
relaytable = dict()
T = typing.TypeVar("T")

DEBUG_GRAPH = False


class ParamGraph:
    item_table: dict[str, typing.Any] = dict()
    relay_table: dict[str, dict] = dict()

    def __init__(self, *args, **kwargs):
        dict.__init__(self.item_table, *args, **kwargs)

    def __setitem__(self, k, v):
        self.item_table[k] = v

    def __getitem__(self, k):
        return self.item_table[k]

    def __delitem__(self, key):
        self.item_table[key].dispose()

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
        for i in self.item_table.values():
            if i.name.startswith(name):
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

    def get_relay(self, node, name):
        return self.item_table[self.relay_table[node.uuid][name]]

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

    def get_terms(self):
        nodes = []
        for node in self.item_table.values():
            if isinstance(node, TermParamGraphNode):
                nodes.append(node)
        return nodes

    def __repr__(self):
        return self.__class__.__name__ + f'({self.item_table.__repr__()}, {self.relay_table.__repr__()})'


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

    def __call__(self, data=None) -> typing.Optional[T]:

        if data is not None:
            if DEBUG_GRAPH:
                print(
                    f"[{self.__class__.__name__}] TERM PARAM UPDATE: node: {self.uuid}\n\t{self.graph.relay_table[self.uuid]} to {data}")
            self.graph.relay_table[self.uuid] = data

        return self.solve()

    def dispose(self):
        del self.graph.item_table[self.uuid]
        del self.graph.relay_table[self.uuid]

    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.name}, value={self.solve()}, uuid={self.uuid})'


@dataclasses.dataclass(unsafe_hash=True)
class ParamGraphNode:
    _params: dataclasses.InitVar['dict[str, typing.Union[ ParamGraphNode, typing.Any]]']
    name: typing.Optional[str] = None
    uuid: typing.Optional[str] = None
    resolver: typing.Optional[typing.Callable] = None

    def __post_init__(self, _params):
        self.graph = pgraph
        if self.uuid is None:
            self.uuid = _uuid.uuid4().hex
        if self.name is None:
            self.name = f'untitled{len(self.graph.get_from_startswith("untitled"))}'
        if self.resolver is None:
            self.resolver = lambda slf: slf
        self.graph.item_table[self.uuid] = self
        self.graph.relay_table[self.uuid] = dict()

        self(**_params)

    def subgraph(self):
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

    def get(self, k):
        try:
            self.__getitem__(k)
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
        return self.params.keys()

    def todict(self, no_attrs=False):
        dct = {}
        for k in self.keys():

            if isinstance(self.params[k], ParamGraphNode):
                dct[k] = self.graph.get_relay(self, k).todict(no_attrs)
            else:
                dct[k] = self.graph.get_relay(self, k)()

        if no_attrs:
            return dct
        return {
            "kind": self.__class__.__name__,
            "name": self.name,
            "uuid": self.uuid,
            "value": self.solve(),
            "params": dct
        }

    def solve(self):
        dct = {}
        for k in self.keys():

            if isinstance(self.params[k], ParamGraphNode):

                dct[k] = self.graph.get_relay(self, k).solve()
            else:
                dct[k] = self.graph.get_relay(self, k).solve()
        return self.resolver(dct)

    @property
    def params(self):
        return self.graph.get_relays(self)

    def __setitem__(self, key, value):

        self.graph.set_relay(self, key, value)

    def __getitem__(self, key):
        return self.graph.get_relay(self, key)

    def __getattr__(self, key):
        if key in self.keys():
            return self.graph.get_relay(self, key)()
        else:
            return getattr(self, key)

    def __call__(self, *args, **params) -> typing.Optional[T]:
        if len(args) > 0:
            params |= dict(zip(list(self.keys())[:len(args)], args))
        if params is not None:

            for k, v in params.items():
                self.graph.set_relay(self, k, v)

        return self.solve()

    def dispose(self):
        del self.graph.item_table[self.uuid]
        for k, v in self.graph.relay_table[self.uuid].items():
            self.graph.item_table[v].dispose()

        del self.graph.relay_table[self.uuid]
        del self


from mmcore.geom.parametric import Linear


def param_graph_node(params: dict):
    def inner(fn):
        def wrapper(kwargs):
            return fn(**kwargs)

        name = fn.__name__ + "_node"

        return ParamGraphNode(params, name=name, resolver=wrapper)

    return inner

