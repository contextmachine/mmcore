__derived_attributes_table__ = dict()
__derived_dispatch_table__ = dict()


class DerivedMethod:
    def __init__(self, name, ref):
        super().__init__()

        self.__mmcore_ref__ = ref
        self._name = name

    def __call__(self, own):
        if own.__qualname__ not in __derived_attributes_table__:
            __derived_attributes_table__[own.__qualname__] = dict()
        __derived_attributes_table__[own.__qualname__][self._name] = self.__mmcore_ref__

        setattr(own, self._name, property(
            lambda slf: __derived_dispatch_table__[self.__mmcore_ref__](slf)))

        return own


class Derive:
    """
    >>> pts = [[0, 0, 0], [1, 0, 0], [1, 2, 0]]

    >>> @Derive
    ... def prev_attr_resolver(self):
    ...     return pts[self._pt - 1]

    >>> @Derive
    ... def next_attr_resolver(self):
    ...     try:
    ...         return pts[self._index + 1]
    ...     except IndexError:
    ...         return pts[(self._index  + 1) % len(pts)]

    >>> @Derive
    ... def prev_attr_resolver(self):
    ...    return pts[self._pt - 1]

    >>> @prev_attr_resolver.derived("prev_pt")
    ... @next_attr_resolver.derived("next_pt")
    ... class ExperimentalLinkedNode:
    ...       def __init__(self, index=1):
    ...           self._index = index







pts = [[0, 0, 0], [1, 0, 0], [1, 2, 0]]



from scipy.spatial import distance


@Derive
def all_tests_test(self):
    return str(self._test) * 2





@Derive
def dist_resolver(self):
    return distance.euclidean(prev_attr_resolver(self), next_attr_resolver(self))


@all_tests_test.derived("test")
@prev_attr_resolver.derived("prev_pt")
@next_attr_resolver.derived("next_pt")
class B:
    _test = 1
    dist = DerivedAttribute("dist_resolver")

    def __init__(self, pt=1):
        self._pt = 1


stop = False


def iii():
    global stop
    for i in np.linspace(0.1, 2 * np.pi, 200):
        time.sleep(0.01)
        pts[0] = [np.cosh(i), np.sinh(i), 1 / i]
    stop = True


th = threading.Thread(target=iii)
th.start()
j = -1
while True:
    if stop:
        break
    j += 1
    print(f'[loop: {j}] {o.dist}', flush=True, end="\r")

    """

    def __init__(self, fun):
        super().__init__()
        self.__mmcore_ref__ = fun.__name__
        __derived_dispatch_table__[self.__mmcore_ref__] = fun

    def __call__(self, *arga, **kwargs):
        return __derived_dispatch_table__[self.__mmcore_ref__](*arga, **kwargs)

    def derived(self, name):
        return DerivedMethod(name, self.__mmcore_ref__)


class DerivedAttribute:

    def __init__(self, ptr):
        self.__mmcore_ref__ = ptr

    def __set_name__(self, own, name):
        self._name = name
        if own.__qualname__ not in __derived_attributes_table__:
            __derived_attributes_table__[own.__qualname__] = dict()

        __derived_attributes_table__[own.__qualname__][self._name] = self.__mmcore_ref__

    def __get__(self, inst, own=None):

        if inst is not None:

            return __derived_dispatch_table__[__derived_attributes_table__[own.__qualname__][self._name]](inst)


        else:
            return __derived_dispatch_table__[__derived_attributes_table__[own.__qualname__][self._name]]


class DerivedProperty(DerivedAttribute):
    def __init__(self, fget):
        self.fget = fget
        self._name = fget.__name__
        __derived_dispatch_table__[fget.__qualname__] = fget

        super().__init__(fget.__qualname__)

    def __get__(self, inst, own=None):
        return super().__get__(inst, own=None)


graphs = dict()


class MmGraph:
    def __new__(cls, uuid=None, name="Base Objects Graph"):
        uuid = uuid if uuid is not None else _uuid.uuid4().hex
        if uuid in graphs.keys():
            return graphs[uuid]
        self = super().__new__(cls)
        self.name = name
        self.table = dict()
        self.relay = dict()
        self.prelay = dict()
        return self


ggraph = MmGraph(uuid="root")
import uuid as _uuid


class GraphLink:
    def __new__(cls, uuid, graph):
        obj = super().__new__(cls)
        obj.uuid, obj.graph = uuid, graph
        return obj

    def deref(self):
        return self.graph.table[self.uuid]

    def __get__(self, instance, own):
        return self.graph.table[self.uuid]


def get_from_full_uuid(uu):
    g, o = uu.split("@")
    return graphs[g].table[o]


def find_in_all(obj):
    o = obj.uuid
    dct = dict()

    for k, v in graphs.items():
        if o in v.table.keys():
            dct[k] = v.table[o]

    return dct


class MmBase:
    _uuid: str

    def __new__(cls, uuid=None, name="Base Object", graph=ggraph, **kwargs):
        if uuid is None:
            uuid = _uuid.uuid4().hex
        else:
            if uuid in graph.table.keys():
                return graph.table[uuid]
        self = super().__new__(cls)
        self._uuid = uuid
        self._name = name
        self._graph = graph
        self._graph.table[uuid] = self
        self._graph.relay[uuid] = dict()
        self._graph.prelay[uuid] = dict()
        self.update(**kwargs)

        return self

    @property
    def uuid(self):
        return self._uuid

    @uuid.setter
    def uuid(self, v):
        self._graph.table[v] = self._graph.table[self._uuid]
        self._graph.relay[v] = self._graph.relay[self._uuid]
        self._graph.prelay[v] = self._graph.prelay[self._uuid]
        for k, vv in self._graph.prelay[self._uuid].items():
            self._graph.relay[k][vv] = v
        del self._graph.table[self._uuid]
        del self._graph.relay[self._uuid]
        del self._graph.prelay[self._uuid]
        self._uuid = v

    def __call__(self, **kwargs):
        self.update(**kwargs)
        return self

    def __getattr__(self, item: str):
        if item.startswith("_"):
            return object.__getattribute__(self, item)
        elif item in self._graph.relay[self._uuid].keys():
            res = self._graph.relay[self._uuid][item]
            if isinstance(res, GraphLink):

                return res.deref()
            elif isinstance(res, str):
                if res in self._graph.table.keys():
                    return self._graph.table[res]

            return res
        else:
            return object.__getattribute__(self, item)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, MmBase):

                self._graph.relay[self._uuid][k] = v._uuid

                v._graph.prelay[v._uuid][self._uuid] = k


            else:
                self._graph.relay[self._uuid][k] = v
