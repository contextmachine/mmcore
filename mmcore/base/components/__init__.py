import uuid as _uuid
from abc import abstractmethod

import randomname

from mmcore.base import AGeom
from mmcore.base.params import ParamGraphNode


class Component:
    name: str = None
    uuid: str = None
    __exclude__ = ["uuid"]

    def __new__(cls, *args, name=None, uuid=None, **params):
        self = super().__new__(cls)

        if uuid is None:
            uuid = _uuid.uuid4().hex
            if name is None:
                name = randomname.get_name(sep="_")
        self.uuid = uuid
        self.name = name


        dct = dict(zip(list(cls.__annotations__.keys())[:len(args)], args))
        params |= dct

        #print(params)
        self.__params__ = params
        for k, v in params.items():
            if v is not None:

                setattr(self, k, v)

        prms = dict(name=name)
        for k in params.keys():
            if not k.startswith("_") and (k not in self.__exclude__):
                prms[k] = params[k]

        node = ParamGraphNode(prms, uuid=self.uuid, name=self.name, resolver=self)
        self.param_node = node
        node()
        return node

    def __call__(self, **params):
        for k, p in params.items():
            if p is not None:
                setattr(self, k, p)
        return self

    @property
    def endpoint(self):
        return f"params/node/{self.param_node.uuid}"

    def __hash__(self):
        return self.param_node.__hash__()

    def __setstate__(self, state):
        for k,v in state.items():
            setattr(self,k,v)


    def __getstate__(self):
        dct = dict(self.__params__)
        dct |= dict(
            uuid=self.uuid,
            name=self.name

        )
        return dct


class ComponentProxy(Component):

    def __class_getitem__(cls, item):
        comp, item = item

        def call(slf, **kws):
            cls.__call__(slf, **kws)
            dct = {}
            for k in item.__annotations__:
                if kws.get(k) is None:
                    dct[k] = slf.__getattribute__(k)
                else:
                    dct[k] = kws[k]
            slf.proxy = item(**kws)

            return comp.__call__(slf, **kws)

        def __getattr__(slf, k):
            if k == "proxy":
                return cls.__getattribute__(slf, "proxy")
            elif hasattr(slf.proxy, k):
                return getattr(slf.proxy, k)
            else:
                return cls.__getattribute__(slf, k)

        anno = dict(cls.__annotations__)
        anno |= item.__annotations__
        return type(f'{cls.__name__}[{item.__name__}]', (comp,), {
            "__qualname__": f'{cls.__name__}[{item.__name__}]',
            "__getattr__": __getattr__,
            "__call__": call,
            "__annotations__": anno
        })


# a = ParamGraphNode(dict(x=1.0, y=2.0, z=3.0), name="A")
# b = ParamGraphNode(dict(x=-1.0, y=-2.0, z=-3.0), name="B")
# c = ParamGraphNode(dict(x=10.0, y=20.0, z=30.0), name="С")
# d = ParamGraphNode(dict(x=-11.0, y=12.0, z=13.0), name="D")

from mmcore.geom.materials import ColorRGB

# render_lines.todict(no_attrs=True) will return the complete dictionary of parameters affecting the system.


# I use json.dumps(..., indent=3) to visually print out the whole dictionary, I could also use something like pprint,
# but it"s important to show that the parameters are parsed to the scolar simplest data types.
# We can decompose a system of any complexity into a parameter tree with prime, numbers, strings, boolean values, etc.


col = ColorRGB(70, 70, 70).decimal
col2 = ColorRGB(157, 75, 75).decimal


class GeometryComponent(Component):
    color = (100, 100, 100)

    def __new__(cls, *args, color=None, **kwargs):

        if color is None:
            color = cls.color

        return super().__new__(cls, *args, color=color, **kwargs)

    @abstractmethod
    def solve(self) -> AGeom:
        ...

    def __repr3d__(self) -> AGeom:
        self._repr3d = self.solve()
        self._repr3d._endpoint = f"params/node/{self.param_node.uuid}"
        self._repr3d.controls = self.param_node.todict()

        return self._repr3d

    def root(self):
        if self._repr3d:
            return self._repr3d.root()
        else:
            return self.__repr3d__().root()
