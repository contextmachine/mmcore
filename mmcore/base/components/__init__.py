import uuid as _uuid

from mmcore.base.params import ParamGraphNode


class Component:
    __exclude__ = ()
    def __new__(cls, *args, name=None, uuid=None, **params):
        self = super().__new__(cls)

        if uuid is None:
            self.uuid = _uuid.uuid4().hex
        self.uuid = uuid
        self.name = name

        dct = dict(zip(list(cls.__annotations__.keys())[:len(args)], args))
        params |= dct

        print(params)

        for k, v in params.items():
            if v is not None:
                setattr(self, k, v)
        prms = dict()
        for k in params.keys():
            if not k.startswith("_") and (k not in self.__exclude__):
                prms[k] = params[k]

        node = ParamGraphNode(prms, uuid=uuid, name=self.name, resolver=self)
        self.param_node = node
        node.solve()

        return node

    def __call__(self, **params):
        for k, p in params.items():
            if p is not None:
                setattr(self, k, p)
        return self

    @property
    def endpoint(self):
        return f"params/node/{self.param_node.uuid}"


a = ParamGraphNode(dict(x=1.0, y=2.0, z=3.0), name="A")
b = ParamGraphNode(dict(x=-1.0, y=-2.0, z=-3.0), name="B")
c = ParamGraphNode(dict(x=10.0, y=20.0, z=30.0), name="С")
d = ParamGraphNode(dict(x=-11.0, y=12.0, z=13.0), name="D")

from mmcore.geom.materials import ColorRGB

# render_lines.todict(no_attrs=True) will return the complete dictionary of parameters affecting the system.


# I use json.dumps(..., indent=3) to visually print out the whole dictionary, I could also use something like pprint,
# but it"s important to show that the parameters are parsed to the scolar simplest data types.
# We can decompose a system of any complexity into a parameter tree with prime, numbers, strings, boolean values, etc.


col = ColorRGB(70, 70, 70).decimal
col2 = ColorRGB(157, 75, 75).decimal

