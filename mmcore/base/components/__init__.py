import string
import uuid as _uuid

from mmcore.base import APoints, AGroup
from mmcore.base.params import ParamGraphNode


class Component:

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
                self.__dict__[k] = v
        prms = dict()
        for k in params.keys():
            if not k.startswith("_"):
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


class ControlPoint(Component):
    x: float = 0
    y: float = 0
    z: float = 0
    size: float = 0.5
    color: tuple = (157, 75, 75)

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def __call__(self, **kwargs):
        super().__call__(**kwargs)
        self.__repr3d__()
        return self

    def __repr3d__(self):
        self._repr3d = APoints(uuid=self.uuid,
                               name=self.name,
                               geometry=[self.x, self.y, self.z],
                               material=PointsMaterial(color=ColorRGB(*self.color).decimal, size=self.size),
                               _endpoint=self.endpoint,
                               controls=self.param_node.todict())
        return self._repr3d

    def __iter__(self):
        return iter([self.x, self.y, self.z])


class ControlPointList(Component):

    def __new__(cls, points=(), *args, **kwargs):
        node = super().__new__(cls, *args, **dict(zip(string.ascii_lowercase[:len(points) + 1], points)),
                               _points_keys=list(string.ascii_lowercase[:len(points)]), **kwargs)

        return node

    def __array__(self):
        return np.array([list(i) for i in self.points], dtype=float)

    def __iter__(self):
        return iter(self.points)

    def __len__(self):
        return self.points.__len__()

    def __getitem__(self, item):
        return self.points.__getitem__(item)

    @property
    def points(self):
        lst = []
        for k in self._points_keys:
            lst.append(getattr(self, k))
        return lst

    def __repr3d__(self):
        self._repr3d = AGroup(seq=self.points, uuid=self.uuid, name=self.name)

        return self._repr3d


a = ParamGraphNode(dict(x=1.0, y=2.0, z=3.0), name="A")
b = ParamGraphNode(dict(x=-1.0, y=-2.0, z=-3.0), name="B")
c = ParamGraphNode(dict(x=10.0, y=20.0, z=30.0), name="С")
d = ParamGraphNode(dict(x=-11.0, y=12.0, z=13.0), name="D")

from mmcore.geom.materials import ColorRGB

# render_lines.todict(no_attrs=True) will return the complete dictionary of parameters affecting the system.


# I use json.dumps(..., indent=3) to visually print out the whole dictionary, I could also use something like pprint,
# but it"s important to show that the parameters are parsed to the scolar simplest data types.
# We can decompose a system of any complexity into a parameter tree with prime, numbers, strings, boolean values, etc.


import numpy as np

col = ColorRGB(70, 70, 70).decimal
col2 = ColorRGB(157, 75, 75).decimal

from mmcore.base.models.gql import PointsMaterial
