import numpy as np

from mmcore.base.components import Component
from mmcore.base.sharedstate import serve
from mmcore.geom.parametric import Circle, Linear, fillet_helper, fillet_lines
from mmcore.geom.parametric.pipe import spline_pipe_mesh


# cgraph = ComponentGraph()

class Spline(Component):
    path: dict
    color: tuple = (0, 0, 0)

    def __new__(cls, path={"points": {"a": {'x': 0, 'y': 0, 'z': 200},
                                      "b": {'x': -47, 'y': -315, 'z': 200},
                                      "c": {'x': -785, 'y': -844, 'z': 200},
                                      "d": {'x': -704, 'y': -1286, 'z': 200},
                                      "e": {'x': -969, 'y': -2316, 'z': 200}}}, thickness=4.0, color=(0, 0, 0),
                **kwargs):
        return super().__new__(cls, path=path, thickness=thickness, color=color, **kwargs)

    def __call__(self, **kwargs):
        super().__call__(**kwargs)
        self.__repr3d__()
        return self

    def __repr3d__(self):
        self._repr3d = spline_pipe_mesh(points=[[pt['x'], pt['y'], pt['z']] for pt in self.path['points'].values()],
                                        thickness=self.thickness, color=self.color, uuid=self.uuid, name=self.name,
                                        _endpoint=f"params/node/{self.param_node.uuid}",
                                        controls=self.param_node.todict())
        return self._repr3d


class Spline2(Component):
    points: list
    color: tuple = (0, 0, 0)

    def __new__(cls, points=[{'x': 0, 'y': 0, 'z': 0},
                             {'x': -47, 'y': -315, 'z': 0},
                             {'x': -785, 'y': -844, 'z': 0},
                             {'x': -704, 'y': -1286, 'z': 0},
                             {'x': -969, 'y': -2316, 'z': 0}], thickness=4.0, color=(0, 0, 0), **kwargs):
        return super().__new__(cls, points=points, thickness=thickness, color=color, **kwargs)

    def __call__(self, **kwargs):
        super().__call__(**kwargs)
        self.__repr3d__()
        return self

    def __repr3d__(self):
        self._repr3d = spline_pipe_mesh(points=[[pt['x'], pt['y'], pt['z']] for pt in self.points],
                                        thickness=self.thickness, color=self.color, uuid=self.uuid, name=self.name,
                                        _endpoint=f"params/node/{self.param_node.uuid}",
                                        controls=self.param_node.todict())
        return self._repr3d


class TangObj(Component):
    path: dict
    radius = 20
    color: tuple = (0, 0, 0)

    def __new__(cls, path={"points": {"a": {'x': -90, 'y': 190, 'z': 0},
                                      "b": {'x': 10, 'y': 110, 'z': 0},
                                      "c": {'x': -120, 'y': -140, 'z': 0}}}, radius=20, thickness=0.1, color=(0, 0, 0),
                **kwargs):
        return super().__new__(cls, path=path, thickness=thickness, color=color, radius=radius, **kwargs)

    def __call__(self, **kwargs):
        super().__call__(**kwargs)
        self.__repr3d__()
        return self

    def __repr3d__(self):
        self._repr3d = objs_to_repr3d(uuid=self.uuid, name=self.name, fillet_helper_result=fillet_helper(
            fillet_lines([[pt['x'], pt['y']] for pt in self.path['points'].values()], self.radius)),
                                      thickness=self.thickness, color=self.color, controls=self.param_node.todict(),
                                      endpoint=f"params/node/{self.param_node.uuid}")
        return self._repr3d


# @param_graph_node_native
# def spline(path=path, color=(0, 0, 0)):
#   return ALine(geometry=path.tessellate(), material=gql.LineBasicMaterial(color=ColorRGB(*color).decimal, uuid="spline-material"), uuid="spline",name="spline")


def objs_to_repr3d(uuid, name, fillet_helper_result, color=(0, 0, 0), thickness=1, controls=None, endpoint=None):
    r = fillet_helper_result.pop(0)
    obj = spline_pipe_mesh(points=[r.evaluate(l).tolist() for l in np.linspace(0, 1, 6)],
                           degree=1,
                           thickness=thickness, color=color, uuid=uuid, name=name,
                           _endpoint=endpoint,
                           controls=controls)
    for i, item in enumerate(fillet_helper_result):

        if isinstance(item, Linear):

            obj.__setattr__('part' + f'{i}',
                            spline_pipe_mesh(points=[item.evaluate(l).tolist() for l in np.linspace(0, 1, 6)],
                                             degree=1,

                                             thickness=thickness, color=color, uuid=uuid + str(i), name=name + str(i),
                                             _endpoint=endpoint,
                                             controls=controls))

        elif isinstance(item, Circle):

            obj.__setattr__('part' + f'{i}',
                            spline_pipe_mesh(points=[item.evaluate(l).tolist() for l in np.linspace(0, 1, 32)],
                                             degree=1,
                                             u_count=64,
                                             thickness=thickness, color=color, uuid=uuid + str(i), name=name + str(i),
                                             _endpoint=endpoint,
                                             controls=controls))

    return obj

serve.start()
spl = Spline(uuid="test_spline", name="test_spline")
spl2 = Spline2(uuid="test_spline2", name="test_spline2")
ttt = TangObj(uuid="fillet_object", name="fillet_object")
