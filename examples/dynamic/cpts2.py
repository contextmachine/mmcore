import copy

import itertools
import numpy as np
import rich
from dotenv import find_dotenv, load_dotenv

from mmcore.base import AMesh
from mmcore.base.components import Component
from mmcore.base.models.gql import MeshBasicMaterial
from mmcore.geom.materials import ColorRGB
from mmcore.geom.parametric.pipe import spline_pipe_mesh

dv = find_dotenv()
print(dv)
load_dotenv(dv)
from mmcore.geom.parametric import NurbsCurve
from mmcore.geom.parametric.algorithms import variable_line_offset_2d, offset_curve_3d_np
from mmcore.services.redis import connect, sets

rconn = connect.get_cloud_connection(

)
sets.rconn = rconn
from mmcore.base.sharedstate import serve
import shapely

from mmcore.geom.shapes import Shape, offset


def get_row(pth, i):
    for ptk in pth["points"].keys():
        r, c = ptk.replace("pt", "").split("_")
        if r == str(i):
            yield spl(pth["points"][ptk])


def extract_corners(pts):
    u_count, v_count = len(pts), len(pts[0])
    corners = []
    for i_ in range(v_count):
        corners.append(tuple(pts[0][i_]))
    for j_ in range(u_count):
        t = tuple(pts[j_][v_count - 1])
        if t not in corners:
            corners.append(t)
    for ir_ in range(v_count):
        t = tuple(pts[u_count - 1][v_count - 1 - ir_])
        if t not in corners:
            corners.append(t)
    for jr_ in range(u_count):
        t = tuple(pts[u_count - 1 - jr_][0])
        if t not in corners:
            corners.append(t)
    return corners


def make_nurbs_curve(points, degree):
    def nurbs_curve_wrapper(t):
        return NurbsCurve(points, degree=degree).evaluate(t)

    return nurbs_curve_wrapper


# ncurve, ncsetter,ncgetter=make_nurbs_curve([[0, 0, 200], [-47, -315, 20], [-785, -844, 2], [-1446.9613453661368, -1286, 20], [-969, -2316, 200]], degree=2)
def moff(crv, var_off=((0, 1), (0.3, 0.5), (0.6, -0.5), (1, 0)
                       )
         ):
    ss = intp.interp1d(*list(zip(*var_off)))

    def wrap(t):
        return offset_curve_3d(crv, ss(t).tolist())

    return wrap


from collections import deque
from mmcore.geom.parametric.algorithms import offset_curve_3d


def mesh_curve(crv, thickness=1.0, tess_range=100, tess_eps=0.01):
    dq = deque()
    offcrv = offset_curve_3d(crv, thickness)
    for t in np.linspace(0. + tess_eps, 1 - tess_eps, tess_range).tolist():
        # print(t)
        dq.append(crv(t))

        dq.appendleft(offcrv(t))
    return Shape(list(dq)).mesh_data



def uvshape(pts, thickness=1.0):
    u_count, v_count = len(pts), len(pts[0])
    corners = [list(l + (0,)) for l in
               list(shapely.Polygon(extract_corners(pts)).buffer(thickness / 2).boundary.coords)]
    holes = []
    for i in range(u_count):

        for j in range(v_count):

            if i < (u_count - 1) and j < (v_count - 1):
                ppp = [pts[i][j + 1], pts[i][j], pts[i + 1][j], pts[i + 1][j + 1]]

                pl = shapely.Polygon(ppp).buffer(-thickness / 2)
                holes.append([list(l + (0,)) for l in list(pl.boundary.coords)])
    return Shape(corners, holes=holes)



initial_path = {"points": {
    "pt0_0": {
        "x": -100.85264610432489,
        "y": -100.85264610432391,
        "z": 0.0
    },
    "pt0_1": {
        "x": -100.85264610432466,
        "y": -50.42632305216171,
        "z": 0.0
    },
    "pt0_2": {
        "x": -100.85264610432445,
        "y": 4.3655745685100554e-13,
        "z": 0.0
    },
    "pt0_3": {
        "x": -100.8526461043241,
        "y": 50.4263230521627,
        "z": 0.0
    },
    "pt0_4": {
        "x": -100.85264610432394,
        "y": 100.85264610432489,
        "z": 0.0
    },
    "pt1_0": {
        "x": -50.4263230521627,
        "y": -100.85264610432417,
        "z": 0.0
    },
    "pt1_1": {
        "x": -50.42632305216248,
        "y": -50.42632305216197,
        "z": 0.0
    },
    "pt1_2": {
        "x": -50.42632305216219,
        "y": 2.1827872842550277e-13,
        "z": 0.0
    },
    "pt1_3": {
        "x": -50.4263230521619,
        "y": 50.42632305216241,
        "z": 0.0
    },
    "pt1_4": {
        "x": -50.42632305216168,
        "y": 100.85264610432459,
        "z": 0.0
    },
    "pt2_0": {
        "x": -5.093170329928399e-13,
        "y": -100.85264610432438,
        "z": 0.0
    },
    "pt2_1": {
        "x": -2.9103830456733704e-13,
        "y": -50.42632305216226,
        "z": 0.0
    },
    "pt2_2": {
        "x": -4.3655745685100554e-13,
        "y": 7.275957614183426e-14,
        "z": 0.0
    },
    "pt2_3": {
        "x": -2.1827872842550277e-13,
        "y": 50.42632305216212,
        "z": 0.0
    },
    "pt2_4": {
        "x": 7.275957614183426e-14,
        "y": 100.85264610432438,
        "z": 0.0
    },
    "pt3_0": {
        "x": 50.42632305216168,
        "y": -100.85264610432466,
        "z": 0.0
    },
    "pt3_1": {
        "x": 50.42632305216197,
        "y": -50.42632305216248,
        "z": 0.0
    },
    "pt3_2": {
        "x": 50.426323052161756,
        "y": 7.275957614183426e-14,
        "z": 0.0
    },
    "pt3_3": {
        "x": 50.42632305216212,
        "y": 50.4263230521619,
        "z": 0.0
    },
    "pt3_4": {
        "x": 50.42632305216241,
        "y": 100.8526461043241,
        "z": 0.0
    },
    "pt4_0": {
        "x": 100.85264610432387,
        "y": -100.85264610432492,
        "z": 0.0
    },
    "pt4_1": {
        "x": 100.85264610432417,
        "y": -50.426323052162736,
        "z": 0.0
    },
    "pt4_2": {
        "x": 100.85264610432438,
        "y": -5.093170329928399e-13,
        "z": 0.0
    },
    "pt4_3": {
        "x": 100.85264610432466,
        "y": 50.42632305216168,
        "z": 0.0
    },
    "pt4_4": {
        "x": 100.85264610432489,
        "y": 100.85264610432387,
        "z": 0.0
    }
}}
vals = {
    "path": {
        "points": {
            "pt0": {
                "x": 9,
                "y": -3,
                "z": 0
            },
            "pt1": {
                "x": 13,
                "y": -3,
                "z": 0
            },
            "pt2": {
                "x": 13,
                "y": 0,
                "z": 0
            },
            "pt3": {
                "x": 10.980110404804643,
                "y": 0,
                "z": 0
            },
            "pt4": {
                "x": 10.933087881805529,
                "y": 4,
                "z": 0
            },
            "pt5": {
                "x": 16,
                "y": 4,
                "z": 0
            },
            "pt6": {
                "x": 16,
                "y": 6,
                "z": 0
            },
            "pt7": {
                "x": 7,
                "y": 7,
                "z": 0
            },
            "pt8": {
                "x": 7,
                "y": 4,
                "z": 0
            }
        }
    },
    "thickness": 1,
    "color": [
        50,
        120,
        190
    ],
    "offsets": [
        -0.1,
        -0.1,
        -0.6,
        -0.3,
        -0.1,
        -0.1,
        -0.4,
        -0.1,
        -0.6
    ]
}
spline_path = {"points": {"pt_0": {'x': 0, 'y': 0, 'z': 200},
                          "pt_1": {'x': -47, 'y': -315, 'z': 200},
                          "pt_2": {'x': -785, 'y': -844, 'z': 200},
                          "pt_3": {'x': -704, 'y': -1286, 'z': 200},
                          "pt_4": {'x': -969, 'y': -2316, 'z': 200}}}
def md_to_spline_mesh(md, uuid, name, color=(0, 0, 0), **kwargs):
    return AMesh(uuid=uuid, name=name if name is not None else uuid, geometry=md.to_buffer(),
                 material=MeshBasicMaterial(color=ColorRGB(*color).decimal), **kwargs)


def shape_to_mesh(pts, uuid, name, thickness=0.1, color=(0, 0, 0), **kwargs):
    shape = Shape(pts, holes=[offset(pts, -thickness)])
    shape.tessellate()
    # print(shape, pts)
    return md_to_spline_mesh(shape.mesh_data,
                             uuid=uuid,
                             name=name,
                             color=color,
                             **kwargs)


def objs_to_repr3d(uuid, name, fillet_helper_result, color=(0, 0, 0), thickness=1, controls=None, endpoint=None):
    r = fillet_helper_result.pop(0)
    obj = md_to_spline_mesh(points=[r.evaluate(l).tolist() for l in np.linspace(0, 1, 6)],
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
def spl(pt):
    return [pt['x'], pt['y'], pt['z']]


class GeomComp(Component):
    def __new__(cls, color=(0, 0, 0), **kwargs):
        return super().__new__(cls, color=color, **kwargs)

    def __call__(self, **kwargs):
        print(kwargs)

        super().__call__(**kwargs)

        self.__repr3d__()
        return self

class Spline(Component):
    path: dict
    color: tuple = (0, 0, 0)
    degree: int = 2
    thickness: float = 50

    def __new__(cls, path=copy.deepcopy(spline_path), thickness=40.0, degree: int = 3, color=(0, 0, 0),
                **kwargs):
        return super().__new__(cls, path=path, thickness=thickness, color=color, degree=degree, **kwargs)

    def __call__(self, **kwargs):
        # print(kwargs)

        super().__call__(**kwargs)

        self.__repr3d__()
        return self

    def __repr3d__(self):
        ncurve = self.curve()
        self._repr3d = spline_pipe_mesh(points=[ncurve(t).tolist() for t in np.linspace(0.01, 0.99, 100)],
                                        thickness=self.thickness, color=self.color, uuid=self.uuid, name=self.name,
                                        _endpoint=f"params/node/{self.param_node.uuid}",
                                        controls=self.param_node.todict())

        return self._repr3d

    def curve(self):
        nc = make_nurbs_curve(points=[[pt['x'], pt['y'], pt['z']] for pt in self.path['points'].values()],
                              degree=self.degree)
        return nc


class OffCurve(Component):
    path: dict
    offset: int = 100
    count: int = 5
    thickness: float = 40

    def __new__(cls, path, thickness=40.0, count: int = 5, color=(0, 0, 0),
                **kwargs):

        return super().__new__(cls, path=path, thickness=thickness, color=color, **kwargs)

    def __call__(self, **kwargs):
        # print(kwargs)

        super().__call__(**kwargs)

        self.__repr3d__()
        return self

    def __repr3d__(self):
        crvs = self.curves()
        crv1 = crvs[0]
        self._repr3d = spline_pipe_mesh(points=[crv1(t).tolist() for t in np.linspace(0.01, 0.99, 100)],
                                        thickness=self.thickness, color=self.color, uuid=self.uuid,
                                        name=self.name,
                                        _endpoint=f"params/node/{self.param_node.uuid}",
                                        controls=self.param_node.todict())

        for i, crv in enumerate(crvs[1:]):
            self._repr3d.__setattr__(f"part{i}",
                                     spline_pipe_mesh(points=[crv(t).tolist() for t in np.linspace(0.01, 0.99, 100)],
                                                      thickness=self.thickness, color=self.color,
                                                      uuid=self.uuid + str(i), name=self.name + str(i),
                                                      _endpoint=f"params/node/{self.param_node.uuid}",
                                                      controls=self.param_node.todict()))
        return self._repr3d

    def curves(self):

        crv = make_nurbs_curve(points=[[pt['x'], pt['y'], pt['z']] for pt in self.path['points'].values()],
                               degree=self.degree)

        cc2 = []
        for i in range(self.count):
            cc2.append(offset_curve_3d_np(crv, self.offset[i] * i))
        return cc2


class Grid(Component):

    path: dict
    color: tuple = (0, 0, 0)
    degree: int = 2
    thickness: float = 1.0

    def __new__(cls, path=initial_path, thickness=1.0, degree=2, color=(0, 0, 0),
                **kwargs):
        return super().__new__(cls, path=path, thickness=thickness, color=color, degree=degree, **kwargs)

    def __call__(self, **kwargs):
        super().__call__(**kwargs)
        self.__repr3d__()
        return self

    def path_to_points(self, u_count):
        return [list(get_row(self.path, i)) for i in range(u_count)]

    def __repr3d__(self):

        shape = uvshape(self.path_to_points(5), thickness=self.thickness)

        self._repr3d = md_to_spline_mesh(shape.mesh_data,
                                         name=self.name,
                                         uuid=self.uuid,
                                         color=self.color,
                                         _endpoint=f"params/node/{self.param_node.uuid}",
                                         controls=self.param_node.todict())

        return self._repr3d


class TwstCell(Component):
    path: dict
    offsets: list
    color: tuple = (0, 0, 0)
    color_fill: tuple = (160, 50, 70)
    _repr3d = None
    _hset = None
    thickness: float = 1.0

    def path_to_points(self):
        return [list(p.values()) for p in self.path['points'].values()]

    def __new__(cls, path=initial_path, thickness=1.0, color=(0, 0, 0), color_fill=(160, 50, 70), offsets=None,
                **kwargs):
        if offsets is None:
            *offsets, = itertools.repeat(0.5, len(path['points'].keys()))
        kk = copy.deepcopy(
            dict(path=copy.deepcopy(path), thickness=thickness, color=color, offsets=offsets, color_fill=color_fill,
                 **kwargs))

        if "uuid" in kwargs:
            _hset = sets.Hdict(f"cpts2:{kwargs.get('uuid')}")
            rich.print(dict(_hset))
            kk.update(dict(_hset))

            rich.print(kk)

        return super().__new__(cls, **kk)

    def __repr3d__(self):
        pts = self.path_to_points()
        *_shape, = variable_line_offset_2d(pts, self.offsets)
        shps = [pts, _shape]
        shps.sort(key=lambda x: shapely.Polygon(x).length, reverse=True)
        shape = Shape(shps[0], holes=[shps[1]])
        shape2 = Shape(shps[1])

        md = md_to_spline_mesh(shape.mesh_data,
                               name=self.name,
                               uuid=self.uuid,
                               color=self.color, _endpoint=f"params/node/{self.param_node.uuid}",
                               properties={"area": shape.shapely_polygon().area},
                               controls=self.param_node.todict())
        md.__setattr__("part2", md_to_spline_mesh(shape2.mesh_data,
                                                  name=self.name + "2",
                                                  uuid=self.uuid + "2",
                                                  color=self.color_fill,
                                                  _endpoint=f"params/node/{self.param_node.uuid}",
                                                  properties={"area": shape2.shapely_polygon().area},
                                                  controls=self.param_node.todict()))
        self._repr3d = md

        return self._repr3d

    def update_redis(self, kws):
        if self._hset is None:
            self._hset = sets.Hdict(f"cpts2:{self.uuid}")
        if len(self._hset) == 0:

            for k, v in self.param_node.todict().items():
                self._hset[k] = v

        for k, v in kws.items():
            if v is not None:
                if self._hset[k] != v:
                    self._hset[k] = v

    def __call__(self, **kwargs):
        super().__call__(**kwargs)
        pts = self.path_to_points()
        if len(self.offsets) > len(pts):
            self.offsets = self.offsets[pts[:len(pts)]]
        elif len(self.offsets) < len(pts):
            self.offsets.extend(list(itertools.repeat(0.5, len(pts) - len(self.offsets))))
        self.update_redis(kwargs)
        self.__repr3d__()

        return self


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


# spln = Spline(uuid="test_spline", name="test_spline", degree=3, path=copy.deepcopy(spline_path), thickness=30.1)

# s#pln = Spline(uuid="test_spline", name="test_spline", degree=2,path=copy.deepcopy(spline_path), thickness=30.1)
# spln_off = OffCurve(path=spln().path, uuid="test_spline_offset", name="test_spline_offset", degree=3,
#                    offset=[100, 100, 100, 100, 100],
#                    thickness=40,
#                    count=5)
#grid = Grid(uuid="test_grid", name="test_grid", path=copy.deepcopy(initial_path), thickness=0.1)

cell = TwstCell(uuid="test_cell", name="test_cell",
                path=vals['path'],
                offsets=vals['offsets'],
                color=vals['color'])

serve.start()
