import copy
import itertools

import rich
from dotenv import find_dotenv, load_dotenv

from mmcore.base import AMesh
from mmcore.base.components import Component
from mmcore.base.models.gql import MeshBasicMaterial
from mmcore.geom.materials import ColorRGB

dv = find_dotenv()
print(dv)
load_dotenv(dv)
from mmcore.geom.parametric.algorithms import variable_line_offset_2d
from mmcore.services.redis import connect, sets

rconn = connect.get_cloud_connection(

)
sets.rconn = rconn
from mmcore.base.sharedstate import serve
import shapely

from mmcore.geom.shapes import Shape


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

def md_to_spline_mesh(md, uuid, name, color=(0, 0, 0), **kwargs):
    return AMesh(uuid=uuid, name=name if name is not None else uuid, geometry=md.to_buffer(),
                 material=MeshBasicMaterial(color=ColorRGB(*color).decimal), **kwargs)


def spl(pt):
    return [pt['x'], pt['y'], pt['z']]


class Spline(Component):
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

        self._repr3d = md_to_spline_mesh(shape.mesh_data, name=self.name, uuid=self.uuid, color=self.color,
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


spln = Spline(uuid="test_spline", name="test_spline", path=copy.deepcopy(initial_path))

cell = TwstCell(uuid="test_cell", name="test_cell",
                path=vals['path'],
                offsets=vals['offsets'],
                color=vals['color'])

serve.start()
