# This is a sample Python script.
import dataclasses
import typing
import warnings
from collections import namedtuple
from dataclasses import dataclass

from ezdxf.enums import TextEntityAlignment
from itertools import zip_longest

import numpy as np

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import sys

import shapely
from shapely.geometry import mapping

import mmcore.geom.curves.bspline
from mmcore.geom.parametric import PlaneLinear, algorithms


@dataclass
class ShapeInterface:
    bounds: list[list[float]]
    holes: typing.Optional[list[list[list[float]]]] = None
    _poly = None

    def to_poly(self):
        if self._poly is None:
            self._poly = shapely.Polygon(self.bounds, self.holes)
        return self._poly

    def to_world(self, plane=None):
        if plane is not None:
            bounds = [plane.point_at(pt) for pt in self.bounds]
            holes = [[plane.point_at(pt) for pt in hole] for hole in self.holes]
            return ShapeInterface(bounds, holes=holes)

        return self

    def from_world(self, plane=None):
        if plane is not None:
            bounds = [plane.in_plane_coords(pt) for pt in self.bounds]
            holes = [[plane.in_plane_coords(pt) for pt in hole] for hole in self.holes]
            return ShapeInterface(bounds, holes=holes)
        return self

    def split(self, cont: 'Contour'):

        self.split_result = split2d(self.to_poly(), cont.poly)
        return self.split_result


contours_db = dict()


@dataclass
class ContourShape(ShapeInterface):

    def to_world(self, plane=None):
        if plane is not None:
            bounds = [plane.in_plane_coords(pt) for pt in self.bounds]
            holes = [[plane.in_plane_coords(pt) for pt in hole] for hole in self.holes]
            return ContourShape(bounds, holes=holes)
        return self


@dataclass
class Contour:
    shapes: list[ContourShape]
    plane: typing.Optional[PlaneLinear] = None

    def __post_init__(self):

        self.shapes = [shape.to_world(self.plane) for shape in self.shapes]
        if len(self.shapes) == 1:
            self.poly = self.shapes[0].to_poly()
        else:
            self.poly = shapely.multipolygons(shape.to_poly() for shape in self.shapes)

        # print(self.poly)

    def __eq__(self, other):

        return self.poly == other.poly

    @property
    def has_local_plane(self):
        return self.plane is not None


import ezdxf

HATCH_TYPES = {'inner': ezdxf.const.HATCH_STYLE_IGNORE, 'outer': ezdxf.const.HATCH_STYLE_OUTERMOST}


def poly_to_shapes(poly: typing.Union[shapely.Polygon, shapely.MultiPolygon]):
    crds = list(mapping(poly)['coordinates'])
    if isinstance(poly, shapely.MultiPolygon):
        return [ShapeInterface(crd[0], holes=crd[1:]) for crd in crds]
    else:
        return [ShapeInterface(crds[0], holes=crds[1:])]


def poly_to_points(poly: typing.Union[shapely.Polygon, shapely.MultiPolygon], ignore_holes=True):
    crds = list(mapping(poly)['coordinates'])
    if isinstance(poly, shapely.MultiPolygon):
        if ignore_holes:
            return [crd[0] for crd in crds]
        else:
            holes = []
            bounds = []
            for crd in crds:
                bounds.append(crd[0])

                holes.append(crd[1:])
            return bounds, holes

    else:
        if ignore_holes:
            return [crds[0]]
        else:
            holes = []
            bounds = []
            bounds.append(crds[0])
            holes.append(crds[1:])
            return bounds, holes


def poly_to_coords(poly: typing.Union[shapely.Polygon, shapely.MultiPolygon], ):
    return mapping(poly)['coordinates']


SplitResult = namedtuple("SplitResult", ['shapes', 'mask'])
import ezdxf

[(60.0, -130.0, 0.0), (52.0, 2.0, 0.0), (-8.0, -94.0, 0.0)]


# doc.styles.new("CXM Standard", dxfattribs={"font" : "Arial"})
@dataclass
class DxfText:
    text: str = 'A'
    high: float = 3.5
    style: str = "CXM"
    align: str = "MIDDLE_CENTER"
    layer: str = "CXM_Text"
    color: typing.Optional[int] = None

    def __post_init__(self):
        self.dxfattribs = dict(
            styles=self.style,
            layer=self.layer

        )
        if self.color is not None:
            self.dxfattribs['color'] = self.color

    def convert(self, origin, msp):
        text = msp.add_text(
            self.text,
            high=self.high,
            dxfattribs=self.dxfattribs
        )
        text.set_placement(
            origin,
            align=TextEntityAlignment[self.align]
        )

        return text


class DXFObjectColorPalette:
    lines: int = 0
    hatch: int = 254
    text: int = 33


dxfentities = []


@dataclass
class DxfPanel:
    shape: 'Panel'
    fill: typing.Optional[str] = None
    text: typing.Optional[DxfText] = None
    color: DXFObjectColorPalette = dataclasses.field(default_factory=DXFObjectColorPalette)

    def __post_init__(self):
        dxfentities.append(self)

    def convert(self, msp):

        # for shapes
        self.create_polylines(msp)

    def create_polylines(self, msp):
        # for shapes
        if self.shape.split_result.mask != 2:

            for sh in self.shape.split_result.shapes:
                hs = [pt[:2] for pt in sh.bounds]

                print(hs)
                msp.add_lwpolyline(hs)

    def create_text(self, origin, msp):
        if self.text is not None:
            return self.text.convert(origin, msp)

    def create_hatches(self, shapes, msp):
        if self.fill:
            hatch = msp.add_hatch(
                color=self.shape.color.hatch,
                dxfattribs={
                    "hatch_style": ezdxf.const.HATCH_STYLE_NESTED,
                    # 0 = nested: ezdxf.const.HATCH_STYLE_NESTED
                    # 1 = outer: ezdxf.const.HATCH_STYLE_OUTERMOST
                    # 2 = ignore: ezdxf.const.HATCH_STYLE_IGNORE
                },
            )

            for sh in shapes[1:]:
                hatch.paths.add_polyline_path(
                    [bmd[:2] for bmd in sh.bounds],
                    is_closed=True,
                    flags=ezdxf.const.BOUNDARY_PATH_EXTERNAL,
                )
                if sh.holes:
                    for hole in sh.holes:
                        hatch.paths.add_polyline_path(
                            [pt[:2] for pt in hole],
                            is_closed=True,
                            flags=ezdxf.const.BOUNDARY_PATH_OUTERMOST,
                        )


@dataclass
class Panel(ShapeInterface):
    color: DXFObjectColorPalette = dataclasses.field(default_factory=DXFObjectColorPalette)
    text: typing.Optional[DxfText] = None
    fill: typing.Optional[str] = None

    def __post_init__(self):
        self.bound_convertor = DxfPanel(
            self,
            fill=self.fill,
            text=self.text,
            color=self.color
        )

        self.hatch_shape = None
        if self.fill:
            if self.fill == "inner":

                self.hatch_shape = poly_to_shapes(self.hatch_hole())[0]

            elif self.fill == "outer":

                self.hatch_shape = poly_to_shapes(shapely.Polygon(self.bounds) - self.hatch_hole().buffer(-0.01))
            else:
                raise "Hatch Attension"
            self.hatch_convertor = DxfPanel(
                self.hatch_shape,
                fill=self.fill,
                color=self.color
            )

    def hatch_hole(self):
        if self.fill:
            a, b, c = self.shape.bounds

            return shapely.Polygon([algorithms.centroid([a, b]).tolist(), algorithms.centroid([b, c]).tolist(),
                                    algorithms.centroid([c, a]).tolist()])
        else:
            return []

    def split(self, cont: Contour):

        self.split_result = super().split(cont)
        self.hatch_split = []
        if self.hatch_shape:

            for sh in self.hatch_shape:

                r = mmcore.geom.curves.bspline.split(cont)
                if r.mask != 2:
                    self.hatch_split.extend(r.shapes)

    def convert(self, msp):
        # for shapes
        # lwpolyline = msp.add_lwpolyline([for pt in self.bounds]),

        # hatch = msp.add_hatch(color=self.color)

        # The first path has to set flag: 1 = external
        # flag const.BOUNDARY_PATH_POLYLINE is added (OR) automatically
        if self.split_result.mask != 2:
            self.bound_convertor.convert(msp)
            self.bound_convertor.create_text(list(self.to_poly().centroid.coords)[:2], msp)

            if self.fill is not None:
                self.hatch_convertor.create_hatches(self.hatch_split, msp)


def split2d(poly1, cont):
    if poly1.intersects(cont):

        if poly1.within(cont):
            # print("inside")

            return SplitResult(poly_to_shapes(poly1), 0)


        else:

            # print("intersects")
            poly3 = poly1.intersection(cont)

            return SplitResult(poly_to_shapes(poly3), 1)

    else:
        return SplitResult(poly_to_shapes(poly1), 2)


def travpt(fun):
    def wrapper(points, *args, **kwargs):
        i = 0

        def wrp(ptts):
            nonlocal i
            for pt in ptts:
                if all([isinstance(x, float) for x in pt]):
                    res = fun(pt, *args, **kwargs)

                    yield res
                else:

                    yield list(wrp(pt))

        return list(wrp(points))

    return wrapper


@travpt
def split3d_vec(point, plane=None):
    if plane:
        return algorithms.ray_plane_intersection(np.array(point), np.array([0.0, 0.0, 1.0]), plane).tolist()
    else:
        return point


def split3d(poly3, plane=None) -> list:
    if plane:
        return split3d_vec(list(mapping(poly3)['coordinates']), plane=plane)
    else:
        return poly_to_points(poly3)


ContourIntersectionResult = namedtuple('ContourIntersectionResult', ['points', 'mask', 'target_plane'])


class DxfPanelExporter:
    def __init__(self, path="test.dxf", version="R2018", setup=False):
        self.doc = ezdxf.new(dxfversion=version, setup=setup)
        self.msp = self.doc.modelspace()
        self.path = path
        self.doc.styles.new("CXM Standard", dxfattribs={"font": "Arial"})
        self.doc.layers.add("CXM_Contour", color=5)
        self.doc.layers.add("CXM_Text", color=251)

    def __enter__(self):
        return self

    def __call__(self, shapes: list[Panel], contour: Contour):
        for o in shapes:
            o.split(cont=contour)
            o.convert(self.msp)
        return self.doc, self.msp

    def __exit__(self, exc_type, *args):
        name, ext = self.path.split(".")
        print(exc_type, *args)
        self.doc.saveas(f'{name}_error.dxf')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
