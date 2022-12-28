#  Copyright (c) 2022. Computational Geometry, Digital Engineering and Optimizing your construction process"
import json
import os
import warnings
from json import JSONDecoder, JSONEncoder
from typing import Any

import mmcore.addons.comp
import numpy as np
import rhino3dm as rh


def rhino_transform_from_matrix(matrix):
    t = rh.Transform(1.0)

    i, j = np.asarray(matrix).shape
    for ii in range(i):
        for jj in range(j):
            setattr(t, f"M{ii}{jj}", float(np.asarray(matrix)[ii, jj]))

    return t


def DecodeToCommonObject(item):
    if item is None:
        return None
    elif isinstance(item, str):
        return DecodeToCommonObject(json.loads(item))
    elif isinstance(item, list):
        return [DecodeToCommonObject(x) for x in item]
    return rh.CommonObject.Decode(item)


def EncodeFromCommonObject(item):
    if item is None:
        return None
    elif isinstance(item, rh.CommonObject):
        return rh.CommonObject.Encode(item)
    elif isinstance(item, list):
        return [EncodeFromCommonObject(x) for x in item]
    else:
        pass


def encode_dict(item: Any):
    return rh.ArchivableDictionary.EncodeDict({"data": item})


def decode_dict(item: str):
    return rh.ArchivableDictionary.DecodeDict(item)["data"]


class RhinoEncoder(JSONEncoder):
    def default(self, o: Any) -> dict:
        return EncodeFromCommonObject(o)

    def encode(self, o: Any) -> str:
        return json.dumps(self.default(o))


class RhinoDecoder(JSONDecoder):
    def __init__(self, *, object_hook=None, parse_float=None, parse_int=None, parse_constant=None, strict=True,
                 object_pairs_hook=None):
        super().__init__(object_hook=object_hook, parse_float=parse_float, parse_int=parse_int,
                         parse_constant=parse_constant, strict=strict, object_pairs_hook=object_pairs_hook)

    def decode(self, s: str, *args, **kwargs) -> dict:
        dct = super().decode(s, *args, **kwargs)
        return DecodeToCommonObject(dct)


class RhinoBind:
    source_cls = None

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.source = self.source_cls(*args, **kwargs)


class RhDesc:
    def __set_name__(self, own, name):
        self.name = name

    def __get__(self, inst, ow=None):
        return getattr(inst.source, self.name)

    def __set__(self, inst, v):
        setattr(inst.source, self.name, v)


class RhinoPoint(RhinoBind):
    source_cls = rh.Point3d
    X = RhDesc()
    Y = RhDesc()
    Z = RhDesc()
    DistanceTo = RhDesc()

    @property
    def xyz(self):
        return self.X, self.Y, self.Z

    def __str__(self):
        return f"RhinoPoint({self.__array__()})"

    def __repr__(self):
        return f"RhinoPoint({self.__array__()})"

    def __list__(self):
        return list(self.xyz)

    def __array__(self):
        return np.array(self.xyz)

    def __sub__(self, other):
        return self.__class__(self.X - other.X, self.Y - other.Y, self.Z - other.Z)

    def __add__(self, other):
        return self.__class__(self.X + other.X, self.Y + other.Y, self.Z + other.Z)


class RhinoAxis(RhinoBind):
    """
    >>> ax = RhinoAxis(RhinoPoint(1, 2, 3), RhinoPoint(12, 2, 3))
    >>> ax.end
    RhinoPoint([12.  2.  3.])
    >>> ax.end.DistanceTo(ax.start.source)
    11.0

    """
    source_cls = rh.Line
    From = RhDesc()
    To = RhDesc()

    def __init__(self, start, end, **kwargs):
        super().__init__(start.source, end.source, **kwargs)

    @property
    def start(self):
        return RhinoPoint(self.From.X, self.From.Y, self.From.Z)

    @property
    def end(self):
        return RhinoPoint(self.To.X, self.To.Y, self.To.Z)


class RhinoCircle(RhinoBind):
    source_cls = rh.Circle
    radius = 1.0
    Center = RhDesc()
    Plane = RhDesc()


class RhinoRuledSurf(RhinoBind):
    source_cls = rh.NurbsSurface.CreateRuledSurface
    curve_a = None
    curve_b = None

    def __init__(self, curve_a, curve_b, **kwargs):
        super().__init__(curve_a.source.ToNurbsCurve(), curve_b.source.ToNurbsCurve(), **kwargs)


class RhinoBiCone:
    source_cls = rh.NurbsSurface.CreateRuledSurface
    radius_start = 1.0
    radius_end = 0.5
    point_start = RhinoPoint(22, 22, -11)
    point_end = RhinoPoint(1, 0, 1)

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__ |= kwargs

    @property
    def plane1(self):
        return rh.Plane(self.point_start.source,
                        rh.Vector3d(*(self.point_end - self.point_start).__array__()))

    @property
    def plane2(self):
        return rh.Plane(self.point_end.source,
                        rh.Vector3d(*(self.point_end - self.point_start).__array__()))

    @property
    def c1(self):
        t = rhino_transform_from_matrix(mmcore.conversions.compas.from_rhino_plane_transform(self.plane2).matrix)
        n = rh.Circle(self.radius_end).ToNurbsCurve()
        n.Transform(t)
        return n

    @property
    def c2(self):
        t = rhino_transform_from_matrix(mmcore.conversions.compas.from_rhino_plane_transform(self.plane1).matrix)
        n = rh.Circle(self.radius_end).ToNurbsCurve()
        n.Transform(t)
        return n

    @property
    def source(self):
        return self.source_cls(self.c1, self.c2)


def control_points_curve(points: list[list[float]] | np.ndarray, degree: int = 3):
    return rh.NurbsCurve.CreateControlPointCurve(list(map(lambda x: rh.Point3d(*x), points)),
                                                 degree=degree)


def random_control_points_curve(count=5, degree=3):
    return control_points_curve(np.random.random((count, 3)), degree=degree)


def rhino_crv_from_compas(nurbs_curves: list) -> list[rh.NurbsCurve]:
    """
    Convert list of compas-like Nurbs curves to Rhino Nurbs Curves
    :param nurbs_curves:
    :type
    :return:
    :rtype:

    """

    return list(map(lambda x: rh.NurbsCurve.Create(
        x.is_periodic,
        x.degree,
        list(map(lambda y: rh.Point3d(*y), x.points))), nurbs_curves))


def list_curves_to_polycurves(curves):
    poly = rh.PolyCurve()
    for curve in curves:
        poly.AppendSegment(curve)
    return poly


def polyline_from_pts(pts):
    polyline = rh.Polyline(0)
    for pt in pts:
        polyline.Add(*pt)
    if polyline.IsValid:
        return polyline
    else:
        warnings.warn("InValid Rhino Object")
        return polyline


def model_from_json_file(path, modelpath=None):
    model = rh.File3dm()
    pth, _ = path.split(".")
    with open(f"{path}", "r") as f:
        for lod in json.load(f):
            lod["archive3dm"] = 70
            model.Objects.Add(rh.GeometryBase.Decode(lod))
    if modelpath:
        model.Write(f"{modelpath}.3dm", 7)
    else:
        model.Write(f"{pth}.3dm", 7)


def model_from_dir_obj(directory):
    model = rh.File3dm()
    for path in os.scandir(directory):
        pth, _ = path.name.split(".")
        with open(f"{path}", "r") as f:
            for lod in json.load(f):
                lod["archive3dm"] = 70
                model.Objects.Add(rh.GeometryBase.Decode(lod))

    return model


def model_from_multijson_file(directory, model_path):
    model = model_from_dir_obj(directory)
    model.Write(f"{model_path}.3dm", 7)


def get_model_objects(path):
    rr = rh.File3dm().Read(path)
    # noinspection PyTypeChecker
    return list(rr.Objects)


def get_model_geometry(path):
    rr = rh.File3dm().Read(path)
    # noinspection PyTypeChecker
    return [o.Geometry for o in rr.Objects]


def get_model_geometry_from_buffer(buff: bytes):
    rr = rh.File3dm.FromByteArray(buff)
    # noinspection PyTypeChecker
    return [o.Geometry for o in rr.Objects]


def get_model_objects_from_buffer(buff: bytes):
    rr = rh.File3dm.FromByteArray(buff)
    # noinspection PyTypeChecker
    return [o for o in rr.Objects]


def get_model_attributes_from_buffer(buff: bytes):
    rr = rh.File3dm.FromByteArray(buff)
    # noinspection PyTypeChecker
    return [o.Attributes for o in rr.Objects]


# noinspection PyUnresolvedReferences
def get_model_attributes(path):
    rr = rh.File3dm().Read(path)
    return [o.Attributes for o in rr.Objects]

    def encode(self, o) -> str:
        dct = self.default(o)

        return json.dumps(dct)
