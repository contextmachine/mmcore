#  Copyright (c) 2022. Computational Geometry, Digital Engineering and Optimizing your construction process"
import json
import os
import uuid
import warnings
from json import JSONDecoder, JSONEncoder
from typing import Any

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
    if hasattr(item, "Encode"):
        return EncodeFromCommonObject(item.Encode())
    elif isinstance(item, dict):
        dd = {}
        for k, v in item.items():
            dd[k] = EncodeFromCommonObject(v)
    elif isinstance(item, list):
        return [EncodeFromCommonObject(x) for x in item]
    else:
        return item


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


def control_points_curve(points: list[list[float]] | np.ndarray, degree: int = 3):
    return rh.NurbsCurve.CreateControlPointCurve(list(map(lambda x: rh.Point3d(*x), points)),
                                                 degree=degree)


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


def model_from_dct(dct, path, modelpath=None):
    model = rh.File3dm()

    dct["archive3dm"] = 70
    model.Objects.Add(rh.GeometryBase.Decode(dct))
    if modelpath:
        model.Write(f"{modelpath}.3dm", 7)
    else:
        model.Write(f"{path}.3dm", 7)
    return model


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


def model_from_geometry_list(objects, model_path):
    model = rg.File3dm()
    [model.Objects.Add(obj) for obj in objects]
    model.Write(f"{model_path}.3dm", 7)
    return model


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


import rhino3dm as rg


class DPoint3d(rg.Point3d):
    def Encode(self):
        return {"array": [self.X, self.Y, self.Z], "type": self.__class__.__bases__[0].__name__}

    @classmethod
    def Decode(cls, dct: dict):
        return cls(*dct["array"])


class DVector3d(rg.Vector3d):
    def Encode(self):
        return {"array": [self.X, self.Y, self.Z], "type": self.__class__.__bases__[0].__name__}

    @classmethod
    def Decode(cls, dct: dict):
        return cls(*dct["array"])


class DPlane(rg.Plane):
    Origin: DPoint3d
    XAxis: DVector3d
    YAxis: DVector3d

    @property
    def Normal(self):
        return self.ZAxis

    def Encode(self):
        return {"array": [self.Origin, self.XAxis, self.YAxis], "type": self.__class__.__bases__[0].__name__}

    @classmethod
    def Decode(cls, dct: dict):
        return cls(*dct["array"])


import subprocess


def start_with_arg():
    proc = subprocess.Popen(["-nosplash"], executable="/Applications/RhinoWIP.app/Contents/MacOS/Rhinoceros")

    proc.send_signal(subprocess.signal.SIGKILL)


class EpsTuple(tuple):
    """
    Tuple with epsilon equal metric.
    It can be useful in geometric transformations with floating point e.g. tessellation .
    --------------------------------------------------------------------------------------------------------------------

    >>> TuplePt.eps = 0.001
    >>> TuplePt((1.0,1.0,1.0))==TuplePt((0.9999,0.9999,0.9999))
    True
    >>> TuplePt((1.0,1.0,1.0))==TuplePt((1.0001,0.9999,0.9999))
    True
    >>> TuplePt((1.0,1.0,1.0))==TuplePt((1.1111,0.9999,0.9999))
    False
    """
    _eps = 0.001

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    @property
    def eps(self): return self._eps

    @eps.setter
    def eps(self, v: float): self._eps = v

    def __eq__(self, other):
        return np.allclose(np.abs((np.asarray(self) - np.asarray(other))) <= self.eps, True)

    def __str__(self):
        return f"{super().__str__()} eps={self.eps}"


Tuple2dVec = TupleUV = Tuple2dPt = EpsTuple[float, float]
Tuple3dPt = Tuple3dVec = EpsTuple[float, float, float]
Tuple4dPt = TupleWPt = EpsTuple[float, float, float, float]


def get_mesh_uv(msh) -> list[TupleUV]: return [TupleUV((i.X, i.Y)) for i in list(msh.TextureCoordinates)]


def get_np_mesh_uv(msh) -> np.ndarray: return np.asarray(get_mesh_uv(msh))


def get_mesh_vertices(msh) -> list[tuple]: return [(i.X, i.Y, i.Z) for i in list(msh.Vertices)]


def get_np_mesh_vertices(msh) -> np.ndarray: return np.asarray(get_mesh_vertices(msh))


def get_mesh_normals(msh) -> list[Tuple3dVec]: return [Tuple3dPt((i.X, i.Y, i.Z)) for i in list(msh.Normals)]


def get_np_mesh_normals(msh) -> np.ndarray: return np.asarray(get_mesh_normals(msh))


# noinspection PyTypeChecker
def create_model_with_items(*items, return_uuids=False) -> rh.File3dm | tuple[rh.File3dm | list[Any]]:
    model3dm = rh.File3dm()
    uuids = [model3dm.Objects.Add(item) for item in items]
    return model3dm if not return_uuids else model3dm, uuids


# noinspection PyTypeChecker
def write_model_with_items(path, *items, return_uuids=False, **kwargs) -> bool | tuple[bool | list[Any]]:
    model3dm, uuids = create_model_with_items(*items, return_uuids=True)
    res = model3dm.Write(path, **kwargs)
    return res if not return_uuids else res, uuids


from mmcore.baseitems import Matchable


class MatchJson(JSONEncoder):
    def default(self, o: Matchable) -> dict[str, Any]:
        return o.__getstate__()


class GeometryAttribute(Matchable):
    __match_args__ = "type", "itemSize", "array", "normalized"


class SurfAttributes(Matchable):
    __match_args__ = "position", "normal", "uv"


class SurfData(Matchable):
    __match_args__ = "index", "attributes"


class GeometriesItem(Matchable):
    __match_args__ = "index", "attributes"


def create_buffer(indices, verts, normals, uv):
    return {
        "uuid": uuid.uuid4().__str__(),
        "type": "BufferGeometry",
        "data": {
            "attributes": {
                "normal": {
                    "array": np.asarray(normals).flatten().tolist(),
                    "itemSize": 3,
                    "type": "Float32Array",
                    "normalized": False
                    },
                "position": {
                    "array": np.asarray(verts, dtype=float).flatten().tolist(),
                    "itemSize": 3,
                    "type": "Float32Array",
                    "normalized": False
                    },
                "uv": {
                    'itemSize': 2,
                    "array": np.asarray(uv, dtype=float).flatten().tolist(),
                    "type": "Float32Array",
                    "normalized": False
                    }
                },
            "index": dict(type='Uint16Array',
                          array=np.asarray(indices).flatten().tolist())
            }
        }


def get_triangle_mesh_only(msh):
    f = msh.Faces
    f.ConvertQuadsToTriangles()
    vertss = get_mesh_vertices(msh)
    normals = get_mesh_normals(msh)
    llst = []
    vv = []
    for i in range(f.TriangleCount):
        pts = f.GetFaceVertices(i)
        lst = []
        for i in range(3):
            pt = pts[i + 1]
            vv.append((pt.X, pt.Y, pt.Z))
            lst.append(vertss.index((pt.X, pt.Y, pt.Z)))

        llst.append(lst)

    return llst, vertss, normals


def mesh_to_buffer(input_mesh) -> dict:
    indices, verts, normals = get_triangle_mesh_only(input_mesh)
    uv = get_np_mesh_uv(input_mesh)

    return create_buffer(indices, verts, normals, uv)


def mesh_to_buffer_geometry(input_mesh) -> dict:
    (indices, position, normals), uv = get_triangle_mesh_only(input_mesh), get_np_mesh_uv(input_mesh)
    return dict(indices=indices, position=position, normals=normals, uv=uv)


def mesh_to_buffer_mesh(mesh,

                        material,
                        name="TestMesh",

                        cast_shadow=True,
                        receive_shadow=True,
                        layers=1,
                        matrix=(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)):
    geom = mesh_to_buffer(mesh)
    return {
        "metadata": {
            "version": 4.5,
            "type": "Object",
            "generator": "Object3D.toJSON"
            },
        "geometries": [geom],
        "materials": [material.data],
        "object": {
            "uuid": uuid.uuid4().__str__(),
            "type": "Mesh",
            "name": name,
            "castShadow": cast_shadow,
            "receiveShadow": receive_shadow,
            "layers": layers,
            "matrix": matrix,
            "geometry": geom["uuid"],
            "material": material.uuid
            }

        }


def create_root(rootobj):
    return {
        "metadata": {
            "version": 4.5,
            "type": "Object",
            "generator": "Object3D.toJSON"
            },
        "geometries": [],
        "materials": [],
        "object": rootobj
        }


def obj_notation_from_mesh(name, geometry_uuid, material_uuid, userdata={},
                           matrix=(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1), uid=None):
    if uid is None:
        uid = uuid.uuid4().__str__()
    return {
        'uuid': uid,
        'type': 'Mesh',
        'name': name,
        'castShadow': True,
        'receiveShadow': True,
        'userData': userdata,
        'layers': 1,
        'matrix': matrix,
        'geometry': geometry_uuid,
        'material': material_uuid
        }


def group_notation_from_mesh(name, userdata={},
                             matrix=(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1), children=[], uid=None):
    if uid is None:
        uid = uuid.uuid4().__str__()
    return {
        'uuid': uid,
        'type': 'Group',
        'name': name,
        'castShadow': True,
        'receiveShadow': True,
        'userData': userdata,
        'layers': 1,
        'matrix': matrix,
        'children': children
        }


def tree_obj_from_mesh_obj(obj):
    return obj_notation_from_mesh(obj.name, geometry_uuid=obj.geometry, material_uuid=obj.material,
                                  userdata=obj.userData, matrix=obj.matrix)
