#  Copyright (c) 2022. Computational Geometry, Digital Engineering and Optimizing your construction processe"


import copy
import json
import uuid
from enum import Enum
from typing import Any, Optional

import numpy as np
import pydantic
from OCC.Core.Tesselator import ShapeTesselator
from OCC.Extend.TopologyUtils import discretize_edge, discretize_wire, is_edge, is_wire
from mmcore.geom.materials import MeshPhongBasic
from pydantic.main import ModelMetaclass
from pydantic.types import UUID4, conlist


class ThreeTypes(str, Enum):
    Object3d = "Object"
    Group = "Group"
    Mesh = "Mesh"
    BufferGeometry = "BufferGeometry"


ThreeTransformMatrix = conlist(float | int, min_items=16, max_items=16)

zero_transform = ThreeTransformMatrix((1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1))


class ThreeObjectBase(pydantic.BaseModel, metaclass=ModelMetaclass):
    ...


class ThreeObject3d(pydantic.BaseModel):
    uuid: UUID4
    type: ThreeTypes = ThreeTypes.Object3d
    layers: int = 1
    geometry: Optional[UUID4 | None]
    material: Optional[UUID4 | None]
    matrix: ThreeTransformMatrix = zero_transform
    userData: dict[str, Any] = {}

    def __init__(self, **data: Any):
        super().__init__(**data)


class ThreeMesh(ThreeObject3d):
    type: ThreeTypes.Mesh


def object_scheme_advance(name: str,

                          __uuid: UUID4 | None,
                          geometry_uuid: UUID4,
                          material_uuid: UUID4,
                          matrix: ThreeTransformMatrix = zero_transform,
                          object_type: ThreeObject3d = ThreeObject3d,
                          userdata: dict = None):
    if userdata is None:
        userdata = {}
    if __uuid is None:
        __uuid = uuid.uuid4()

    return object_type(
        uuid=__uuid,
        name=name,
        matrix=matrix,
        geometry=geometry_uuid,
        material=material_uuid,
        userData=userdata
        )


data_scheme = {
    "uuid": None,
    "type": None,
    "data": {
        "attributes": {
            "position": {
                "itemSize": None,
                "type": None,
                "array": None}
            }
        },

    }

edges_data = {
    "uuid": None,
    "type": "BufferGeometry",
    "data": {
        "attributes": {
            "position": {
                "itemSize": 3,
                "type": "Float32Array",
                "array": []}
            }
        },
    }


def area_2d(pts: list | np.ndarray):
    """

    @param pts: array like [[ 1., 2.],[ 0., 0.],[ -1.33, 0.], ...],
        or collections of objects with __array__ method (np.asarray compatibility).


    @return:
    """
    lines = np.hstack([np.asarray(pts), np.roll(pts, -1, axis=0)])
    area = 0.5 * abs(sum(x1 * y2 - x2 * y1 for x1, y1, x2, y2 in lines))
    return area


def tesselation(shape, export_edges=False, mesh_quality=1.0):
    tess = ShapeTesselator(shape)
    tess.Compute(compute_edges=export_edges,
                 mesh_quality=mesh_quality,
                 parallel=True)
    return tess


def export_edge_data_to_json(edge_hash, point_set, scheme=None):
    global edges_data
    data = copy.deepcopy(edges_data)
    if scheme is None:
        scheme = data_scheme
    sch = copy.deepcopy(scheme)
    """ Export a set of points to a LineSegment buffergeometry
    """
    # first build the array of point coordinates
    # edges are built as follows:
    # points_coordinates  =[P0x, P0y, P0z, P1x, P1y, P1z, P2x, P2y, etc.]
    points_coordinates = []
    for point in point_set:
        for coord in point:
            points_coordinates.append(coord)
    # then build the dictionnary exported to json
    data |= {
        "uuid": edge_hash,
        "data": {
            "attributes": {
                "position": {
                    "array": points_coordinates}
                }
            },
        }

    sch |= edges_data

    return sch


def shaper(tess, frst, color, material=MeshPhongBasic, export_edges=False):
    # tesselate

    # and also to JSON
    shd = uuid.uuid4().__str__()
    shape_dict = json.loads(tess.ExportShapeToThreejsJSONString(shd))

    # draw edges if necessary
    try:
        del shape_dict["metadata"]
    except Exception as err:
        pass
    a, b = repr(frst.data.product).split("(")
    prms = b.replace(")", "").split(",")
    _, cls = a.split("=")
    mat = material(color=color)
    sch = data_scheme(cls,
                      guid=frst.data.guid,
                      mat=mat.data,
                      mat_guid=mat.uuid,
                      data=shape_dict["data"],
                      geometry_id=shd,
                      userdata=dict(
                          id=frst.data.id,
                          ifc_params=prms,
                          parent_id=frst.data.parent_id,
                          product=repr(frst.data.product),
                          brep_data=frst.data.geometry.id,
                          properties=dict(
                              type=cls,
                              context=frst.data.context)),
                      element_type=prms[3])
    sde = {"edges": []}
    # draw edges if necessary
    if export_edges:
        # export each edge to a single json
        # get number of edges
        nbr_edges = tess.ObjGetEdgeCount()
        for i_edge in range(nbr_edges):
            # after that, the file can be appended
            str_to_write = ''
            edge_point_set = []
            nbr_vertices = tess.ObjEdgeGetVertexCount(i_edge)
            for i_vert in range(nbr_vertices):
                edge_point_set.append(tess.GetEdgeVertex(i_edge, i_vert))
            # write to file
            edge_hash = "edg%s" % uuid.uuid4()
            sde["edges"].append(
                export_edge_data_to_json(edge_hash, edge_point_set, scheme=scheme))
        shape_dict["children"] = []
        shape_dict["children"].append(sde)
    return sch


def shaper(tess, export_edges, scheme=None):
    if scheme is None:
        scheme = data_scheme
    sch = copy.deepcopy(scheme)
    shape_uuid = uuid.uuid4()
    shape_dict = json.loads(tess.ExportShapeToThreejsJSONString(str(shape_uuid)))

    sde = {"edges": []}
    # draw edges if necessary
    if export_edges:
        # export each edge to a single json
        # get number of edges
        nbr_edges = tess.ObjGetEdgeCount()
        for i_edge in range(nbr_edges):
            # after that, the file can be appended
            str_to_write = ''
            edge_point_set = []
            nbr_vertices = tess.ObjEdgeGetVertexCount(i_edge)
            for i_vert in range(nbr_vertices):
                edge_point_set.append(tess.GetEdgeVertex(i_edge, i_vert))
            # write to file
            edge_hash = "edg%s" % uuid.uuid4()
            sde["edges"].append(
                export_edge_data_to_json(edge_hash, edge_point_set, scheme=scheme))
        shape_dict["children"] = []
        shape_dict["children"].append(sde)

    sch |= shape_dict
    return sch


def topo_converter(
    shape,
    *args,
    export_edges=False,
    color=(0.65, 0.65, 0.7),
    mesh_quality=1.,
    deflection=0.1,

    scheme=None,
    **kwargs
    ):
    # if the shape is an edge or a wire, use the related functions
    if scheme is None:
        scheme = data_scheme
    obj_hash = uuid.uuid4()

    if is_edge(shape):
        print("discretize an edge")
        pnts = discretize_edge(shape, deflection, *args, **kwargs)
        data = export_edge_data_to_json(obj_hash, pnts, scheme=scheme)

    elif is_wire(shape):
        print("discretize a wire")
        pnts = discretize_wire(shape)

        data = export_edge_data_to_json(obj_hash, pnts, scheme=scheme)

    else:

        data = shaper(tesselation(shape, export_edges, mesh_quality),
                      export_edges,
                      color)

    # store this edge hash

    return data
