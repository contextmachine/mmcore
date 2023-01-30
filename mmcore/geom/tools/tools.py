#  Copyright (c) 2022. Computational Geometry, Digital Engineering and Optimizing your construction processe"


import copy
import json
import sys
import uuid

import numpy as np
from OCC.Core.Tesselator import ShapeTesselator
from OCC.Display.WebGl.threejs_renderer import spinning_cursor
from OCC.Extend.TopologyUtils import discretize_edge, discretize_wire, is_edge, is_wire

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

    # update spinning cursor
    sys.stdout.write("\r%s mesh shape, %i triangles     " % (next(spinning_cursor()),

                                                             tess.ObjGetTriangleCount()))
    sys.stdout.flush()
    return tess


def export_edge_data_to_json(edge_hash, point_set, scheme=None):
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
    edges_data = {
        "uuid": edge_hash,
        "type": "BufferGeometry",
        "data": {
            "attributes": {
                "position": {
                    "itemSize": 3,
                    "type": "Float32Array",
                    "array": points_coordinates}
                }
            },
        }

    sch |= edges_data

    return sch


def shaper(tess, export_edges, color=None, specular_color=None, shininess=None,
           transparency=None, line_color=None,
           line_width=None, scheme=None):
    if scheme is None:
        scheme = data_scheme
    sch = copy.deepcopy(scheme)
    shape_uuid = uuid.uuid4()

    shape_hash = "shp%s" % shape_uuid
    # tesselate

    # and also to JSON
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
    specular_color=(0.2, 0.2, 0.2),
    shininess=0.9,
    transparency=0.,
    line_color=(0, 0., 0.),
    line_width=1.,
    mesh_quality=1.,
    deflection=0.1,

    scheme=None,
    **kwargs
    ):
    # if the shape is an edge or a wire, use the related functions
    if scheme is None:
        scheme = data_scheme
    obj_hash = "edg%s" % uuid.uuid4()

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
                      color,
                      specular_color,
                      shininess,
                      transparency,
                      line_color,
                      line_width,

                      scheme=scheme)

    # store this edge hash

    return data
