import numpy as np
from multipledispatch import dispatch

from mmcore.geom.mesh import create_mesh_tuple
from mmcore.geom.shapes.shape import ShapeInterface, bounds_earcut, shape_earcut


@dispatch(ShapeInterface, tuple)
def to_mesh(shape: ShapeInterface, color=(0.3, 0.3, 0.3)):
    return to_mesh(shape, color, dict())


@dispatch(ShapeInterface, tuple, dict)
def to_mesh(shape: ShapeInterface, color=(0.3, 0.3, 0.3), props: dict = None):
    pos, ixs, _ = shape_earcut(shape)
    return create_mesh_tuple(dict(position=pos), ixs, color, extras=props)


@dispatch(np.ndarray, tuple, dict)
def to_mesh(shape: np.ndarray, color=(0.3, 0.3, 0.3), props: dict = None):
    pos, ixs, _ = bounds_earcut(shape)
    return create_mesh_tuple(dict(position=pos), ixs, color, extras=props)


@dispatch(np.ndarray, tuple)
def to_mesh(shape: np.ndarray, color=(0.3, 0.3, 0.3)):
    return to_mesh(shape, color, dict())


def mesh_from_bounds(bounds, color=(0.3, 0.3, 0), props=None):
    pos, ixs, _ = bounds_earcut(bounds)
    return create_mesh_tuple(dict(position=np.array(pos)), np.array(ixs), color, extras=dict(properties=props))
def mesh_from_shapes(shps, cols, stats):
    for shp, c, props in zip(shps, cols, stats):
        pos, ixs, _ = shape_earcut(shp)

        yield create_mesh_tuple(dict(position=pos), ixs, c, extras=dict(properties=props))
