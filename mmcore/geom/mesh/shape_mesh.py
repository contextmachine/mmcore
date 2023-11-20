from mmcore.geom.mesh import create_mesh_tuple
from mmcore.geom.shapes.shape import shape_earcut


def mesh_from_shapes(shps, cols, stats):
    for shp, c, props in zip(shps, cols, stats):
        pos, ixs, _ = shape_earcut(shp)

        yield create_mesh_tuple(dict(position=pos), ixs, c, extras=dict(properties=props))
