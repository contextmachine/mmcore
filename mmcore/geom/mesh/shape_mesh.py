import numpy as np
from multipledispatch import dispatch

from mmcore.geom.mesh.mesh_tuple import create_mesh_tuple
from mmcore.geom.shapes.shape import ShapeInterface, bounds_earcut, shape_earcut,bounds_holes_earcut


@dispatch(ShapeInterface, tuple)
def to_mesh(shape: ShapeInterface, color=(0.3, 0.3, 0.3)):
    """
    Convert the given shape into a mesh tuple.

    :param shape: The shape to convert.
    :type shape: ShapeInterface
    :param color: The color of the mesh. Defaults to (0.3, 0.3, 0.3).
    :type color: tuple
    :return: The mesh tuple.
    :rtype: tuple
    """
    pos, ixs, _ = shape_earcut(shape)
    return create_mesh_tuple(dict(position=pos), ixs, color, extras=dict())



@dispatch(ShapeInterface, tuple, dict)
def to_mesh(shape: ShapeInterface, color=(0.3, 0.3, 0.3), props: dict = None):
    """
    Converts a given shape to a mesh representation.

    :param shape: The shape to convert to mesh.
    :type shape: ShapeInterface
    :param color: The color of the mesh. Defaults to (0.3, 0.3, 0.3).
    :type color: tuple
    :param props: Additional properties for the mesh. Defaults to None.
    :type props: dict
    :return: The mesh representation of the shape.
    :rtype: tuple
    """
    pos, ixs, _ = shape_earcut(shape)
    return create_mesh_tuple(dict(position=pos), ixs, color, extras=props)


@dispatch(np.ndarray, tuple, dict)
def to_mesh(shape: np.ndarray, color=(0.3, 0.3, 0.3), props: dict = None):
    """
    Convert a shape to a mesh.

    :param shape: The shape to convert to a mesh.
    :type shape: np.ndarray
    :param color: The color of the mesh. Defaults to (0.3, 0.3, 0.3).
    :type color: tuple, optional
    :param props: Additional properties for the mesh. Defaults to None.
    :type props: dict, optional
    :return: The generated mesh.
    :rtype: Any
    """
    pos, ixs, _ = bounds_earcut(shape)
    return create_mesh_tuple(dict(position=pos), ixs, color, extras=props)


@dispatch(list, tuple)
def to_mesh(shape: list, color=(0.3, 0.3, 0.3)):
    """
    :param shape: A list of shape coordinates representing the dimensions of the mesh.
    :type shape: list
    :param color: A tuple representing the RGB values of the color. Defaults to (0.3, 0.3, 0.3).
    :type color: tuple
    :return: A mesh generated from the given shape and color.
    :rtype: Mesh
    """
    return mesh_from_bounds(shape, color, dict())
@dispatch(np.ndarray, tuple)
def to_mesh(shape: np.ndarray, color=(0.3, 0.3, 0.3)):
    """
    Convert the given shape to a mesh with the specified color.

    :param shape: An ndarray representing the shape of the mesh.
    :type shape: numpy.ndarray

    :param color: The color of the mesh in RGB format. Defaults to (0.3, 0.3, 0.3).
    :type color: tuple

    :return: The converted mesh.
    :rtype: numpy.ndarray
    """
    return to_mesh(shape, color, dict())


def mesh_from_bounds(bounds, color=(0.3, 0.3, 0), props=None):
    """
    Create a mesh from given bounds.

    :param bounds: a list of coordinates defining the bounds of the mesh
    :type bounds: list
    :param color: the color of the mesh, specified as a RGB tuple (default is (0.3, 0.3, 0))
    :type color: tuple
    :param props: additional properties of the mesh (default is None)
    :type props: any
    :return: a tuple representing the mesh with position, indices, color and extras
    :rtype: tuple
    """
    if bounds[0] == bounds[-1]:
        bounds = bounds[:-1]
    pos, ixs, _ = bounds_earcut(bounds.tolist() if isinstance(bounds, np.ndarray) else bounds)
    return create_mesh_tuple(dict(position=np.array(pos)), indices=np.array(ixs), color=color,
                             extras=dict(properties=props))


def mesh_from_shapes(shps, cols, stats):
    """
    Generates a mesh from given shapes.

    :param shps: List of shapes.
    :type shps: list
    :param cols: List of colors for each shape.
    :type cols: list
    :param stats: List of properties for each shape.
    :type stats: list
    :return: Generator that yields mesh tuples.
    :rtype: generator
    """
    for shp, c, props in zip(shps, cols, stats):
        pos, ixs, _ = shape_earcut(shp)

        yield create_mesh_tuple(dict(position=np.array(pos)), indices=ixs, color=c, extras=dict(properties=props))


def mesh_from_bounds_and_holes( bounds,holes,color=(0.3, 0.3, 0), props=None):
    """
    Generates a mesh from given shapes.

    :param shps: List of shapes.
    :type shps: list
    :param cols: List of colors for each shape.
    :type cols: list
    :param stats: List of properties for each shape.
    :type stats: list
    :return: Generator that yields mesh tuples.
    :rtype: generator
    """



    pos, ixs, _ = bounds_holes_earcut(bounds=bounds.tolist() if isinstance(bounds, np.ndarray) else bounds,
                                      holes=holes.tolist() if isinstance(holes, np.ndarray) else holes)
    return create_mesh_tuple(dict(position=np.array(pos)), indices=np.array(ixs), color=color,
                             extras=dict(properties=props))


