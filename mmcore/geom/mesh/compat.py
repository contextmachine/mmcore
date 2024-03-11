import typing
import uuid as _uuid

import numpy as np

from mmcore.base import AMesh
from mmcore.base.models.gql import (BufferGeometry,
                                    create_buffer_index,
                                    create_buffer_position,
                                    create_buffer_uv,
                                    create_float32_buffer)
from mmcore.geom.mesh.consts import simpleMaterial


def create_buffer_objectid(array):
    """
    :param array: the array of object ids to be stored in the buffer object
    :type array: list
    :return: a buffer object with the provided array of object ids
    :rtype: dict

    """
    return {
        'type': 'Uint16Array',
        "itemSize": 1,
        "array": array
    }


def create_mesh_buffer(
        uuid,
        position=None,
        uv=None,
        index=None,
        normal=None,
        _objectid=None,
        color: typing.Optional[list[float]] = None, threejs_type="BufferGeometry"):
    """
    :param uuid: The unique identifier for the mesh buffer
    :type uuid: str
    :param position: The position data for the mesh buffer
    :type position: list[float]
    :param uv: The UV data for the mesh buffer
    :type uv: list[float]
    :param index: The index data for the mesh buffer
    :type index: list[int]
    :param normal: The normal data for the mesh buffer
    :type normal: list[float]
    :param _objectid: The object ID data for the mesh buffer
    :type _objectid: list[int]
    :param color: The color data for the mesh buffer
    :type color: list[float]
    :param threejs_type: The type of the mesh buffer (default is "BufferGeometry")
    :type threejs_type: str
    :return: A BufferGeometry object representing the mesh buffer
    :rtype: BufferGeometry
    """

    attra = dict(position=create_buffer_position(position))
    if color is not None:
        attra['color'] = create_float32_buffer(color)
    if normal is not None:
        attra['normal'] = create_float32_buffer(normal)
    if uv is not None:
        attra['uv'] = create_buffer_uv(uv)

    if _objectid is not None:
        attra['_objectid'] = create_buffer_objectid(_objectid)
    if index is not None:
        ixs = create_buffer_index(index)
        return BufferGeometry(**{
            "uuid": uuid,
            "type": threejs_type,
            "data": {
                "attributes": attra,
                "index": ixs

            }
        })

    else:
        return BufferGeometry(**{
            "uuid": uuid,
            "type": threejs_type,
            "data": {
                "attributes": attra

            }
        })


def create_mesh_buffer_from_mesh_tuple(mesh, uuid=None):
    if uuid is None:
        uuid = _uuid.uuid4().hex
    return create_mesh_buffer(uuid + 'geom',
                              **{k: attr.tolist() for k, attr in mesh[0].items()},
                              index=mesh.indices.tolist() if isinstance(mesh.indices, np.ndarray) else mesh.indices)
def build_mesh_with_buffer(mesh,
                           uuid=None,
                           name: str = "Mesh",
                           material=simpleMaterial,
                           props=None, controls=None,
                           **kwargs):
    """
    Builds a mesh with buffer.

    :param mesh: The mesh object.
    :type mesh: <class 'Mesh'>
    :param uuid: The UUID of the mesh. Defaults to None.
    :type uuid: Optional[str]
    :param name: The name of the mesh. Defaults to "Mesh".
    :type name: str
    :param material: The material of the mesh. Defaults to simpleMaterial.
    :type material: <class 'Material'>
    :param props: The properties of the mesh. Defaults to None.
    :type props: Optional[dict]
    :param controls: The controls of the mesh. Defaults to None.
    :type controls: Optional[Any]
    :param kwargs: Additional keyword arguments.
    :type kwargs: Any
    :return: The mesh with buffer.
    :rtype: <class 'AMesh'>
    """
    if uuid is None:
        uuid = _uuid.uuid4().hex

    index = None if mesh[1] is None else mesh[1].tolist()

    if props is None:
        props = mesh[2].get('properties', {})

    m = AMesh(uuid=uuid,
              name=name,
              geometry=create_mesh_buffer(uuid + 'geom',
                                          **{k: attr.tolist() for k, attr in mesh[0].items()},
                                          index=index.tolist() if isinstance(index, np.ndarray) else index
                                          ),
              material=material,

              properties=props,
              controls=controls,
              **kwargs)

    if 'children' in mesh[2]:
        m.add_userdata_item('children', mesh[2]['children'])

    return m
