import typing
import uuid as _uuid
from mmcore.base.models.gql import (BufferGeometry,
                                    MeshPhongMaterial,
                                    create_buffer_index,
                                    create_buffer_position,
                                    create_buffer_uv,
                                    create_float32_buffer)
from mmcore.geom.mesh.consts import simpleMaterial, vertexMaterial
from mmcore.base import AMesh


def create_buffer_objectid(array):
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


def build_mesh_with_buffer(mesh,
                           uuid=None,
                           name: str = "Mesh",
                           material=simpleMaterial,
                           props=None, controls=None,
                           **kwargs):
    if uuid is None:
        uuid = _uuid.uuid4().hex
    index = None if mesh.indices is None else mesh.indices.tolist()
    if props is None:
        props = mesh.extras.get('properties', {})

    m = AMesh(uuid=uuid,
              name=name,
              geometry=create_mesh_buffer(uuid + 'geom',
                                          **{k: attr.tolist() for k, attr in mesh.attributes.items()},
                                          index=index
                                          ),
              material=material,

              properties=props,
              controls=controls,
              **kwargs)

    if 'children' in mesh.extras:
        m.add_userdata_item('children', mesh.extras['children'])

    return m
