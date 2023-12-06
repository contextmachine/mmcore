import uuid as _uuid

import numpy as np

from mmcore.base import AMesh, Props, ageomdict
from mmcore.base.ecs import ECS
from mmcore.base.ecs.components import component
from mmcore.base.ecs.system import system
from mmcore.common.settings import DEBUG
from mmcore.geom.mesh import MeshTuple, create_mesh_buffer, simpleMaterial
from mmcore.geom.mesh.compat import MeshSupport, to_mesh


class ViewerMesh(AMesh):

    @classmethod
    def from_mesh_tuple(cls, mesh: MeshTuple, uuid=None, name="Mesh", material=simpleMaterial, **kwargs):
        if uuid is None:
            uuid = _uuid.uuid4().hex

        index = None if mesh.indices is None else mesh.indices.tolist()

        return ViewerMesh(uuid=uuid,
                          name=name,
                          geometry=create_mesh_buffer(uuid + 'geom',
                                                      **{k: attr.tolist() for k, attr in mesh.attributes.items()},
                                                      index=index),
                          material=material,
                          **kwargs)


colormap = dict()


@ECS.init_component
@component()
class Color:
    value: tuple = (0.5, 0.5, 0.5)


@component()
class PropsColor:
    ref: str = 'unit'
    colormap: dict = colormap


@system(debug=DEBUG)
def props_color_system(props: Props, color: Color, pcl=PropsColor):
    v = props[pcl.ref]
    if v not in props.colormap:
        props.colormap[v] = tuple(np.random.random(3))

    color.value = colormap[v]


def mesh_worker(mesh: MeshSupport):
    print(mesh)
    mesh.mesh = to_mesh(mesh.ref)
    index = None if mesh.mesh.indices is None else mesh.mesh.indices.tolist()
    ageomdict[mesh.uuid + 'geom'] = create_mesh_buffer(mesh.uuid + 'geom',
                                                       **{k: attr.tolist() for k, attr in mesh.mesh.attributes.items()},
                                                       index=index)
