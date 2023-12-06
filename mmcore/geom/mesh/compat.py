import typing

from mmcore.base.ecs import ECS
from mmcore.base.ecs.components import component, request_component_type
from mmcore.geom.extrusion import to_mesh
from mmcore.geom.mesh import MeshTuple, vertexMaterial
from mmcore.geom.mesh.shape_mesh import to_mesh
from mmcore.geom.rectangle import to_mesh

__all__ = ['to_mesh', 'attach_mesh', 'meshes', "MeshSupport"]


@ECS.init_component
@component()
class MeshSupport:
    mesh: MeshTuple = None
    ref: typing.Any = None
    props = None
    material = vertexMaterial
    view = None


meshes = request_component_type(MeshSupport.component_type)


def attach_mesh(uuid: str, ref, shared_components):
    if ref is None:
        raise ValueError('ref')
    mesh = to_mesh(ref)
    amsh = mesh.amesh(uuid=uuid, material=vertexMaterial)

    amsh.properties.update({k: getter(ref) for k, getter in shared_components.items()})

    return MeshSupport(ref=ref, mesh=mesh, view=amsh, props=amsh.properties, material=vertexMaterial, uuid=uuid)
