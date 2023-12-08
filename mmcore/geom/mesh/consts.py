from mmcore.base import AMesh
from mmcore.base.models.gql import BufferGeometry, MeshPhongMaterial, MeshStandardVertexMaterial, MeshStandardMaterial
from mmcore.geom.materials import ColorRGB

Vec3Union = tuple[float, float, float]
Vec2Union = tuple[float, float]
Vec4Union = tuple[float, float, float, float]
vertexMaterial = MeshStandardVertexMaterial(uuid='vxmat', vertexColors=True, side=2)
simpleMaterial = MeshPhongMaterial(uuid='sxmat', color=ColorRGB(200, 200, 200).decimal, side=2)

DEFAULT_COLOR = (0.5, 0.5, 0.5)
MESH_OBJECT_ATTRIBUTE_NAME = '_objectid'
EXTRACT_MESH_ATTRS_PERFORMANCE_METHOD_LIMIT = 500
