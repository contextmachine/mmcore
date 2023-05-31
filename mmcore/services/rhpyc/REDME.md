# Rhpyc Service

Based on mmcore & rhinocode

## Basic Example:

Closest point on circle curve for nine lines of code.

```doctest
>>> from mmcore.services.rhpyc import get_connection
>>> from mmcore.addons.rhino import point_to_tuple

>>> conn = get_connection("localhost", 18812)
>>> rh = conn.root.getmodule("Rhino")
>>> rg = conn.root.getmodule("Rhino.Geometry")
>>> rs = conn.root.getmodule("rhinoscript")

>>> circle = rg.Circle(rg.BspPlane(0.0,0.0,1.0,0.0), 15.4) # create circle at world xyz plane
>>> point_on_circle = circle.ClosestPoint(rg.Point3d(1,2,3))
>>> point_to_tuple(point_on_circle)
(6.887089370699354, 13.774178741398705, 0.0)
```

## Advance Example

Creating webgl object from rhino geometry for ten lines of code.

```doctest
import geom.utils.tools>>>import geom.buffer from mmcore.services.rhpyc import get_connection
>>> conn = get_connection("localhost", 18812)
>>> rg = conn.root.getmodule("Rhino.Geometry")

>>> sph=rg.Sphere(rg.Point3d(1,2,3), 18)
>>> sph_mesh = rg.MeshObject.CreateFromSphere(sph)
>>> sph_brep_arr = rg.MeshObject.CreateFromBrep(sph.ToBrep()) # Mesh.CreateFromBrep return mesh array (Mesh[]) object.

>>> from mmcore.addons import rhino
>>> from mmcore.geom.materials import MeshPhysicalBasic, ColorRGB
    webgl_object = geom.utils.tools.mesh_to_buffer_mesh(sph_brep_arr[0], MeshPhysicalBasic(ColorRGB(255,40,22)))
>>> webgl_object # some parts of the json output were omitted
{'metadata': {'version': 4.5,
  'type': 'Object',
  'generator': 'Object3D.toJSON'},
  'geometries': [{'uuid': '46444538-4f6a-4ad0-97dc-3568d5d6ddcd',
    'type': 'BufferGeometry',
    'data': {'attributes': {'normal': {'array': [...
     ...]}}}],
  'materials': [{'stencilZFail': 7680,
     'roughness': 0.24,
     'type': 'MeshPhysicalMaterial',
     'color': 16721942,
     'thickness': 1.72,
      ...}],
  'object': {'uuid': '7e32f6dc-c2a7-48e0-ac09-0f2a377d1939',
    'type': 'Mesh',
    'name': 'TestMesh',
    'castShadow': True,
    'receiveShadow': True,
    'layers': 1,
    'matrix': (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1),
    'geometry': '46444538-4f6a-4ad0-97dc-3568d5d6ddcd',
    'material': '64ce2001-8bc2-4dbf-a4c5-978017d3a4b2'}}

```
![Solid view in Three.js editor](http://storage.yandexcloud.net/box.contextmachine.space/content/Screenshot%202023-02-16%20at%2017.59.54.png)
![Wireframe view in Three.js editor](http://storage.yandexcloud.net/box.contextmachine.space/content/Screenshot%202023-02-16%20at%2017.59.19.png)
