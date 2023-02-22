import os

import rpyc
import yaml


def get_connection():
    """


    @return: connection

    [1] Basic Example:
    >>> from mmcore.services.rhpyc import get_connection
    >>> from mmcore.addons.rhino import point_to_tuple
    >>> conn = get_connection("localhost", 18812)
    >>> rh = conn.root.getmodule("Rhino")
    >>> rg = conn.root.getmodule("Rhino.Geometry")
    >>> rs = conn.root.getmodule("rhinoscript")
    >>> circle = rg.Circle(rg.Plane(0.0,0.0,1.0,0.0), 15.4) # create circle at world xyz plane
    >>> point_on_circle = circle.ClosestPoint(rg.Point3d(1,2,3))
    >>> point_to_tuple(point_on_circle)
    (6.887089370699354, 13.774178741398705, 0.0)

    ---
    [2] Advance Example
    Creating threejs object from rhino geometry
    >>> from mmcore.addons import rhino
    >>> sph=rg.Sphere(rg.Point3d(1,2,3), 18)
    >>> sph_mesh = rg.Mesh.CreateFromSphere(sph)
    >>> sph_brep_arr = rg.Mesh.CreateFromBrep(sph.ToBrep()) # Mesh.CreateFromBrep return mesh array (Mesh[]) object.
    >>> from mmcore.geom.materials import MeshPhysicalBasic, ColorRGB
    >>> webgl_object = rhino.mesh_to_buffer_mesh(sph_brep_arr[0], MeshPhysicalBasic(ColorRGB(255,40,22)))
    >>> webgl_object
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
       'iridescence': 0,
       'depthWrite': True,
       'stencilFunc': 519,
       'sheenRoughness': 0,
       'envMapIntensity': 1,
       'sheenColor': 0,
       'reflectivity': 0.64,
       'clearcoatRoughness': 0,
       'metalness': 0,
       'stencilRef': 0,
       'uuid': '64ce2001-8bc2-4dbf-a4c5-978017d3a4b2',
       'colorWrite': True,
       'stencilFail': 7680,
       'color': 16721942,
       'thickness': 1.72,
       'emissive': 0,
       'transmission': 0,
       'depthTest': True,
       'clearcoat': 0,
       'sheen': 0,
       'specularColor': 16777215,
       'stencilWriteMask': 255,
       'iridescenceThicknessRange': [100, 10000],
       'stencilZPass': 7680,
       'iridescenceIOR': 1.56,
       'stencilFuncMask': 255,
       'stencilWrite': False,
       'side': 2,
       'attenuationColor': 16777215,
       'specularIntensity': 1,
       'depthFunc': 3}],
    'object': {'uuid': '7e32f6dc-c2a7-48e0-ac09-0f2a377d1939',
      'type': 'Mesh',
      'name': 'TestMesh',
      'castShadow': True,
      'receiveShadow': True,
      'layers': 1,
      'matrix': (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1),
      'geometry': '46444538-4f6a-4ad0-97dc-3568d5d6ddcd',
      'material': '64ce2001-8bc2-4dbf-a4c5-978017d3a4b2'}}
    """
    with open(os.getenv("RPYC_CONFIGS")) as f:
        data = yaml.unsafe_load(f)
        if list(data.keys())[0] == "service" and data["service"]["name"] == "rhpyc":
            configs = data["service"].get("configs")
            attrs = data["service"].get("attributes")

            hosts = configs.get("hosts")
            port = attrs.get("port")
    hosts = hosts if hosts is not None else [os.getenv("RHINO_RPYC_HOST")]
    port = port if port is not None else os.getenv("RHINO_RPYC_PORT")
    i = -1
    while True:
        i += 1
        try:
            print(hosts[i])

            conn = rpyc.connect(host=hosts[i], port=port)
            conn.ping()
            if not conn.closed:
                print(f"{hosts[i]} success!!!")
                rhpyc_conn = conn
                break


        except ConnectionRefusedError:
            print(f"{hosts[i]} fail...")

    return rhpyc_conn


class PrintTools:
    def __init__(self, conn):
        self.conn = conn
        self.conn.root.execute("import Rhino;import Rhino.Geometry as rg;import rhinoscript as rs")
        self.conn.root.execute("rdir = lambda obj:dir(obj)")
        self.conn.root.execute("hlp = lambda obj:help(obj)")

    def rdir(self, obj): return self.conn.root.namespace['rdir'](obj)

    def rhelp(self, obj): return self.conn.root.namespace['hlp'](obj)


def remote_help(obj):
    return obj.__doc__
