import gzip
import json

import numpy as np

import mmcore.base.models.gql
from mmcore.base.basic import getattr_, objdict, geomdict, matdict
from mmcore.node import node_eval

mapattrs = {
    "normal": mmcore.base.models.gql.Normal,
    "position": mmcore.base.models.gql.Position,
    "uv": mmcore.base.models.gql.Uv

}

attributes=[]
def geom_attributes_from_dict(att):
    ddct = {}
    for k, v in att.items():
        ddct[k] = mapattrs[k](**v)
    if len(ddct.keys()) == 1:
        return mmcore.base.models.gql.Attributes1(**ddct)
    elif len(ddct.keys()) == 3:
        return mmcore.base.models.gql.Attributes2(**ddct)
    elif len(ddct.keys()) == 2:
        return mmcore.base.models.gql.Attributes3(**ddct)
    else:
        return mmcore.base.models.gql.Attributes(**ddct)


def create_buffer_from_dict(kwargs) -> mmcore.base.models.gql.BufferGeometry:
    if "index" in kwargs["data"]:

        dct = mmcore.base.models.gql.BufferGeometry(**{
            "uuid": kwargs.get('uuid'),
            "type": kwargs.get('type'),
            "data": mmcore.base.models.gql.Data(**{
                "attributes": geom_attributes_from_dict(kwargs['data']['attributes']),

            })
        })

        dct.data.index: mmcore.base.models.gql.Index(**kwargs.get('data').get("index"))
    else:

        dct = mmcore.base.models.gql.BufferGeometry(**{
            "uuid": kwargs.get('uuid'),
            "type": kwargs.get('type'),
            "data": mmcore.base.models.gql.Data1(**{
                "attributes": geom_attributes_from_dict(kwargs['data']['attributes']),

            })
        })

    return dct


def create_buffer_from_occ(kwargs) -> mmcore.base.models.gql.BufferGeometry:
    return mmcore.base.models.gql.BufferGeometry(**{
        "uuid": kwargs.get('uuid'),
        "type": kwargs.get('type'),
        "data": mmcore.base.models.gql.Data(**{
            "attributes": mmcore.base.models.gql.Attributes(**{

                "normal": mmcore.base.models.gql.Normal(**kwargs.get('data').get('attributes').get("normal")),
                "position": mmcore.base.models.gql.Position(**kwargs.get('data').get('attributes').get("position")),

            })

        })
    })


def create_buffer(uuid, normals, vertices, uv, indices) -> mmcore.base.models.gql.BufferGeometry:
    return mmcore.base.models.gql.BufferGeometry(**{
        "uuid": uuid,
        "type": "BufferGeometry",
        "data": mmcore.base.models.gql.Data(**{
            "attributes": mmcore.base.models.gql.Attributes(**{

                "normal": mmcore.base.models.gql.Normal(**{
                    "array": np.asarray(normals, dtype=float).flatten().tolist(),
                    "itemSize": 3,
                    "type": "Float32Array",
                    "normalized": False
                }),
                "position": mmcore.base.models.gql.Position(**{
                    "array": np.asarray(vertices, dtype=float).flatten().tolist(),
                    "itemSize": 3,
                    "type": "Float32Array",
                    "normalized": False
                }),
                "uv": mmcore.base.models.gql.Uv(**{
                    'itemSize': 2,
                    "array": np.asarray(uv, dtype=float).flatten().tolist(),
                    "type": "Float32Array",
                    "normalized": False

                }),

            }),
            "index": mmcore.base.models.gql.Index(**dict(type='Uint16Array',
                                                         array=np.asarray(indices, dtype=int).flatten().tolist()))
        })
    })


def parse_attribute(attr):
    arr = np.array(attr.array)
    shp = int(len(arr) / attr.itemSize), attr.itemSize
    arr.reshape(shp)
    return arr


@node_eval
def foo_three_js_type():
    # language=JavaScript
    return '''const THREE = require("three");
              const result = new THREE.Object3D();
              console.log(JSON.stringify(result.toJSON()))'''


def dumps_graph(path):
    uu = [getattr_(i) for i in list(objdict.values())]
    with gzip.open(f"{path}/buffer.gz", "w", compresslevel=9) as f:
        f.write(json.dumps({"geometry": dict(geomdict), "materials": dict(matdict)}).encode())
    with open(f"{path}/graph.json", "w") as fg:
        json.dump(uu, fg, cls=sketch.ExEncoder, ensure_ascii=False)


def is_geometry(obj):
    if isinstance(obj, dict):
        return "geometry" in obj.keys()

    else:
        return hasattr(obj, "geometry")


def pop_in_keys(dct, key):
    if key in dct.keys():
        return dct.pop(key)

"""
def create_object_from_threejs(kwargs):
    def trav(obj):
        cls = getattr(sys.modules['__main__'], obj.pop("type"), __default=Object3D)

        geometry = pop_in_keys("geometry")

        if isinstance(cls, GeometryObject):
            material = pop_in_keys("material")
            children = pop_in_keys("children")
            return cls.from_three(obj, geometry, material)
        if "children" in obj.keys() and (len(obj["children"]) > 0):
            children = obj.pop("children")
            cls(**obj)
            for child in obj.children: ...
"""