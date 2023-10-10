# TODO this code dont working
import json

from mmcore.base import AMesh
from mmcore.base.models.gql import MeshPhongMaterial, create_buffer_from_dict


def mesh_from_threejs_json(path: str, material: MeshPhongMaterial = None) -> AMesh:
    with open(path, "r") as fl:
        geom = json.load(fl)
        object_dict = geom["object"]
        object_dict.pop("material"), object_dict.pop("geometry")
        childs = None
        if "children" in object_dict.keys():
            childs = object_dict.pop("children")
        if material is None:
            material = MeshPhongMaterial(**geom["materials"][0])

        mesh = AMesh(geometry=create_buffer_from_dict(geom["geometries"][0]), material=material, **object_dict)
        if childs is not None:
            for i, m in enumerate(childs):
                child_mesh = mesh_from_threejs_json(m)
                mesh.__setattr__(f"part-{i}", child_mesh)
        return mesh