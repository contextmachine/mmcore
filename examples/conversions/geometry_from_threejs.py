import json

from mmcore.base import AMesh
from mmcore.base.models.gql import BufferGeometry, MeshPhongMaterial, create_buffer_from_dict
from mmcore.geom.materials import ColorRGB
from mmcore.base.sharedstate import serve


def geometry_from_threejs_json(path: str, material: MeshPhongMaterial = None) -> AMesh:
    """
    Use geometry & material in file to create new AMesh object
    @param path:
    @param material:
    @return:
    """
    with open(path, "r") as fl:
        geom = json.load(fl)
        return AMesh(geometry=create_buffer_from_dict(geom["geometries"][0]),
                     material=material if material is None else MeshPhongMaterial(**geom["materials"][0]))


if __name__ == '__main__':
    serve.start()
    panel = geometry_from_threejs_json("examples/data/panel.json",
                                       material=MeshPhongMaterial(color=ColorRGB(22, 150, 70).decimal))
    # open debug viewer to see result
