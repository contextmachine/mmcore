import json

from mmcore.base import AGroup
from mmcore.geom.shapes import Earcut, Shape


def test_earcut():
    with open("../tests/data/shapes.json") as f:
        shape_data = json.load(f)
    grp = AGroup(name="test_shapes", uuid="test_shapes")
    grp.scale(0.001, 0.001, 0.001)
    for shape in shape_data['shapes']:
        ec = Earcut(boundary=shape['bounds'], holes=shape['holes'])
        grp.add(ec.mesh_data.to_mesh())
    grp.dump("shapes_test_result_ec.json")


def test_shape():
    with open("../tests/data/shapes.json") as f:
        shape_data = json.load(f)
    grp = AGroup(name="test_shapes", uuid="test_shapes")
    grp.scale(0.001, 0.001, 0.001)
    for shape in shape_data['shapes']:
        shp = Shape(boundary=shape['bounds'], holes=shape['holes'])

        grp.add(shp.mesh_data.to_mesh())
    grp.dump("shapes_test_result_shapes.json")


if __name__ == "__main__":
    test_shape()
