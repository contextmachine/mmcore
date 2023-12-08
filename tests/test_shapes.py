import json
import pickle
from unittest import TestCase

import numpy as np

from mmcore.geom.shapes import Earcut, Shape


def mdss(a, b):
    rr = []
    for i in ['vertices', 'normals', 'indices', 'uv']:
        ai, bi = getattr(a, i), getattr(b, i)
        if not all([ai is None, bi is None]):

            res = np.allclose(ai, bi)
            if not res:
                rr.append((i, ai, bi))
    return not len(rr) > 0, rr


class TestUmeshMeshCandidate(TestCase):

    def test_earcut(self):
        with open("../tests/data/shapes.json") as f:
            shape_data = json.load(f)
        with open("../tests/data/shapes_test_ec.pkl", 'rb') as f:
            ec_result_data = pickle.load(f)

        for i, shape in enumerate(shape_data['shapes']):
            ec = Earcut(boundary=shape['bounds'], holes=shape['holes'])

            r, _ = mdss(ec_result_data[i], ec.mesh_data)
            self.assertTrue(r)

    def test_shape(self):
        with open("../tests/data/shapes.json") as f:
            shape_data = json.load(f)
        with open("../tests/data/shapes_test_result_shapes.pkl", 'rb') as f:
            shape_result_data = pickle.load(f)

        for i, shape in enumerate(shape_data['shapes']):
            shp = Shape(boundary=shape['bounds'], holes=shape['holes'])
            r, _ = mdss(shape_result_data[i], shp.mesh_data)
            self.assertTrue(r)
