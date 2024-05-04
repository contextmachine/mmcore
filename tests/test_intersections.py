from unittest import TestCase
"""
import numpy as np

from mmcore.geom.interfaces import Ray, collection
from mmcore.geom.intersections import intersect
from mmcore.geom.plane import create_plane
from mmcore.geom.sphere import Sphere


class Test(TestCase):
    def test_sphere_ray_intersection(self):
        result = np.array([0.66666667, 0.33333333, 0.66666667, 0.66666667, 0.84106867, 0.46364761])
        sph = Sphere(1)
        res = intersect(sph, Ray(np.array((2, 1, 2)), -np.array((2, 1, 2))))

        self.assertTrue(np.allclose(result, res))

    def test_sphere_ray_intersection_multiply(self):
        result = np.array([[0.49544190421798684,
                            0.6489401699198232,
                            0.5774201030526094,
                            -0.4792596212013541,
                            0.9552310868724129,
                            -200.14319549213994],
                           [0.6116930080288732,
                            0.1842721690451333,
                            0.7693344082023048,
                            0.2709910006362598,
                            0.6929976950795251,
                            100.82356757942586],
                           [0.6971358775994971,
                            0.5043871640814943,
                            0.5095048153584094,
                            0.14520789846733426,
                            1.0361871176904063,
                            -99.90463181733872]])
        sph = Sphere(1)
        origins = np.array([[0.88014699, 0.6794143, 0.96630839],
                            [0.50213094, 0.17339094, 0.60248626],
                            [0.62116803, 0.47791134, 0.42499213]])
        normals = [[0.80270708, 0.06358589, 0.81143557],
                   [0.40430152, 0.04015355, 0.61569626],
                   [0.52316812, 0.1823291, 0.58201526]]
        rays = collection(np.array([origins, normals]), Ray)

        res = intersect(sph, rays)

        self.assertTrue(np.allclose(result, res))

    def test_plane_ray_intersection(self):
        result = np.array([0.66666667, 0.33333333, 0., 0.66666667, -3., -2.])
        pln = create_plane()
        res = intersect(pln,
                        Ray(np.array((2, 1, 2)),
                            np.array((0, 0, -1)) - np.array((2, 1, 2)))
                        )

        self.assertTrue(np.allclose(res, result))

"""