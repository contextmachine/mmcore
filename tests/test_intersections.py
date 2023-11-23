from unittest import TestCase

import numpy as np

from mmcore.geom.intersections import Ray, ray_intersection
from mmcore.geom.plane import create_plane
from mmcore.geom.sphere import Sphere


class Test(TestCase):
    def test_sphere_ray_intersection(self):
        result = np.array([0.66666667, 0.33333333, 0.66666667, 0.66666667, 0.84106867, 0.46364761])
        sph = Sphere(1)
        res = ray_intersection(sph, Ray(np.array((2, 1, 2)), -np.array((2, 1, 2))))

        self.assertTrue(np.allclose(result, res))

    def test_plane_ray_intersection(self):
        result = np.array([0.66666667, 0.33333333, 0., 0.66666667, -3., -2.])
        pln = create_plane()
        res = ray_intersection(pln,
                               Ray(np.array((2, 1, 2)),
                                   np.array((0, 0, -1)) - np.array((2, 1, 2)))
                               )

        self.assertTrue(np.allclose(res, result))
