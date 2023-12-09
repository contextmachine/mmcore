import timeit
import unittest

import numpy as np

from mmcore.geom.plane import plane_from_normal_numeric, test_plane_num


class TestPlane(unittest.TestCase):
    def test_from_normal(self):
        res = test_plane_num(plane_from_normal_numeric(np.random.random((1000, 3)), np.zeros(3)))
        print(res)
        self.assertTrue(all(res))

    def test_timeit_from_normal(self):
        res = timeit.timeit(
            "import numpy as np;from mmcore.geom.plane import plane_from_normal_numeric, test_plane_num;test_plane_num(plane_from_normal_numeric(np.random.random(3),np.zeros(3)))",
            number=1000)
        print('all:', res, 'per loop:', res / 1000)
        self.assertLess(res, 0.3)
