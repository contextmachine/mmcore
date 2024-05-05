import unittest
from pathlib import Path

import numpy as np

from mmcore.geom.curves import NURBSpline
import time
from mmcore.numeric.curve_intersection import curve_ppi
import time

class MyTestCase(unittest.TestCase):
    def setUp(self):
        from mmcore.geom.curves import NURBSpline

        self.aa, self.bb = NURBSpline(
            np.array(
                [
                    (-13.654958030023677, -19.907874497194975, 0.0),
                    (3.7576433265207765, -39.948793039632903, 0.0),
                    (16.324284871574083, -18.018771519834026, 0.0),
                    (44.907234268165922, -38.223959886390297, 0.0),
                    (49.260384607302036, -13.419216444520401, 0.0),
                ]
            )
        ), NURBSpline(
            np.array(
                [
                    (40.964758489325661, -3.8915666456564679, 0.0),
                    (-9.5482124270650726, -28.039230791052990, 0.0),
                    (4.1683178868166371, -58.264878428828240, 0.0),
                    (37.268687446662931, -58.100608604709883, 0.0),
                ]
            )
        )




    def test_ppi_one_intersection(self):
        s = time.time()
        res = curve_ppi(self.aa,self.bb, 0.001, tol_bbox=0.1, eager=True)
        print("time: ",divmod(time.time() - s,60))
        print(res)
        self.assertTrue(np.allclose(res, [(0.600738525390625, 0.371673583984375)]))


# add assertion here


if __name__ == "__main__":
    unittest.main()
