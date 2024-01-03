from unittest import TestCase

import numpy as np

from mmcore.geom.polycurve import PolyCurve
from mmcore.geom.vec import dist


class TestPolyCurve(TestCase):
    def setUp(self) -> None:
        self.pts = np.array(
                [(-22.047791358681653, -0.8324885498102903, 0.0), (-9.3805108456226147, -28.660718210796471, 0.0),
                    (22.846252925543098, -27.408177802862003, 0.0), (15.166676249569946, 2.5182225098112045, 0.0), ]
                )

        self.pt = np.array((24.707457249539218, -8.0614698399460814, 0.0))
        self.eq1 = np.array(
                [[-22.04779136, -0.83248855, 0.0], [-9.38051085, -28.66071821, 0.0], [22.84625293, -27.4081778, 0.0],
                    [24.70745725, -8.06146984, 0.0], [15.16667625, 2.51822251, 0.0], ]
                )

        self.poly = PolyCurve.from_points(self.pts)

    def test_init(self):
        self.assertTrue(np.allclose(self.poly.corners, self.pts))

    def test_props(self):
        self.assertTrue(np.allclose(self.poly.lengths, np.array([30.57564982, 32.2510955, 30.89604074, 37.36500855]), )
                )
        self.assertTrue(np.allclose(self.poly.units, np.array(
                [[0.41429309, -0.91014352, 0.0], [0.99924555, 0.03883714, 0.0], [-0.24856184, 0.96861603, 0.0],
                    [-0.99597107, -0.0896751, 0.0], ]
                ), )
                )

    def test_mbr_origin(self):
        d = dist(self.poly.mbr.origin, self.poly.corners)
        self.assertTrue(np.all(d[0] <= d[1:]))

    def test_mbr_normal(self):
        self.assertTrue(np.allclose(self.poly.mbr.normal - np.array([0.0, 0.0, 1.0]), 0.0)
                )

    def test_insert_corners(self):
        print(self.poly.corners, self.pt, "\n\n")
        self.poly.insert_corner(self.pt)
        print(self.poly.corners, self.eq1)
        self.assertTrue(np.allclose(self.poly.corners, self.eq1))

    def test_corners_setter(self):
        corns = np.array(
                [[-22.04779136, -0.83248855, 0.0], [-9.38051085, -28.66071821, 0.0], [22.84625293, -27.4081778, 0.0],
                    [24.70745725, -8.06146984, 0.0], ]
                )
        self.poly.corners = corns
        self.assertTrue(np.allclose(self.poly.corners - corns, 0.0))
