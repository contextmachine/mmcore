import timeit
import unittest

from mmcore.geom.intersections import intersect
from mmcore.geom.line import Line
from mmcore.geom.plane import Plane, plane_from_normal_numeric
from mmcore.geom.vec import *


@vectorize(signature='(j,i)->()')
def test_plane_num(pln):
    """
    Calculate if the given plane is perpendicular to the coordinate axes.

    :param pln: A list representing the plane. The first element is not considered.
                 The next three elements are the coordinates of three non-collinear points in the plane.
    :type pln: list

    :return: A boolean value indicating if the given plane is perpendicular to the coordinate axes.
    :rtype: bool
    """
    X, Y, Z = pln[1:]
    return np.allclose([dot(X, Y), dot(Y, Z), dot(Z, X)], 0)

class TestPlane(unittest.TestCase):

    def test_from_normal(self):
        res = test_plane_num(plane_from_normal_numeric(np.random.random((1000, 3)), np.zeros(3)))
        print(res)
        self.assertTrue(all(res))

    def test_timeit_from_normal(self):
        res = timeit.timeit(
            "import numpy as np;from mmcore.geom.plane import plane_from_normal_numeric;from tests.test_plane import test_plane_num;test_plane_num(plane_from_normal_numeric(np.random.random(3),np.zeros(3)))",
            number=1000)
        print('all:', res, 'per loop:', res / 1000)
        self.assertLess(res, 0.3)

    def test_plane_intersection(self):
        check = Line.from_ends(*np.array([(-14.581811182367993, -10.914653736467775, 1.1467086205536206),
                                          (-14.958123938728997, -10.017969457130338, 0.91680501947801241)]))

        a = Plane(np.array(
            [(-15.252781720625956, -8.9343930316149311, 0.0), (0.43558724226601764, -0.90014651828193271, 0.0),
             (-0.19802682989028661, -0.095826578201100243, 0.97550122580849274),
             (-0.87809403199127201, -0.42491588877704106, -0.21999399638655617)]))
        b = Plane(np.array(
            [(-21.840274609354235, -16.810687378633077, 0.0), (0.46067064412253583, 0.88757115638337858, -0.0),
             (0.8483114468989893, -0.44029391654845362, 0.29412404884770366),
             (0.26105602215591778, -0.13549431503459985, -0.95576725403700347)]))
        res = intersect(a, b)
        print(res)
        print(check.closest_distance(res))
        self.assertTrue(np.allclose(check.closest_distance(res), 0.))