import numpy as np
import rhino3dm as rg

from mmcore.addons.rhino import control_points_curve


def random_pointlist(count):
    return list(map(lambda x: rg.Point3d(*x), np.random.random((count, 3))))


def random_control_points_curve(count=5, degree=3):
    return control_points_curve(np.random.random((count, 3)), degree=degree)
