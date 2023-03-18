import numpy as np
from mmcore.addons import ModuleResolver
with ModuleResolver() as rsl:
    import rhino3dm
import rhino3dm as rg


def control_points_curve(points: list[list[float]] | np.ndarray, degree: int = 3):
    return rg.NurbsCurve.CreateControlPointCurve(list(map(lambda x: rg.Point3d(*x), points)),
                                                 degree=degree)


def random_pointlist(count):
    return list(map(lambda x: rg.Point3d(*x), np.random.random((count, 3))))


def random_control_points_curve(count=5, degree=3):
    return control_points_curve(np.random.random((count, 3)), degree=degree)
