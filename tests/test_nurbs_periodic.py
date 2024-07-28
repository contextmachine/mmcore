import numpy as np

from mmcore.geom.curves.bspline import NURBSpline
import pytest


def test_nurbs_periodic():
    pts = np.array(
        [[148.39347323205118, -60.435505127468502, 0.0],
         [148.39347323205118, -60.435505127468502, 312.99479852664467],
         [15.867952362691028, -60.435505127468502, 312.99479852664467],
         [15.867952362691028, 79.868450977375943, 312.99479852664467],
         [127.81520754187642, 79.868450977375943, 312.99479852664467],
         [127.81520754187642, 263.24739314874569, 312.99479852664467],
         [225.93283148956732, 263.24739314874569, 312.99479852664467],
         [225.93283148956732, 178.59891945435234, 312.99479852664467],
         [336.65283186482787, 178.59891945435234, 312.99479852664467],
         [336.65283186482787, 64.991782255356810, 312.99479852664467]]
        )

    for i in range(3):
        nc = NURBSpline(pts, degree=i + 1)
        start, end = nc.interval()
        assert not np.allclose(nc.evaluate(start), nc.evaluate(end))
        assert not nc.is_periodic()
        assert not nc.is_closed()
        assert nc.is_open()

        nc.make_periodic()
        assert nc.is_periodic()
        assert nc.is_closed()
        assert not nc.is_open()
        start, end = nc.interval()
        assert np.allclose(nc.evaluate(start), nc.evaluate(end))
        assert np.allclose(nc.control_points[:nc.degree], nc.control_points[len(pts):])

#nc.make_periodic()
#pts1=nc(np.linspace(*nc.interval(),100)).tolist()
