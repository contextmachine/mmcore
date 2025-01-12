from typing import Union

import numpy as np

from mmcore.numeric.intersection.ccx import curve_x_ray
from mmcore.geom.curves.curve import Curve
from mmcore.geom.implicit import Implicit2D
from mmcore.geom.polygon import is_point_in_polygon_bvh,polygon_build_bvh,segments_by_loop

def _is_closed_curve(crv: Curve):
    pt1, pt2 = crv.evaluate_multi(np.array(crv.interval(), dtype=float))
    return np.allclose(pt1 - pt2, 0.)


def point_in_parametric_curve(crv: Curve, xyz):

    if not _is_closed_curve(crv):
        return False
    if hasattr(crv,'degree'):
        if crv.degree==1:
            polygon=crv.evaluate_multi(np.arange(*crv.interval()))
            edges = [(polygon[i], polygon[(i + 1) % len(polygon)]) for i in range(len(polygon))]
            bvh_root=polygon_build_bvh(edges)
            is_point_in_polygon_bvh(bvh_root,xyz)
    crr=crv.interval()
    rngg=crr[1]-crr[0]
    print(rngg)

    return (len(curve_x_ray(crv, xyz, axis=0,step= rngg)) % 2) > 0


def point_in_implicit_curve(crv: Implicit2D, xyz):
    res = crv.implicit(xyz[:2])
    if res <= 0:
        return True
    elif np.isclose(res, 0):
        return True
    else:
        return False


def point_in_curve(crv: Union[Curve, Implicit2D], xyz):
    if hasattr(crv, 'implicit'):
        return point_in_implicit_curve(crv, xyz)
    elif hasattr(crv, 'evaluate'):
        return point_in_parametric_curve(crv, xyz)
    else:
        raise ValueError('Unknown curve protocol')
