import numpy as np
from mmcore.geom.nurbs import NURBSCurve, NURBSSurface

from mmcore._test_data import csx as csx_cases
from mmcore.numeric.intersection.csx import nurbs_csx


def test_nurbs_csx():
    surface, curve = csx_cases[0]

    tolerance = 1e-6
    import time

    s = time.perf_counter_ns()
    result = nurbs_csx(curve, surface, tolerance, 1e-10)
    e = time.perf_counter_ns() - s

    # CHECK

    # To verify the robustness of this the implementation, let's check the distance between
    # the point estimated on the curve and on the surface in the intersection parameters.

    for typ, pt, (t, u, v) in result:
        pt1 = surface.evaluate_v2(u, v)  # evaluate point on surface
        pt2 = curve.evaluate(t)  # evaluate point on curve
        print(pt1, pt2)
        dist = np.linalg.norm(surface.evaluate_v2(u, v) - curve.evaluate(t))  # Must be less than the tolerance
        print(f'error: {dist} (must be less than {tolerance})')

        assert dist < tolerance  # If dist>=tolerance an AssertionError will be raised

    print(f"CSX performed at: {e * 1e-9} secs.")