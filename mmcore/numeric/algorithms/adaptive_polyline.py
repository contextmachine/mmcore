import numpy as np

from mmcore.geom.nurbs import NURBSCurve
from mmcore.numeric.vectors import norm

def chord_length(R, h):
    return 2 * np.sqrt(2 * R * h - (h * h))


def adaptive_polyline(curve: NURBSCurve, tol:float):
    greville_abscissae_points=np.array(curve.evaluate_multi(curve.greville_abscissae))
    if curve.degree < 2 or np.allclose(np.array(norm(np.asarray(curve.control_points)-greville_abscissae_points)),0):

        return greville_abscissae_points, curve.greville_abscissae

    params = [*tuple(curve.interval())]
    points = [curve.evaluate(params[0]), curve.evaluate(params[1])]

    def subdivide(curve,t0, t1, p0, p1, tol, points_list):
        segment_length = np.linalg.norm(p0 - p1)

        t_mid = (t0 + t1) / 2
        p_curve_mid = np.asarray(curve.evaluate(t_mid))
        curvature_vector = np.asarray(curve.curvature(t_mid))
        curvature_length = np.linalg.norm(curvature_vector)
        p_line_mid = (p0 + p1) * 0.5
        if (not np.isclose(curvature_length, 0)) and np.isfinite(curvature_length):

            R = 1 / curvature_length

            L = chord_length(R, tol)

            if (L >= segment_length) and (np.linalg.norm(p_curve_mid - p_line_mid) < tol):
                return
        # Subdivide further
        subdivide(t0, t_mid, p0, p_curve_mid, tol, points_list)
        points_list.append((t_mid, p_curve_mid))
        subdivide(t_mid, t1, p_curve_mid, p1, tol, points_list)

    points_list = []
    subdivide(curve,params[0], params[1], points[0], points[1], tol, points_list)

    # Sort points based on parameter to maintain correct order
    points_list.sort(key=lambda x: x[0])

    # Construct the final ordered points list
    final_points = [points[0]]
    final_params = [params[0]]
    for t, pt in points_list:
        final_points.append(pt)
        final_params.append(t)
    final_points.append(points[1])
    final_params.append(params[1])

    return np.asarray(final_points), np.asarray(final_params, dtype=float)
