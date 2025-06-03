import numpy as np
import sys
sys.setrecursionlimit(100000)
from mmcore.geom.nurbs import NURBSCurve
from mmcore.numeric.vectors import norm,dot

def chord_length(R, h):
    return 2 * np.sqrt(2 * R * h - (h * h))

def adaptive_polyline(curve: NURBSCurve, tol:float, max_depth=10):
    if isinstance(curve,NURBSCurve):
        greville_abscissae_points=np.array(curve.evaluate_multi(curve.greville_abscissae))
        if curve.degree < 2 :

            return greville_abscissae_points, curve.greville_abscissae
        normals=np.asarray(curve.control_points) - greville_abscissae_points
        dn=np.array(norm(normals))
        if np.allclose(dn,0):
            return greville_abscissae_points, curve.greville_abscissae
        _res=[]
        for i in range(normals.shape[0]):
            if np.isclose(dn[i],0):
                continue

            n=normals[i]
            n/=dn[i]
            t=curve.greville_abscissae[i] #TODO: Заменить на inflection points
            tangent=np.array(curve.derivative( t))
            tangent/=np.linalg.norm(tangent
                                    )
            _res.append(1.-np.abs(np.dot(n,tangent)))



        if np.allclose(_res,0):
            return greville_abscissae_points, curve.greville_abscissae


    else:
        t0, t1 = curve.interval()

        _prms = t0 + (t1 - t0) * np.random.random(7)
        if np.all([np.allclose(curve.curvature(_p), 0) for _p in _prms]):
            return curve.points()

    params = [*tuple(curve.interval())]
    points = [curve.evaluate(params[0]), curve.evaluate(params[1])]

    def subdivide(curve,t0, t1, p0, p1, tol, points_list, current_depth):
        if current_depth>=max_depth:
            return
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
        subdivide(curve,t0, t_mid, p0, p_curve_mid, tol, points_list, current_depth+1)
        points_list.append((t_mid, p_curve_mid))
        subdivide(curve,t_mid, t1, p_curve_mid, p1, tol, points_list, current_depth+1)

    points_list = []
    subdivide(curve,params[0], params[1], points[0], points[1], tol, points_list, 0)

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
