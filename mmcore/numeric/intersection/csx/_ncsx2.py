import numpy as np

from mmcore.geom.nurbs import subdivide_surface, split_curve,NURBSCurve,NURBSSurface,CurveSurfaceEq
from mmcore.numeric.aabb import aabb_intersect_fast_3d
from mmcore.numeric.newton.cnewton import newtons_method

from mmcore.numeric.algorithms.adaptive_polyline import chord_length
from mmcore.numeric.divide_and_conquer import divide_and_conquer_min_3d
from ._ncsx_new_intersections_test import new_intersection_candidates

def int_cs( initial_curve, initial_surface,spt=1e-3, tol=1e-7, debug=False,**kwargs):
    stack = [(initial_surface, initial_curve)]
    results=[]
    while stack:

        _surface, _curve = stack.pop()
        if not aabb_intersect_fast_3d(_surface.bbox(), _curve.bbox()):
            continue
        t0, t1 = _curve.interval()
        (u0, u1), (v0, v1) = _surface.interval()


        t_mid = (t1 - t0) * 0.5 + t0
        u_mid = (u1 - u0) * 0.5 + u0
        v_mid = (v1 - v0) * 0.5 + v0

        surf_curve_eq=CurveSurfaceEq(_curve,_surface)

        if surf_curve_eq(np.array([t_mid, u_mid, v_mid]))<=spt:

            result = newtons_method(surf_curve_eq, np.array([t_mid, u_mid, v_mid]),tol=min(tol,1e-8),max_iter=5
                                )

            if result is None:
                #print('n', t_mid,u_mid,v_mid)
                for s in subdivide_surface(_surface, u_mid, v_mid, tol=1e-12, normalize_knots=False):
                    for c in split_curve(_curve, t_mid, tol=1e-12, normalize_knots=False):
                        stack.append((s, c))
                continue


            t, u, v = result
            if not (t0 <= t <= t1 and u0 <= u <= u1 and v0 <= v <= v1):
                continue
            if debug:

                print(t, u, v)

            if not any([all([abs(_t - t) < tol , abs(_u - u) < tol  , abs(_v - v)< tol ])  for  int_type, xyz,(_t, _u, _v) in results]):
                #print('g', t, u, v)
                crv_pt = np.array(_curve.evaluate(t))
                surf_pt = np.array(_surface.evaluate_v2(u, v))
                d = surf_pt - crv_pt

                dn = np.linalg.norm(d)

                if dn < spt:

                    results.append(("transversal",surf_pt,(t, u, v)))
                    if (u - u0) < 1e-11 or (v - v0) < 1e-11 or (t - t0) < 1e-11 or (u1 - u) < 1e-11 or (v1 - v) < 1e-11 or (t1 - t) < 1e-11:
                        continue
                    cand = new_intersection_candidates(_surface, _curve, u, v, t, surf_pt)
                    if debug:
                        print(len(cand))
                    stack.extend(cand)
                    continue


        if (u_mid - u0) < 1e-11 or (v_mid - v0) < 1e-11 or (t_mid - t0) < 1e-11:
            continue

        else:
            for s in subdivide_surface(_surface, u_mid, v_mid, tol=1e-12, normalize_knots=False):
                for c in split_curve(_curve, t_mid, tol=1e-12, normalize_knots=False):
                    stack.append((s, c))


    return sorted(results,key=lambda x:x[2][0])



if __name__ == "__main__":
    cpts = np.array(
        [[-9.1796875, 13.229166666666666, -4.5186767578125], [-9.1796875, 14.739583333333332, -4.49395751953125],
         [-9.1796875, 16.432291666666664, -4.580108642578125], [-9.1796875, 18.372395833333332, -4.8531036376953125]]
    )
    spts = np.array([[[-5.849180481790346, 18.372395833333336, -1.5018374203712104],
                      [-5.858792686592141, 16.432291666666668, -2.719633841323509],
                      [-5.871782152540512, 14.739583333333334, -3.1229032219131403],
                      [-5.8852911971268185, 13.229166666666668, -3.116512598566417]],
                     [[-6.88688536134276, 18.372395833333336, -1.6832837287863824],
                      [-6.894094514944105, 16.432291666666668, -2.9325012796112815],
                      [-6.9038366144053835, 14.739583333333334, -3.3409109281989706],
                      [-6.913968397845114, 13.229166666666668, -3.3217200441495054]],
                     [[-7.97766402100707, 18.372395833333336, -2.2658223298287345],
                      [-7.983070886208079, 16.432291666666668, -3.617300740689732],
                      [-7.990377460804037, 14.739583333333334, -4.0455099012702895],
                      [-7.997976298383835, 13.229166666666668, -3.992029157974829]],
                     [[-9.1863730157553, 18.372395833333336, -2.7409113689805125],
                      [-9.190428164656058, 16.432291666666668, -4.174381706229336],
                      [-9.195908095603027, 14.739583333333334, -4.61538352236864],
                      [-9.201607223787876, 13.229166666666668, -4.527011611303911]]]
                    )

    u, v, t = 0.9939461136471586, 0.995759608283125, 0.004240391716877873
    surf = NURBSSurface(np.array(spts), (3, 3))

    surf.normalize_knots()

    curve = NURBSCurve(cpts)
    ress = new_intersection_candidates(surf, curve, u, v, t, np.array(surf.evaluate_v2(u, v)))