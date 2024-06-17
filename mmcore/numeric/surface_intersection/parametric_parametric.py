
import numpy as np
import scipy

from scipy.optimize import fsolve
from scipy.spatial import KDTree

from mmcore.geom.surfaces import Surface
from mmcore.geom.vec import cross, norm
from mmcore.numeric import evaluate_curvature
from mmcore.numeric.aabb import aabb_overlap, aabb
from mmcore.numeric.closest_point import closest_point_on_line
from mmcore.numeric.plane import plane_plane_intersect, plane_plane_plane_intersect
from mmcore.geom.curves.knot import interpolate_curve
from mmcore.geom.curves.bspline import NURBSpline
from mmcore.numeric.vectors import scalar_norm

TOL = 0.1


def freeform_step_debug(s1, s2, uv1, uv2):
    pl1, pl2 = s1.plane_at(uv1), s2.plane_at(uv2)
    ln = np.array(plane_plane_intersect(pl1, pl2))
    np1 = closest_point_on_line((ln[0], ln[0] + ln[1]), pl1[0])
    np2 = closest_point_on_line((ln[0], ln[0] + ln[1]), pl2[0])

    return np1, np1 + (np2 - np1) / 2, np2


def solve_marching(s1:Surface, s2:Surface, uv1, uv2, tol,side=1):
    pl1, pl2 = s1.plane_at(uv1), s2.plane_at(uv2)

    marching_direction = cross(pl1[-1], pl2[-1])

    K = evaluate_curvature(marching_direction*side, pl1[-1])[1]

    r = 1 / scalar_norm(K)

    step = np.sqrt(r ** 2 - (r - tol) ** 2) * 2
    #print(r,step)
    new_pln = np.array([pl1[0] + marching_direction*side * step, pl1[-1], pl2[-1], marching_direction*side])

    return plane_plane_plane_intersect(pl1, pl2, new_pln), step


def improve_uv(s, uv_old, xyz_better):
    x_old, y_old, z_old = s.evaluate(uv_old)

    dxdu, dydu, dzdu = s.derivative_u(uv_old)
    dxdv, dydv, dzdv = s.derivative_v(uv_old)
    x_better, y_better, z_better = xyz_better

    xy = [[dxdu, dxdv], [dydu, dydv]], [x_better - x_old, y_better - y_old]
    xz = [[dxdu, dxdv], [dzdu, dzdv]], [x_better - x_old, z_better - z_old]
    yz = [[dydu, dydv], [dzdu, dzdv]], [y_better - y_old, z_better - z_old]

    ##print( xy,xz,yz,'\n\n')
    max_det = sorted([xy, xz, yz], key=lambda Ab: scipy.linalg.det(Ab[0]), reverse=True)[0]
    return np.linalg.solve(*max_det)


def freeform_step(s1, s2, uvb1, uvb2, tol, cnt=0):
    xyz_better = freeform_step_debug(s1, s2, uvb1, uvb2)[1]

    uvb1_better = uvb1 + improve_uv(s1, uvb1, xyz_better)
    uvb2_better = uvb2 + improve_uv(s2, uvb2, xyz_better)
    if np.any(uvb1_better < 0.) or np.any(uvb2_better < 0.) or np.any(uvb1_better > 1.) or np.any(uvb2_better > 1.):
        return
    xyz1_new = s1(uvb1_better)
    xyz2_new = s2(uvb2_better)

    if np.linalg.norm(xyz1_new - xyz2_new) < tol:
        return (xyz1_new, uvb1_better), (xyz2_new, uvb2_better)
    else:

        return freeform_step(s1, s2, uvb1_better, uvb2_better, tol, cnt + 1)


def marching_step(s1, s2, uvb1, uvb2, tol, cnt=0,side=1):
    xyz_better, step = solve_marching(s1, s2, uvb1, uvb2, tol,side=side)
    uvb1_better = uvb1 + improve_uv(s1, uvb1, xyz_better)
    uvb2_better = uvb2 + improve_uv(s2, uvb2, xyz_better)
    if np.any(uvb1_better<0.) or np.any(uvb2_better<0.) or np.any(uvb1_better>1.)or np.any(uvb2_better>1.):

        return

    xyz1_new = s1(uvb1_better)
    xyz2_new = s2(uvb2_better)
    if np.linalg.norm(xyz1_new - xyz2_new) < tol:
        return (xyz1_new, uvb1_better), (xyz2_new, uvb2_better), step

    else:

        return marching_step(s1, s2, uvb1_better, uvb2_better, tol, cnt + 1,side=side)


def check_surface_edge(uv, surf_interval, tol):
    bounds_u, bounds_v = surf_interval
    u, v = uv

    return any([u <= 0, u >= 1, v <= 0, v >= 1])


def check_wrap_back(xyz, initial_xyz, step, tol):
    distance = np.linalg.norm(xyz - initial_xyz)

    return (distance < step / 2)


def stop_check(s1, s2, xyz1, xyz2, uv1, uv2, initial_xyz, step, tol, iterations=0, max_iter=100):
    res = [
        norm(xyz1 - xyz2) >= tol,
        check_surface_edge(uv1, s1.interval(), tol) or check_surface_edge(uv2, s2.interval(), tol),
        check_wrap_back(xyz1, initial_xyz, step, tol) or check_wrap_back(xyz2, initial_xyz, step, tol),
        iterations >= max_iter

    ]
    r = any(res)

    return r


def marching_method(
        s1, s2, initial_uv1, initial_uv2, kd=None,tol=1e-3, max_iter=1000, no_ff=False,side=1
):
    iterations = 0
    use_kd = kd is not None
    ixss = set()
    res = marching_step(s1, s2, initial_uv1, initial_uv2, tol=tol, side=side)
    if res is None:
        return

    (xyz1, uv1_new), (xyz2, uv2_new),step = res

    #print()



    uvs = [(uv1_new, uv2_new)]
    pts = [xyz1]
    if use_kd:
        ixss.update(kd.query_ball_point(xyz1, step*2))


    #print(uv1_new, uv2_new)

    steps = [step]

    for i in range(max_iter):


        uv1, uv2 = uv1_new, uv2_new

        res=marching_step(s1, s2, uv1, uv2,  tol=tol,side=side)

        if res is None:
            break
        else:


           (xyz1, uv1_new), (xyz2, uv2_new), step = res
           pts.append(xyz1)
           uvs.append((uv1_new.tolist(), uv2_new.tolist()))
           steps.append(step)
           if use_kd:
              ixss.update(kd.query_ball_point(xyz1, step*2))



    return uvs, pts, steps, list(ixss)


def _ssss(res,tol):
    ixs = []
    pts = []
    uvsa = []
    uvsb = []
    for i in range(len(res.a.uv)):
        uv1, uv2 = res.a.uv[i], res.b.uv[i]
        rr = freeform_step(res.a.surf, res.b.surf, uv1, uv2,tol=tol)
        if rr is None:

            continue
        else:



            ixs.append(i)
            pts.append(np.copy(rr[0][0]))
            uvsa.append(np.copy(rr[0][1]))
            uvsb.append(np.copy(rr[1][1]))

    pts = np.array(pts)
    uvsa=np.array(uvsa)
    uvsb=np.array(uvsb)

    return KDTree(pts), uvsa, uvsb


def surface_local_cpt(surf1, surf2):
    def fun(t):
        return np.array(surf2.evaluate(np.array([t[1], t[2]]))) - np.array(surf1.evaluate(np.array([0.00, t[0]])))

    return fsolve(fun, np.array([0.5, 0.5, 0.5]))


def surface_ppi(surf1, surf2, tol=0.1):

    res=find_closest(surf1,surf2,tol=tol)
    if res is None:
        return

    kd,uvs1,uvs2=_ssss(res,tol)

    curves=[]
    curves_uvs=[]
    stepss=[]
    data=kd.data
    ii=0
    l=len(uvs2)

    def _next():
        nonlocal ii,kd,data,uvs1,uvs2
        if kd is None:
            raise StopIteration
        if data.shape[0]<1:
            raise StopIteration
        if ii>=l:
            raise StopIteration
        print(ii)
        rr=[]
        ress = marching_method(surf1, surf2, uvs1[0], uvs2[0], kd=kd, tol=tol, side=1)
        ii += 1
        if ress is not None:
            uvs, pts, steps, ixss = ress

            rmv = np.array(ixss, dtype=int)
            print(rmv)
            data = np.delete(data, rmv, axis=0)
            uvs1 = np.delete(uvs1, rmv, axis=0)
            uvs2 = np.delete(uvs2, rmv, axis=0)
            if len(data.shape) == 2:
                kd = KDTree(data)

            else:
                kd = None

            return list(pts)

        ress_back = marching_method(surf1, surf2, uvs1[0], uvs2[0], kd=kd, tol=tol, side=-1)
        if ress_back is not None:
            uvs, pts, steps, ixss = ress_back

            rmv = np.array(ixss, dtype=int)
            data = np.delete(data, rmv, axis=0)
            uvs1 = np.delete(uvs1, rmv, axis=0)
            uvs2 = np.delete(uvs2, rmv, axis=0)
            print(rmv)
            if len(data.shape) == 2:
                kd = KDTree(data)
            else:
                kd = None

            return list(reversed(pts))

        elif ress is None and ress_back is None:

            if len(data.shape) == 2:
                data = np.delete(data, 0, axis=0)
                uvs1 = np.delete(uvs1, 0, axis=0)
                uvs2 = np.delete(uvs2, 0, axis=0)

                kd = KDTree(data)

            else:

                kd = None
            return None









    while True:
        try:
            res=_next()
            if res is None:
                pass
            else:

                curves.append(res)
        except StopIteration as err:
            break

    return curves,curves_uvs,stepss
def uvs(u_count,v_count):
    u=np.linspace(0.,1.,u_count)
    v=np.linspace(0.,1.,v_count)
    uv=[]
    for i in range(u_count):
        for j in range(v_count):
            uv.append((u[i],v[j]))
    return np.array(uv)


from collections import namedtuple

SurfaceStuff = namedtuple("SurfaceStuff", ['surf', 'kd', 'pts', 'uv', 'bbox'])
ClosestSurfaces = namedtuple("ClosestSurfaces", ['a', 'b'])


def find_closest(surf1, surf2, tol=1e-3):
    _uvs = uvs(2, 2)
    tol=tol*10

    pts1 = surf1(_uvs)
    pts2 = surf2(_uvs)
    min1max1 = aabb(pts1)
    min2max2 = aabb(pts2)
    if aabb_overlap((min1max1), (min2max2)):
        cnts1 = int(max((min1max1[1] - min1max1[0]) // (tol*2)))
        cnts2 = int(max((min2max2[1] - min2max2[0]) // (tol*2)))


        _uvs1 = uvs(cnts1, cnts1)
        _uvs2 = uvs(cnts2, cnts2)
        pts1 = surf1(_uvs1)
        pts2 = surf2(_uvs2)
        kd1 = KDTree(pts1)
        kd2 = KDTree(pts2)
        a, b = zip(*(kd1.sparse_distance_matrix(kd2, tol).keys()))
        a = np.array(a, dtype=int)
        b = np.array(b, dtype=int)

        return ClosestSurfaces(SurfaceStuff(surf1, kd1, pts1[a], _uvs1[a], min1max1),
                               SurfaceStuff(surf2, kd2, pts2[b], _uvs2[b], min2max2))
    else:
        return None

if __name__ == '__main__':
    import dill

    with open('tests/coons1.pkl', 'rb') as f:
        patch1 = dill.load(f)

    with open('tests/coons2.pkl', 'rb') as f:
        patch2 = dill.load(f)

    cc = surface_ppi(patch1, patch2, 0.01)