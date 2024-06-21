import time

import numpy as np
import scipy

from scipy.optimize import fsolve
from scipy.spatial import KDTree

from mmcore.geom.surfaces import Surface, Coons
from mmcore.geom.vec import cross, norm
from mmcore.numeric import evaluate_curvature,calgorithms
from mmcore.numeric.aabb import aabb_overlap, aabb
from mmcore.numeric.algorithms import intersection_curve_point
from mmcore.numeric.algorithms import intersection_curve_point
from mmcore.numeric.algorithms.point_inversion import point_inversion_surface
from mmcore.numeric.closest_point import closest_point_on_line
from mmcore.numeric.plane import plane_plane_intersect, plane_plane_plane_intersect
from mmcore.geom.curves.knot import interpolate_curve
from mmcore.geom.curves.bspline import NURBSpline
from mmcore.numeric.vectors import scalar_norm, scalar_cross, scalar_unit

TOL = 0.1


def freeform_step_debug(s1, s2, uv1, uv2):
    pl1, pl2 = s1.plane_at(uv1), s2.plane_at(uv2)
    ln = np.array(plane_plane_intersect(pl1, pl2))
    np1 = closest_point_on_line((ln[0], ln[0] + ln[1]), pl1[0])
    np2 = closest_point_on_line((ln[0], ln[0] + ln[1]), pl2[0])

    return np1, np1 + (np2 - np1) / 2, np2


def solve_marching(s1:Surface, s2:Surface, uv1, uv2, tol,side=1):
    pl1, pl2 = np.asarray(s1.plane_at(uv1)), np.asarray(s2.plane_at(uv2))

    marching_direction = scalar_unit(scalar_cross(pl1[-1], pl2[-1]))

    tng=np.zeros((2,3))


    calgorithms.evaluate_curvature(marching_direction*side, pl1[-1], tng[0],tng[1] )

    K=tng[1]
    #K=evaluate_curvature(marching_direction*side, pl1[-1])[1]

    r = 1 / scalar_norm(K)
    #print(r,K)
    step = np.sqrt(np.abs(r ** 2 - (r - tol) ** 2)) * 2
    #print(r,K,r ** 2 - (r - tol) ** 2)

    new_pln = np.array([pl1[0] + marching_direction*side * step, pl1[-1], pl2[-1], marching_direction*side])

    return plane_plane_plane_intersect(pl1, pl2, new_pln), step


def improve_uv(s:Surface, uv_old, xyz_better):
    x_old, y_old, z_old = s.evaluate(uv_old)

    dxdu, dydu, dzdu = s.derivative_u(uv_old)
    dxdv, dydv, dzdv = s.derivative_v(uv_old)
    x_better, y_better, z_better = xyz_better

    xy = np.array([[dxdu, dxdv], [dydu, dydv]]), [x_better - x_old, y_better - y_old]
    xz = np.array([[dxdu, dxdv], [dzdu, dzdv]]), [x_better - x_old, z_better - z_old]
    yz = np.array([[dydu, dydv], [dzdu, dzdv]]), [y_better - y_old, z_better - z_old]

    #print( xy,xz,yz,'\n\n')
    max_det = sorted( [xy, xz, yz], key=lambda Ab: scipy.linalg.det(Ab[0]), reverse=True)[0]
    return np.linalg.solve(*max_det)

def freeform_step(s1, s2, uvb1, uvb2, tol, cnt=0):
    xyz_better = freeform_step_debug(s1, s2, uvb1, uvb2)[1]

    uvb1_better = uvb1 + improve_uv(s1, uvb1, xyz_better)
    uvb2_better = uvb2 + improve_uv(s2, uvb2, xyz_better)
    if np.any(uvb1_better < 0.) or np.any(uvb2_better < 0.) or np.any(uvb1_better > 1.) or np.any(uvb2_better > 1.):
        return
    xyz1_new = s1.evaluate(uvb1_better)
    xyz2_new = s2.evaluate(uvb2_better)

    if np.linalg.norm(xyz1_new - xyz2_new) < tol:
        return (xyz1_new, uvb1_better), (xyz2_new, uvb2_better)
    else:

        return freeform_step(s1, s2, uvb1_better, uvb2_better, tol, cnt + 1)


def marching_step(s1, s2, uvb1, uvb2, tol, cnt=0,side=1):
    xyz_better, step = solve_marching(s1, s2, uvb1, uvb2, tol,side=side)
    #uvb1_better=point_inversion_surface(s1,xyz_better,*uvb1,tol,tol)
    #uvb2_better=point_inversion_surface(s2, xyz_better, *uvb2, tol, tol)
    #uvb1_better = uvb1 + improve_uv(s1, uvb1, xyz_better)
    #uvb2_better = uvb2 + improve_uv(s2, uvb2, xyz_better)

    #uv1=improve_uv(s1, uvb1, xyz_better)
    #uv2=improve_uv(s2, uvb2, xyz_better)

    uvb1_better = uvb1 + improve_uv(s1, uvb1, xyz_better)
    uvb2_better = uvb2 + improve_uv(s2, uvb2, xyz_better)
    if np.any(uvb1_better<0.) or np.any(uvb2_better<0.) or np.any(uvb1_better>1.)or np.any(uvb2_better>1.):

        return

    xyz1_new = s1.evaluate(uvb1_better)
    xyz2_new = s2.evaluate(uvb2_better)
    if np.linalg.norm(xyz1_new - xyz2_new) <=tol:
        return (xyz1_new, uvb1_better), (xyz2_new, uvb2_better), step

    else:

        return marching_step(s1, s2, uvb1_better, uvb2_better, tol, cnt + 1, side=side)


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
        s1, s2, initial_uv1, initial_uv2, kd=None,tol=1e-3, max_iter=100, no_ff=False,side=1
):
    iterations = 0

    use_kd = kd is not None
    ixss = set()
    xyz1_init, xyz2_init = np.copy(s1.evaluate(initial_uv1)), np.copy(s2.evaluate(initial_uv2))
    res = marching_step(s1, s2, initial_uv1, initial_uv2, tol=tol, side=side)
    if res is None:
        print("N")
        return

    (xyz1, uv1_new), (xyz2, uv2_new),step = res

    #print()



    uvs = [(uv1_new, uv2_new)]
    pts = [xyz1]
    if use_kd:
        ixss.update(kd.query_ball_point(xyz1, tol))


    #print(uv1_new, uv2_new)

    steps = [step]

    for i in range(max_iter):


        uv1, uv2 = uv1_new, uv2_new


        res=marching_step(s1, s2, uv1, uv2,  tol=tol,side=side)

        if res is None:
            print("N")
            break
        else:


           (xyz1, uv1_new), (xyz2, uv2_new), step = res
           pts.append(xyz1)
           uvs.append((uv1_new.tolist(), uv2_new.tolist()))
           steps.append(step)
           if use_kd:
               #print(   len(kd.data),ixss,step,tol,kd.query(xyz1, 3))

               ixss.update(kd.query_ball_point(xyz1, step*2))
               #print(ixss)

           if np.linalg.norm(xyz1 - xyz1_init) < step:
               print("b", xyz1_init, xyz1, np.linalg.norm(xyz1 - xyz1_init), step)
               pts.append(xyz1_init)
               uvs.append((initial_uv1.tolist(), initial_uv1.tolist()))
               break
        #print("I", np.linalg.norm(xyz1 - xyz1_init), step*2)


    print(len(pts))
    return uvs, pts, steps, list(ixss)


def _ssss(res,tol):
    ixs = []
    pts = []
    uvsa = []
    uvsb = []
    for i in range(len(res.a.uv)):
        uv1, uv2 = res.a.uv[i], res.b.uv[i]
        rr = freeform_step(res.a.surf, res.b.surf, uv1, uv2, tol=tol)

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

        if data.shape[0]<1:
            raise StopIteration
        if ii>=l:
            raise StopIteration
        #print(ii)

        ress = marching_method(surf1, surf2, uvs1[0], uvs2[0], kd=kd, tol=tol, side=1)
        ii += 1
        if ress is not None:
            uvs, pts, steps, ixss = ress

            rmv = np.array(ixss, dtype=int)
            #print(rmv)
            data = np.delete(data, rmv, axis=0)
            uvs1 = np.delete(uvs1, rmv, axis=0)
            uvs2 = np.delete(uvs2, rmv, axis=0)
            if len(data.shape) == 2:
                kd = KDTree(data)

            else:

                kd = None
                #ress = marching_method(surf1, surf2, uvs1[0], uvs2[0], kd=None, tol=tol, side=1)

            return list(pts)
        else:
            uvs1 = np.delete(uvs1, 0, axis=0)
            uvs2 = np.delete(uvs2, 0, axis=0)
            data = np.delete(data, 0, axis=0)
            if len(data.shape) == 2:
                kd = KDTree(data)

            else:
                kd = None


        if len(uvs1)>0:
            ress_back = marching_method(surf1, surf2, uvs1[0], uvs2[0], kd=kd, tol=tol, side=-1)
            if ress_back is not None:
                uvs, pts, steps, ixss = ress_back

                rmv = np.array(ixss, dtype=int)
                data = np.delete(data, rmv, axis=0)
                uvs1 = np.delete(uvs1, rmv, axis=0)
                uvs2 = np.delete(uvs2, rmv, axis=0)

                if len(data.shape) == 2:
                    kd = KDTree(data)
                else:
                    kd = None

                return list(reversed(pts))

            elif ress is None and ress_back is None:
                uvs1 = np.delete(uvs1, 0, axis=0)
                uvs2 = np.delete(uvs2, 0, axis=0)
                data = np.delete(data, 0, axis=0)
                if len(data.shape) == 2:


                    kd = KDTree(data)

                else:

                    kd = None
                return None
            else:
                uvs1 = np.delete(uvs1, 0, axis=0)
                uvs2 = np.delete(uvs2, 0, axis=0)
                data = np.delete(data, 0, axis=0)
                if len(data.shape) == 2:
                    kd = KDTree(data)

                else:
                    kd = None
                return







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

def mgrid3d(bounds, x_count, y_count, z_count):
    # Создаем линейные пространства
    (minx, miny, minz), (maxx, maxy, maxz) = bounds
    x = np.linspace(minx, maxx, x_count)
    y = np.linspace(miny, maxy, y_count)
    z = np.linspace(minz, maxz, z_count)
    # Создаем 3D сетку точек
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    # Объединяем координаты в один массив
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    return points

def find_closest(surf1, surf2, tol=1e-3):
    _uvs = uvs(2, 2)
    tol=tol*10

    pts1 = surf1(_uvs)
    pts2 = surf2(_uvs)
    min1max1 = aabb(pts1)
    min2max2 = aabb(pts2)

    if aabb_overlap((min1max1), (min2max2)):
        cnts1 =  min(max(int(max((min1max1[1] - min1max1[0]) // (tol*2))),4),25)
        cnts2 =  min(max(int(max((min2max2[1] - min2max2[0]) // (tol*2))),4),25)
        print(cnts1,cnts2)



        _uvs1 = uvs(cnts1, cnts1)
        _uvs2 = uvs(cnts2, cnts2)
        pts1 = np.array([surf1.evaluate(i) for i in _uvs1])
        pts2 = np.array([surf2.evaluate(i) for i in _uvs2])
        kd1 = KDTree(pts1)
        kd2 = KDTree(pts2)
        a, b = zip(*(kd1.sparse_distance_matrix(kd2, tol*2).keys()))
        a = np.array(a, dtype=int)
        b = np.array(b, dtype=int)


        return ClosestSurfaces(SurfaceStuff(surf1, kd1, pts1[a], _uvs1[a], min1max1),
                               SurfaceStuff(surf2, kd2, pts2[b], _uvs2[b], min2max2))
    else:
        return None

if __name__ == '__main__':
    import yappi

    pts1=np.array([[(-6.0558943035701525, -13.657656200983698, 1.0693341635684721), (-1.5301574718208828, -12.758430585795727, -2.4497481670182113), (4.3625055618617772, -14.490138754852163, -0.052702347089249368), (7.7822965141636233, -13.958097981505476, 1.1632592672736894)], [(7.7822965141636233, -13.958097981505476, 1.1632592672736894), (9.3249111495947457, -9.9684277340655711, -2.3272399773510646), (9.9156785503454081, -4.4260877770435245, -4.0868275118021469), (13.184366571517304, 1.1076098797323481, 0.55039832538794542)], [(-3.4282810787748206, 2.5976227512567878, -4.1924897351083787), (5.7125793432806686, 3.1853804927764848, -3.1997049666908506), (9.8891692556257418, 1.2744489476398368, -7.2890391724273922), (13.184366571517304, 1.1076098797323481, 0.55039832538794542)], [(-6.0558943035701525, -13.657656200983698, 1.0693341635684721), (-2.1677078000821663, -4.2388638567221646, -3.2149413059589502), (-3.5823721281354479, -1.1684651343084738, 3.3563417199639680), (-3.4282810787748206, 2.5976227512567878, -4.1924897351083787)]]
)

    pts2=np.array([[(-9.5666205803690971, -12.711321277810857, -0.77093266173210928), (-1.5012583168504101, -15.685662924609387, -6.6022178296290024), (0.62360921189203689, -15.825362292273830, 2.9177845739234654), (7.7822965141636233, -14.858282311330257, -5.1454157090841059)], [(7.7822965141636233, -14.858282311330257, -5.1454157090841059), (9.3249111495947457, -9.9684277340655711, -1.3266123160614773), (12.689851531339878, -4.4260877770435245, -9.5700220471407818), (10.103825228355211, 1.1076098797323481, -5.6331564229411617)], [(-5.1868371621186844, 2.5976267088609308, 0.97022697723726137), (-0.73355849180427846, 3.1853804927764848, 1.4184540026745367), (1.7370638323127894, 4.7726088993795681, -3.4674902939896270), (10.103825228355211, 1.1076098797323481, -5.6331564229411617)], [(-9.5666205803690971, -12.711321277810857, -0.77093266173210928), (-3.9344403681487776, -6.6256134176686521, -6.3569364954962628), (-5.1574735761500676, -1.1684651343084738, 1.0573724488786185), (-5.1868371621186844, 2.5976267088609308, 0.97022697723726137)]]
)


    #with open('tests/coons1.pkl', 'rb') as f:
    #    patch1 = dill.load(f)

    #with open('tests/coons2.pkl', 'rb') as f:
    #    patch2 = dill.load(f)
    patch1 = Coons(*(NURBSpline(pts) for pts in pts1))
    patch2 = Coons(*(NURBSpline(pts) for pts in pts2))
    s=time.time()
    yappi.set_clock_type("wall")  # Use set_clock_type("wall") for wall time
    yappi.start(builtins=True)
    cc = surface_ppi(patch1, patch2, 0.01)
    yappi.stop()
    func_stats = yappi.get_func_stats()
    func_stats.save(f"{__file__.replace('.py','')}_{int(time.time())}.pstat", type='pstat')

    print(time.time()-s)
    print([np.array(c).tolist() for c in cc[0]])