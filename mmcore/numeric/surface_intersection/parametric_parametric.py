import itertools
import time
from enum import Enum

import numpy as np
import scipy

from scipy.optimize import fsolve
from scipy.spatial import KDTree

from mmcore.geom.bvh import BoundingBox, intersect_bvh, intersect_bvh_objects, BVHNode
from mmcore.geom.surfaces import Surface, Coons
from mmcore.geom.vec import cross, norm
from mmcore.numeric import evaluate_curvature,calgorithms
from mmcore.numeric.aabb import aabb_overlap, aabb
from mmcore.numeric.algorithms import intersection_curve_point
from mmcore.numeric.algorithms import intersection_curve_point
from mmcore.numeric.algorithms.point_inversion import point_inversion_surface
from mmcore.numeric.closest_point import closest_point_on_line,closest_point_on_ray
from mmcore.numeric.plane import plane_plane_intersect, plane_plane_plane_intersect
from mmcore.geom.curves.knot import interpolate_curve
from mmcore.geom.curves.bspline import NURBSpline
from mmcore.numeric.vectors import scalar_norm, scalar_cross, scalar_unit
class TerminatorType(int, Enum):
    FAIL=0
    LOOP=1
    EDGE=2
TOL = 0.1
def get_plane(origin,du,dv):
    duu = du/np.linalg.norm(du)

    dn = scalar_unit(scalar_cross(duu, dv))
    dvu = scalar_cross(dn, duu)
    return [origin,duu,dvu,dn]

def freeform_step_debug(pt1,pt2,du1,dv1,du2,dv2):
    pl1, pl2 = get_plane(pt1,du1,dv1),get_plane(pt2,du2,dv2)

    ln = np.array(plane_plane_intersect(pl1, pl2))

    np1 = np.asarray(closest_point_on_ray((ln[0], ln[1]), pt1))
    np2 = np.asarray(closest_point_on_ray((ln[0],  ln[1]), pt2))

    return np1, np1 + (np2 - np1) / 2, np2

def get_normal(du,dv):
    duu = scalar_unit(du)
    dn = scalar_unit(scalar_cross(duu, dv))
    #dvu = scalar_cross(dn, du)
    return duu,dn
def solve_marching(pt1,pt2,du1,dv1,du2, dv2,tol, side=1):

    pl1, pl2 = get_plane(pt1,du1,dv1),get_plane(pt2,du2,dv2
                                            )


    marching_direction = scalar_unit(scalar_cross(pl1[-1], pl2[-1]))

    tng=np.zeros((2,3))


    calgorithms.evaluate_curvature(marching_direction*side, pl1[-1], tng[0],tng[1] )

    K=tng[1]
    #K=evaluate_curvature(marching_direction*side, pl1[-1])[1]

    r = 1 / scalar_norm(K)
    #print(r,K)
    step = np.sqrt(np.abs(r ** 2 - (r - tol) ** 2)) * 2
    #print(r,K,r ** 2 - (r - tol) ** 2)

    new_pln = np.array([pt1 + marching_direction*side * step, pl1[-1], pl2[-1], marching_direction*side])

    return plane_plane_plane_intersect(pl1, pl2, new_pln), step


def improve_uv(du ,dv, xyz_old,xyz_better):
    #x_old, y_old, z_old = s.evaluate(uv_old)
    #x_old, y_old, z_old =xyz_old
    dxdu, dydu, dzdu = du
    dxdv, dydv, dzdv =  dv
    #x_better, y_better, z_better = xyz_better
    delta= xyz_better-xyz_old

    xy = [[dxdu, dxdv], [dydu, dydv]], [delta[0], delta[1]]
    xz = [[dxdu, dxdv], [dzdu, dzdv]], [delta[0], delta[2]]
    yz = [[dydu, dydv], [dzdu, dzdv]], [delta[1], delta[2]]

    #print( xy,xz,yz,'\n\n')
    max_det = sorted( [xy, xz, yz], key=lambda Ab: scipy.linalg.det(Ab[0]), reverse=True)[0]
    return np.linalg.solve(*max_det)

def freeform_step(s1, s2, uvb1, uvb2, tol, cnt=0):
    pt1 = s1.evaluate(uvb1)
    du1 = s1.derivative_u(uvb1)
    dv1 = s1.derivative_v(uvb1)
    pt2 = s2.evaluate(uvb2)
    du2 = s2.derivative_u(uvb2)
    dv2 = s2.derivative_v(uvb2)
    xyz_better = np.array(freeform_step_debug(pt1,pt2,  du1,dv1, du2,dv2)[1] )

    uvb1_better = uvb1 + improve_uv( du1,dv1, pt1, xyz_better)
    uvb2_better = uvb2 + improve_uv(du2,dv2, pt2, xyz_better)
    if np.any(uvb1_better < 0.) or np.any(uvb2_better < 0.) or np.any(uvb1_better > 1.) or np.any(uvb2_better > 1.):
        return
    xyz1_new = s1.evaluate(uvb1_better)
    xyz2_new = s2.evaluate(uvb2_better)
    #print(   np.linalg.norm(xyz1_new -  xyz2_new))
    if np.linalg.norm(xyz1_new - xyz2_new) <= tol:
        # print("e")
        return (xyz1_new, uvb1_better), (xyz2_new, uvb2_better)
    else:

        return freeform_step(s1, s2, uvb1_better, uvb2_better, tol, cnt + 1)


def marching_step(s1, s2, uvb1, uvb2, tol, cnt=0,side=1):
    pt1=s1.evaluate(uvb1)
    du1=s1.derivative_u(uvb1)
    dv1 = s1.derivative_v(uvb1)
    pt2 = s2.evaluate(uvb2)
    du2=s2.derivative_u(uvb2)
    dv2=s2.derivative_v(uvb2)
    xyz_better, step = solve_marching(pt1, pt2, du1,dv1,   du2,dv2,  tol,side=side)
    #uvb1_better=point_inversion_surface(s1,xyz_better,*uvb1,tol,tol)
    #uvb2_better=point_inversion_surface(s2, xyz_better, *uvb2, tol, tol)
    #uvb1_better = uvb1 + improve_uv(s1, uvb1, xyz_better)
    #uvb2_better = uvb2 + improve_uv(s2, uvb2, xyz_better)

    #uv1=improve_uv(s1, uvb1, xyz_better)
    #uv2=improve_uv(s2, uvb2, xyz_better)

    uvb1_better = uvb1 + improve_uv(  du1,dv1, pt1, xyz_better)
    uvb2_better = uvb2 + improve_uv( du2,dv2, pt2, xyz_better)
    if np.any(uvb1_better<0.) or np.any(uvb2_better<0.) or np.any(uvb1_better>1.)or np.any(uvb2_better>1.):

        return

    #xyz1_new = s1.evaluate(uvb1_better)
    #xyz2_new = s2.evaluate(uvb2_better)
    #print( np.linalg.norm(xyz1_new - xyz2_new))
    #if np.linalg.norm(xyz1_new - xyz2_new) <=tol:
        #print("E",np.linalg.norm(xyz1_new - xyz2_new))
    return (xyz_better, uvb1_better), (xyz_better, uvb2_better), step

    #else:

    #    return marching_step(s1, s2, uvb1_better, uvb2_better, tol, cnt + 1, side=side)


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
    terminator=None
    use_kd = kd is not None
    ixss = set()
    xyz1_init, xyz2_init = np.copy(s1.evaluate(initial_uv1)), np.copy(s2.evaluate(initial_uv2))
    res = marching_step(s1, s2, initial_uv1, initial_uv2, tol=tol, side=side)

    if res is None:
        #print("N")
        return  terminator

    (xyz1, uv1_new), (xyz2, uv2_new),step = res
    if use_kd:
        ixss.update(kd.query_ball_point(xyz1_init, step*2))
    #print()



    uvs = [(uv1_new, uv2_new)]
    pts = [ xyz1]



    #print(uv1_new, uv2_new)

    steps = [step]

    for i in range(max_iter):


        uv1, uv2 = uv1_new, uv2_new


        res=marching_step(s1, s2, uv1, uv2,  tol=tol,side=side)

        if res is None:
            #print("N")
            terminator = TerminatorType.EDGE
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
               #print("b", xyz1_init, xyz1, np.linalg.norm(xyz1 - xyz1_init), step)
               pts.append(pts[0])

               uvs.append((initial_uv1.tolist(), initial_uv1.tolist()))
               terminator=TerminatorType.LOOP
               break
        #print("I", np.linalg.norm(xyz1 - xyz1_init), step*2)


    #print(len(pts))
    return uvs, pts, steps, list(ixss), terminator


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

    #res=find_closest(surf1,surf2,tol=tol)
    #if res is None:
    #    return
    res=find_closest2(surf1, surf2, tol=tol)
    if res is None:
        return


    kd,uvs1,uvs2=res

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
            start = np.copy(data[0])
            if ress[-1]!=TerminatorType.LOOP:
                ress_back = marching_method(surf1, surf2, uvs1[0], uvs2[0], kd=kd, tol=tol, side=-1)


                if ress_back is not None:

                    uvs, pts, steps, ixss,terminator = ress
                    uvsb, ptsb, stepsb, ixssb,terminator = ress_back
                    rmv= np.unique(ixss+ixssb)
                    data = np.delete(data, rmv, axis=0)
                    uvs1 = np.delete(uvs1, rmv, axis=0)
                    uvs2 = np.delete(uvs2, rmv, axis=0)
                    if len(data.shape) == 2:
                        kd = KDTree(data)
                    else:

                        kd = None
                    return list(itertools.chain(reversed( ptsb),[start],pts))

            uvs, pts, steps, ixss,terminator = ress

            rmv = np.array(ixss, dtype=int)
            # print(rmv)
            data = np.delete(data, rmv, axis=0)
            uvs1 = np.delete(uvs1, rmv, axis=0)
            uvs2 = np.delete(uvs2, rmv, axis=0)
            if len(data.shape) == 2:
                    kd = KDTree(data)

            else:

                    kd = None
                        # ress = marching_method(surf1, surf2, uvs1[0], uvs2[0], kd=None, tol=tol, side=1)

            return [start]+list(pts)



        else:
            uvs1 = np.delete(uvs1, 0, axis=0)
            uvs2 = np.delete(uvs2, 0, axis=0)
            data = np.delete(data, 0, axis=0)
            if len(data.shape) == 2:
                kd = KDTree(data)

            else:
                kd = None










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
        #print(cnts1,cnts2)



        _uvs1 = uvs(cnts1, cnts1)
        _uvs2 = uvs(cnts2, cnts2)
        pts1 = np.array([surf1.evaluate(i) for i in _uvs1])
        pts2 = np.array([surf2.evaluate(i) for i in _uvs2])
        kd1 = KDTree(pts1)
        kd2 = KDTree(pts2)
        a, b = zip(*(kd1.sparse_distance_matrix(kd2, tol*2).keys()))
        a = np.array(a, dtype=int)
        b = np.array(b, dtype=int)

        #print([pts1[a].tolist(),pts2[b].tolist()])
        return ClosestSurfaces(SurfaceStuff(surf1, kd1, pts1[a], _uvs1[a], min1max1),
                               SurfaceStuff(surf2, kd2, pts2[b], _uvs2[b], min2max2))
    else:
        return None
def find_closest2(surf1:Surface, surf2:Surface, tol=1e-3):

    min1max1:BoundingBox = surf1.tree.bounding_box
    min2max2:BoundingBox = surf2.tree.bounding_box

    if min1max1.intersect(min2max2):
        pts = []
        uvs1 = []
        uvs2 = []
        for first,second in intersect_bvh_objects(surf1.tree,surf2.tree):
            first:BVHNode
            second: BVHNode
            #bb=first.bounding_box.intersection(second.bounding_box)


            uv1=np.average(first.object.uvs,axis=0)
            uv2=np.average(second.object.uvs, axis=0)

            res=freeform_step(surf1,surf2, uv1, uv2, tol=tol)
            if res is not None:
                (xyz1_new, uvb1_better), (xyz2_new, uvb2_better) =res
                #print(uvb1_better, uvb2_better)
                pts.append(xyz1_new)
                uvs1.append(uvb1_better)
                uvs2.append(uvb2_better)
        return KDTree(np.array(pts)), np.array(uvs1),np.array(uvs2)













if __name__ == '__main__':
    import yappi

    pts1=np.array([[(-6.0558943035701525, -13.657656200983698, 1.0693341635684721), (-1.5301574718208828, -12.758430585795727, -2.4497481670182113), (4.3625055618617772, -14.490138754852163, -0.052702347089249368), (7.7822965141636233, -13.958097981505476, 1.1632592672736894)], [(7.7822965141636233, -13.958097981505476, 1.1632592672736894), (9.3249111495947457, -9.9684277340655711, -2.3272399773510646), (9.9156785503454081, -4.4260877770435245, -4.0868275118021469), (13.184366571517304, 1.1076098797323481, 0.55039832538794542)], [(-3.4282810787748206, 2.5976227512567878, -4.1924897351083787), (5.7125793432806686, 3.1853804927764848, -3.1997049666908506), (9.8891692556257418, 1.2744489476398368, -7.2890391724273922), (13.184366571517304, 1.1076098797323481, 0.55039832538794542)], [(-6.0558943035701525, -13.657656200983698, 1.0693341635684721), (-2.1677078000821663, -4.2388638567221646, -3.2149413059589502), (-3.5823721281354479, -1.1684651343084738, 3.3563417199639680), (-3.4282810787748206, 2.5976227512567878, -4.1924897351083787)]]
)

    pts2=np.array([[(-9.1092663228073292, -12.711321277810857, -0.77093266173210928), (-1.5012583168504101, -15.685662924609387, -6.6022178296290024), (0.62360921189203689, -15.825362292273830, 2.9177845739234654), (7.7822965141636233, -14.858282311330257, -5.1454157090841059)], [(7.7822965141636233, -14.858282311330257, -5.1454157090841059), (9.3249111495947457, -9.9684277340655711, -1.3266123160614773), (12.689851531339878, -4.4260877770435245, -8.9585086671785774), (10.103825228355211, 1.1076098797323481, -5.6331564229411617)], [(-5.1868371621186844, 4.7602528056675295, 0.97022697723726137), (-0.73355849180427846, 3.1853804927764848, 1.4184540026745367), (1.7370638323127894, 4.7726088993795681, -3.7548102282588882), (10.103825228355211, 1.1076098797323481, -5.6331564229411617)], [(-9.1092663228073292, -12.711321277810857, -0.77093266173210928), (-3.9344403681487776, -6.6256134176686521, -6.3569364954962628), (-3.9413840306534453, -1.1684651343084738, 0.77546233191951042), (-5.1868371621186844, 4.7602528056675295, 0.97022697723726137)]]
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