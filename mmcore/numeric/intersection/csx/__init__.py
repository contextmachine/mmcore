import numpy as np

from mmcore.numeric.vectors import scalar_norm, norm
from numpy._typing import NDArray

from mmcore.geom.curves.curve import curve_bvh
from mmcore.geom.nurbs import NURBSCurve, NURBSSurface
from mmcore.geom.surfaces import surface_bvh
from mmcore.numeric.plane import inverse_evaluate_plane
from mmcore.numeric.algorithms.point_inversion import point_inversion_surface
from mmcore.numeric.intersection.ccx import curve_pix
from mmcore.numeric.routines import uvs
from scipy.spatial import KDTree
from mmcore.numeric.intersection.csx._ncsx  import nurbs_csx

__all__ = ['curve_surface_ppi', 'closest_curve_surface_ppi','nurbs_csx','curve_surface_intersection']


# Define the difference function
def difference(curve, surface, params):
    t, u, v = params

    return curve.evaluate(t) - surface.evaluate(np.array([u, v]))


def curve_x_plane(curve, plane, axis=2, step=0.5):
    return curve_pix(curve, lambda xyz: inverse_evaluate_plane(plane, xyz)[axis], step=step)


# Define the Jacobian matrix for the Newton-Raphson method
def jacobian(curve, surface, params, J):
    #t, u, v = params
    h = 1e-6

    for i in range(3):
        dp = np.zeros(3)
        dp[i] = h

        J[:, i] = (difference(curve, surface, params + dp) - difference(curve, surface, params - dp)) / (2 * h)

    return J


def _jac_cls(curve, surface, params, J):
    x = curve.derivative(params[0])
    y,z  = -1 * surface.derivatives(params[1:])

    J[0, 0] = x[0]
    J[1, 0] = x[1]
    J[2, 0] = x[2]
    J[0, 1] = y[0]
    J[1, 1] = y[1]
    J[2, 1] = y[2]
    J[0, 2] = z[0]
    J[1, 2] = z[1]
    J[2, 2] = z[2]
    return J


# Newton-Raphson method to refine the solution
def newton_raphson(curve, surface, params, tol=1e-6, max_iter=100):
    J = np.zeros((3, 3))
    if hasattr(surface, 'derivative_v') and hasattr(curve, 'derivative'):
        jac = _jac_cls
    else:
        jac = jacobian

    for i in range(max_iter):
        J[:] = 0.0

        F = difference(curve, surface, params)
        jac(curve, surface, params, J)
        delta = np.linalg.solve(J, -F)

        params = params + delta
        if scalar_norm(delta) < tol:
            return params
    return params


from mmcore.geom.bvh import contains_point
import math


def count_from_length_brute_force(surf, crv, threshold, t_range=None, uv_range=None):
    (umin, umax), (vmin, vmax) = surf.interval() if uv_range is None else uv_range
    (tmin, tmax) = crv.interval() if t_range is None else t_range
    ud = max(surf.isoline_u(umin).evaluate_length((vmin, vmax)),
             surf.isoline_u(umax).evaluate_length((vmin, vmax)))
    vd = max(surf.isoline_v(vmin).evaluate_length((umin, umax)),

             surf.isoline_v(vmax).evaluate_length((umin, umax)))
    u_count = int(math.ceil(ud / threshold))
    v_count = int(math.ceil(vd / threshold))
    t_count = int(math.ceil(crv.evaluate_length((tmin, tmax)) / threshold))
    return t_count, u_count, v_count

from mmcore.geom.bvh import intersect_bvh_objects,contains_point
def bvh_search(curve,surface):
    return [np.array((np.average(i.object.t), *np.average(j.object.uvs, axis=0))) for i, j in
     intersect_bvh_objects(curve.tree, surface.tree)]


def gs2(curve, surface, threshold, t_range=None, uv_range=None,tol=1e-3, counts=None):
    (umin, umax), (vmin, vmax) = surf.interval() if uv_range is None else uv_range
    (tmin, tmax) = curve.interval() if t_range is None else t_range
    bvh1=curve.tree
    bvh2=surface.tree

    intersect_bvh_objects(bvh1,bvh2)

    curve_steps, u_count, v_count = count_from_length_brute_force(surface, curve, threshold, t_range=(tmin, tmax),
                                                                  uv_range=((umin, umax),
                                                                            (vmin, vmax))) if counts is None else counts
    t_min, t_max = t_range
    u_min, u_max = uv_range[0]
    v_min, v_max = uv_range[1]
    uv = uvs(u_count, v_count)
    uv[..., 0] *= u_max - u_min
    uv[..., 0] += u_min
    uv[..., 1] *= v_max - v_min
    uv[..., 1] += v_min
    t_vals = np.linspace(tmin, tmax, curve_steps)
    u_vals = uv[..., 0]
    v_vals = uv[..., 1]

    spts = surface(uv)

    kd = KDTree(spts)

    cpts = curve(t_vals)

    dst, ixs = kd.query(cpts, 1)
    dff = np.diff(dst)
    mdst = threshold
    u_old, v_old, t_old = np.inf, np.inf, np.inf
    indices = ixs[dst <= mdst]

    for t, u, v in zip(t_vals[dst < mdst], u_vals[indices], v_vals[indices]):
        if abs(t_old - t) < (tol) and abs(u_old - u) < (tol) and abs(v_old - v) < (tol):
            u_old = u
            v_old = v
            t_old = t
        else:
            u_old = u
            v_old = v
            t_old = t
            yield t, u, v


# Grid search for initial guesses
def _grid_search(curve, surface, t_range, uv_range, steps, threshold):
    t_min, t_max = t_range
    u_min, u_max = uv_range[0]
    v_min, v_max = uv_range[1]
    uv = uvs(steps, steps)
    uv[..., 0] *= (u_max - u_min)
    uv[..., 0] += u_min
    uv[..., 1] *= (v_max - v_min)
    uv[..., 1] += v_min
    t_vals = np.linspace(t_min, t_max, steps)
    u_vals = uv[..., 0]
    v_vals = uv[..., 1]

    spts = surface(uv)

    kd = KDTree(spts)

    cpts = curve(t_vals)

    initial_guess = []

    for i, pt in enumerate(cpts):
        ii = kd.query_ball_point(pt, threshold)
        if len(ii) > 0:
            for j in ii:
                initial_guess.append(np.array([t_vals[i], u_vals[j], v_vals[j]]))

    return initial_guess

grid_search=gs2
# Main function to find intersections
def find_intersections_grid(curve, surface, t_range, uv_range, steps=50, threshold=0.1, tol=1e-6):

    initial_guesses = list(grid_search(curve, surface, t_range=
    t_range,threshold=threshold,uv_range= uv_range,tol=tol))
    intersections = []
    for guess in initial_guesses:
        refined = newton_raphson(curve, surface, guess,tol=tol)
        if not any(
                np.allclose(refined, intersection, atol=tol)
                for intersection in intersections
        ):
            intersections.append(refined)
    return intersections

def find_intersections_bvh(curve, surface, tol=1e-6):

    initial_guesses = list(bvh_search(curve, surface))
    intersections = []
    for guess in initial_guesses:
        refined = newton_raphson(curve, surface, guess, tol=tol)

        if not any(
                np.allclose(refined, intersection, atol=tol)
                for intersection in intersections
        ):
            intersections.append(refined)
    return intersections


def curve_surface_ppi(curve, surface, steps=50, threshold=0.1, tol=1e-6, t_range=None, uv_range=None, method='bvh'):
    if method == 'bvh':
        return find_intersections_bvh(curve, surface, tol=tol)

    if t_range is None:
        t_range = np.array(curve.interval())
    if uv_range is None:
        uv_range = np.array(surface.interval())


    return find_intersections_grid(curve, surface, steps=steps, threshold=threshold, tol=tol, t_range=t_range,
                                      uv_range=uv_range)


def closest_curve_surface_ppi(curve, surface, initial_guess, tol=1e-6,  max_iter=100):
    return newton_raphson(curve, surface, initial_guess, tol=tol, max_iter=max_iter)


def curve_surface_intersection(curve, surface, tol=1e-6, t_bounds=None, uv_bounds=None):
    if isinstance(curve, NURBSCurve) and isinstance(surface, NURBSSurface):
        return nurbs_csx(curve, surface, tol=tol)
    (umin, umax), (vmin, vmax) = surface.interval() if uv_bounds is None else uv_bounds
    (tmin, tmax) = curve.interval() if  t_bounds is None else  t_bounds
    res=[]
    for t,u,v in find_intersections_bvh(curve, surface, tol=tol):
        if    (tmin<=t<= tmax) and  (umin<=u<=umax) and (vmin<=v<= vmax):
            res.append((t,u,v))

    return np.array(res)


def curve_implicit_intersection(curve, surface, tol=1e-6):
    return curve_pix(curve, surface, tol=tol)



def csx(curve, surface, tol=1e-3,ptol=1e-6):
    if isinstance(curve,NURBSCurve ) and isinstance(surface, NURBSSurface):
        ixs=nurbs_csx(curve, surface, tol=tol, ptol=ptol)
        ixs.sort(key=lambda x: x[2][0])
        typs, pts,params=zip(*ixs)

        return np.array(params)
    elif hasattr(surface,'implicit'):
        return curve_implicit_intersection(curve,surface,tol=ptol)
    else:
        return curve_surface_intersection(curve, surface, tol)


# Define parameter ranges and search for intersections
if __name__ == '__main__':
    from mmcore.geom.curves.bspline import NURBSpline
    import time
    from mmcore.geom.surfaces import Surface


    # Define the parametric surface

    class Surf(Surface):
        def interval(self):
            return (-2, 2), (-2, 2)

        def derivative_v(self, uv):
            return np.array([0., 1., 2 * uv[1]])

        def second_derivative_uu(self, uv):
            return np.array([0., 0., 2.])

        def second_derivative_vv(self, uv):
            return np.array([0., 0., 2.])

        def second_derivative_uv(self, uv):
            return np.array([0., 0., 0.])

        def derivative_u(self, uv):
            return np.array([1., 0., 2 * uv[0]])

        def evaluate(self, uv) -> NDArray[float]:
            u, v = uv

            x_s = u
            y_s = v
            z_s = u ** 2 + v ** 2
            return np.array([x_s, y_s, z_s])


    pts = np.array(
        [
            [-2.2234294448692147, 2.1016928510598412, 0.0],
            [-6.2098720873224416, -2.5628347315202351, 4.8162055568674758],
            [5.6095219450758984, 1.9937246381709586, 2.0460292479598943],
            [-1.9232701532978469, -4.8757216876711720, 4.6488472562898711],
        ]
    )
    # Define the parametric curve
    nc = NURBSpline(pts)
    surf = Surf()
    # Define parameter ranges and search for intersections

    t_range = np.array(nc.interval())

    uv_range = (-2, 2), (-2, 2)
    steps = 50
    s = time.time()

    # Solve intersection
    intersections = curve_surface_ppi(nc, surf, t_range=t_range, uv_range=uv_range, steps=steps)

    print(time.time() - s)
    # Output the results
    print(intersections)

    for i in intersections:
        print(nc.evaluate(i[0]).tolist(), surf(i[1:]).tolist())
        print(i, point_inversion_surface(surf, nc.evaluate(i[0]), *i[1:], 0.5, 0.5), '\n')
