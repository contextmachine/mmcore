import numpy as np

from mmcore.numeric.vectors import scalar_norm
from numpy._typing import NDArray

from mmcore.numeric import inverse_evaluate_plane
from mmcore.numeric.algorithms.point_inversion import point_inversion_surface
from mmcore.numeric.intersection.curve_curve import curve_pii
from mmcore.numeric.routines import uvs
from scipy.spatial import KDTree

__all__ = ['curve_surface_ppi', 'closest_curve_surface_ppi']


# Define the difference function
def difference(curve, surface, params):
    t, u, v = params

    return curve.evaluate(t) - surface.evaluate(np.array([u, v]))


def curve_x_plane(curve, plane, axis=2, step=0.5):
    return curve_pii(
        curve, lambda xyz: inverse_evaluate_plane(plane, xyz)[axis], step=step
    )


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

    y = -1 * surface.derivative_u(params[1:])
    z = -1 * surface.derivative_v(params[1:])
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


# Grid search for initial guesses
def grid_search(curve, surface, t_range, uv_range, steps, threshold):
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


# Main function to find intersections
def find_intersections(curve, surface, t_range, uv_range, steps=50, threshold=0.1, tol=1e-6):
    initial_guesses = grid_search(curve, surface, t_range, uv_range, steps=steps, threshold=threshold)
    intersections = []
    for guess in initial_guesses:
        refined = newton_raphson(curve, surface, guess)
        if not any(
                np.allclose(refined, intersection, atol=tol)
                for intersection in intersections
        ):
            intersections.append(refined)
    return intersections


def curve_surface_ppi(curve, surface, steps=50, threshold=0.1, tol=1e-6, t_range=None, uv_range=None):
    if t_range is None:
        t_range = np.array(curve.interval())
    if uv_range is None:
        uv_range = np.array(surface.interval())

    return find_intersections(curve, surface, steps=steps, threshold=threshold, tol=tol, t_range=t_range,
                              uv_range=uv_range)


def closest_curve_surface_ppi(curve, surface, initial_guess, tol=1e-6, max_iter=100):
    return newton_raphson(curve, surface, initial_guess, tol=tol, max_iter=max_iter)


def curve_surface_intersection(curve, surface, tol=1e-6):
    return curve_surface_ppi(curve, surface, tol=tol)


def curve_implicit_intersection(curve, surface, tol=1e-6):
    return curve_pii(curve, surface, default_tol=tol)


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
