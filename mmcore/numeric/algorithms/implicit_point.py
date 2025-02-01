import math

import numpy as np

from mmcore.numeric.vectors import solve2x2, scalar_norm
from numpy._typing import NDArray

from mmcore.geom.vec import unit, make_perpendicular
from mmcore.numeric import scalar_dot, scalar_cross


from mmcore.numeric.fdm import Grad

__all__=['curve_point', 'curve_point_batch','surface_point', 'surface_plane', 'intersection_curve_point','intersection_curve_point_batch',]

def curve_point(func, initial_point: np.ndarray = None, delta=0.001, grad=None):
    """
    Calculates the point on the curve along the "steepest path".
    @param func:
    @param grad:
    @param initial_point:
    @param delta:
    @return: point on the curve along the "steepest path"
    @type: np.ndarray


    """
    #
    grad = _resolve_grad(func, grad)
    f = func(initial_point)
    g = grad(initial_point)
    cc = sum(g * g)
    if cc > 0:
        f = -f / cc
    else:
        f = 0

    new_point = initial_point + f * g
    d = np.linalg.norm(initial_point - new_point)
    while delta < d:
        initial_point = new_point
        f = func(initial_point)
        g = grad(initial_point)
        cc = sum(g * g)
        if cc > 0:
            f = -f / cc
        else:
            f = 0
        new_point = initial_point + f * g
        d = np.linalg.norm(initial_point - new_point)
    return new_point


import numpy as np


def _resolve_grad_2d(func, grad):
    """
    Resolves the gradient function for a given implicit function. If the user does not
    provide a gradient function, a default one is created using the finite difference method.

    @param func: Implicit function. Callable, accepts a point as an ndarray with shape (N,)
                 and returns a scalar float.
    @param grad: Implicit function gradient. Callable, accepts a point as an ndarray with shape (N,)
                 and returns the gradient vector with shape (N,). If None, a default gradient
                 estimation function will be generated.
    @return: A callable that computes the gradient.
    @rtype: Callable
    """
    if grad is None:
        def grad_default(x, h=1e-5):
            grad_est = np.zeros_like(x, dtype=float)
            # Use central finite differences for each coordinate.
            for i in range(len(x)):
                x1 = np.copy(x)
                x2 = np.copy(x)
                x1[i] += h
                x2[i] -= h
                grad_est[i] = (func(x1) - func(x2)) / (2 * h)
            return grad_est

        return grad_default
    else:
        return grad


def curve_point_batch(func, initial_points: NDArray[float] , delta=0.001, grad=None):
    """
    Calculates the point on the implicit curve along the "steepest path" for each initial point.
    The function can process a single point (with shape (N,)) or a batch of points (with shape (..., N)).

    @param func: Implicit function. Callable, accepts a point as an ndarray with shape (N, )
                 and returns a scalar float.
    @param initial_points: Initial point(s) as an ndarray with shape (N, ) or (..., N), where ... is a multiple axis.
    @param delta: Step size threshold for convergence. The update stops for a point when the
                  change in position is <= delta.
    @param grad: Implicit function gradient. Callable, accepts a point as an ndarray with shape (N, )
                 and returns the gradient vector with shape (N,). If None, a default finite
                 difference estimator will be used.
    @return: The converged point(s) on the curve. If initial_point was of shape (..., N), the result
             is an ndarray with shape (..., N). If initial_point was of shape (N,), the result is an
             ndarray with shape (N,).
    @rtype: np.ndarray
    """
    # Resolve the gradient function (either user-supplied or default finite-difference)


    grad = _resolve_grad_2d(func, grad)

    if initial_points is None:
        raise ValueError("initial_point must be provided.")

    # If initial_point is a single point (1D), convert it to a batch of one point.
    single_input = False
    initial_point = initial_points
    input_shape = initial_points.shape
    if initial_point.ndim == 1:
        initial_point = initial_point.reshape(1, -1)
        single_input = True
    else :


        initial_point = initial_points.reshape(-1, input_shape[-1])
    # Create a working copy of the batch.
    points = initial_point.copy()
    batch_size = points.shape[0]
    # Boolean array to mark which points have converged.
    converged = np.full(batch_size, False, dtype=bool)

    # Iterate until every point's update is small enough.
    while not np.all(converged):
        # Process only those points that have not converged.
        indices = np.where(~converged)[0]
        # Evaluate the function for each non-converged point.
        f_vals = np.array([func(points[i]) for i in indices])
        # Evaluate the gradient for each non-converged point.
        grads = np.array([grad(points[i]) for i in indices])
        # Compute the squared norm of the gradients.
        norm_g_sq = np.sum(grads * grads, axis=1)
        # Determine the update step. For each point, if the gradient norm is positive,
        # the step is given by: step = -func(points) / (||grad||^2); otherwise, step is 0.
        steps = np.where(norm_g_sq > 0, -f_vals / norm_g_sq, 0.0)
        # Compute the new candidate points.
        new_points = points[indices] + steps[:, np.newaxis] * grads
        # Compute the update distances for each non-converged point.
        ds = np.linalg.norm(new_points - points[indices], axis=1)
        # Identify which points still require an update.
        update_mask = ds > delta

        # For points that have not converged, update their positions.
        if np.any(update_mask):
            update_indices = indices[update_mask]
            points[update_indices] = new_points[update_mask]
        # For points whose update step is sufficiently small, mark them as converged.
        done_indices = indices[~update_mask]
        converged[done_indices] = True

    # Return the result in the same shape as the input.
    if single_input:
        return points[0]
    return points.reshape(input_shape)


def intersection_curve_point(surf1, surf2, q0, grad1, grad2, tol=1e-6, max_iter=100, return_grads=False, no_err=False):
    """
        Intersection curve point between two curves.

        :param surf1: The first curve surface function.
        :type surf1: function
        :param surf2: The second curve surface function.
        :type surf2: function
        :param q0: Initial point on the curve.
        :type q0: numpy.ndarray
        :param grad1: Gradient function of the first curve surface.
        :type grad1: function
        :param grad2: Gradient function of the second curve surface.
        :type grad2: function
        :param tol: Tolerance for convergence. Default is 1e-6.
        :type tol: float
        :param max_iter: Maximum number of iterations allowed. Default is 100.
        :type max_iter: int
        :param return_grads: Flag to indicate whether to return gradients. Default is False.
        :type return_grads: bool
        :param no_err: Flag to indicate whether to raise error or return failure status in case of no convergence. Default is False.
        :type no_err: bool
        :return: The intersection point on the curve surface.
        :rtype: numpy.ndarray or tuple

        Example
    ---

    >>> from mmcore.geom.primitives import Sphere
    >>> c1= Sphere(np.array([0.,0.,0.]),1.)
    >>> c2= Sphere(np.array([1.,1.,1.]),1)
    >>> q0 =np.array((0.579597, 0.045057, 0.878821))
    >>> res=intersection_curve_point(c1.implicit,c2.implicit,q0,c1.gradient,c2.gradient,tol=1e-6) # 7 newton iterations
    >>> print(c1.implicit(res),c2.implicit(res))
    4.679345799729617e-10 4.768321293369127e-10
    >>> res=intersection_curve_point(c1.implicit,c2.implicit,q0,c1.gradient,c2.gradient,tol=1e-12) # 9 newton iterations
    >>> print(c1.implicit(res),c2.implicit(res))
    -1.1102230246251565e-16 2.220446049250313e-16
    >>> res=intersection_curve_point(c1.implicit,c2.implicit,q0,c1.gradient,c2.gradient,tol=1e-16) # 10 newton iterations
    >>> print(c1.implicit(res),c2.implicit(res))
    0.0 0.0
    """
    alpha_beta = np.zeros(2, dtype=np.float64)
    qk = np.copy(q0)

    f1, f2, g1, g2 = surf1(qk), surf2(qk), grad1(qk), grad2(qk)

    J = np.array([
        [scalar_dot(g1, g1), scalar_dot(g2, g1)],
        [scalar_dot(g1, g2), scalar_dot(g2, g2)]
    ])

    g = np.array([f1, f2])
    success = solve2x2(J, -g, alpha_beta)
    delta = alpha_beta[0] * grad1(qk) + alpha_beta[1] * grad2(qk)
    qk_next = delta + qk
    d = scalar_norm(qk_next - qk)
    i = 0

    success = True
    while d > tol:

        if i > max_iter:
            if not no_err:
                raise ValueError(f'Maximum iterations exceeded, No convergence {d}')
            else:
                success = False
                break


        qk = qk_next
        f1, f2, g1, g2 = surf1(qk), surf2(qk), grad1(qk), grad2(qk)

        J = np.array([
            [scalar_dot(g1, g1), scalar_dot(g2, g1)],
            [scalar_dot(g1, g2), scalar_dot(g2, g2)]
        ])

        g[:] = f1, f2

        _success = solve2x2(J, -g, alpha_beta)

        #alpha, beta = newton_step(qk, alpha, beta, f1, f2, g1, g2)
        #alpha, beta = newton_step(qk, alpha, beta, f1, f2, g1, g2)
        delta = alpha_beta[0] * g1 + alpha_beta[1] * g2
        qk_next = delta + qk

        d = scalar_norm(delta)

        i += 1

    if return_grads:
        return (success, (qk_next, f1, f2, g1, g2)) if no_err else qk_next, f1, f2, g1, g2

    return (success, qk_next) if no_err else qk_next


import numpy as np


def _resolve_grad_surf(surf, grad):
    """
    Resolves the gradient function for a given implicit surface function.
    If grad is None, a default finite-difference estimator is returned.

    :param surf: Implicit surface function. Callable accepting a point (ndarray of shape (N,)) and returning a scalar.
    :param grad: Gradient function for the surface. Callable accepting a point (ndarray of shape (N,)) and returning an ndarray of shape (N,).
                 If None, a central finite difference estimator is used.
    :return: A callable that computes the gradient.
    """
    if grad is None:
        def grad_default(q, h=1e-5):
            grad_est = np.zeros_like(q, dtype=float)
            for i in range(len(q)):
                q1 = np.copy(q)
                q2 = np.copy(q)
                q1[i] += h
                q2[i] -= h
                grad_est[i] = (surf(q1) - surf(q2)) / (2 * h)
            return grad_est

        return grad_default
    else:
        return grad


def solve2x2_batch(J, r, eps=1e-12):
    """
    Solves a batch of 2x2 linear systems.
    For each system (of size 2x2) in the batch, solve:
         J[i] * x = r[i],
    where J has shape (M, 2, 2) and r has shape (M, 2).

    :param J: numpy.ndarray of shape (M, 2, 2) representing M Jacobian matrices.
    :param r: numpy.ndarray of shape (M, 2) representing the right-hand side vectors.
    :param eps: Tolerance for the determinant to be considered nonzero.
    :return: A tuple (success, sol) where
             success is a boolean ndarray of shape (M,) indicating whether each system was solvable,
             and sol is an ndarray of shape (M, 2) containing the solutions (undefined where success is False).
    """
    M = J.shape[0]
    sol = np.zeros((M, 2), dtype=float)
    # Compute the determinant for each system.
    det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    success = np.abs(det) > eps
    idx = np.where(success)[0]
    sol[idx, 0] = (J[idx, 1, 1] * r[idx, 0] - J[idx, 0, 1] * r[idx, 1]) / det[idx]
    sol[idx, 1] = (-J[idx, 1, 0] * r[idx, 0] + J[idx, 0, 0] * r[idx, 1]) / det[idx]
    return success, sol


def intersection_curve_point_batch(surf1, surf2, q0, grad1=None, grad2=None, tol=1e-6, max_iter=100,
                             return_grads=False, no_err=True):
    """
    Computes the intersection point on the curve defined by the intersection of two implicit surfaces,
    using a Newton-based iterative method. This batched version accepts a set of initial points.

    :param surf1: The first surface implicit function.
                  Callable that accepts a point (ndarray of shape (N,)) and returns a scalar.
    :param surf2: The second surface implicit function.
                  Callable that accepts a point (ndarray of shape (N,)) and returns a scalar.
    :param q0: Initial point(s) on the intersection curve.
               numpy.ndarray of shape (N,) for a single point or (B, N) for a batch of B points.
    :param grad1: Gradient function for surf1.
                  Callable accepting a point (ndarray of shape (N,)) and returning an ndarray of shape (N,).
                  If None, a finite-difference estimator is used.
    :param grad2: Gradient function for surf2.
                  Callable accepting a point (ndarray of shape (N,)) and returning an ndarray of shape (N,).
                  If None, a finite-difference estimator is used.
    :param tol: Tolerance for convergence (default 1e-6).
    :param max_iter: Maximum number of iterations allowed (default 100).
    :param return_grads: If True, the function returns additional information (f1, f2, g1, g2).
                         See return section below.
    :param no_err: If True (default), nonconvergence is handled leniently (the corresponding points are marked as failures).
                   If False, a ValueError is raised for points that do not converge within max_iter iterations.
    :return: Depending on return_grads:
             - If return_grads is True, returns a tuple of six arrays:
                 success  : (B,) boolean array indicating convergence success for each point.
                 qk       : (B, N) array of the final intersection points.
                 f1       : (B,) array with surf1 evaluated at the final points.
                 f2       : (B,) array with surf2 evaluated at the final points.
                 g1       : (B, N) array with grad1 evaluated at the final points.
                 g2       : (B, N) array with grad2 evaluated at the final points.
             - Otherwise, returns a tuple of two arrays:
                 success  : (B,) boolean array.
                 qk       : (B, N) array of the final intersection points.
    """
    # Resolve gradient functions for the surfaces.
    grad1 = _resolve_grad_surf(surf1, grad1)
    grad2 = _resolve_grad_surf(surf2, grad2)

    # Ensure q0 is two-dimensional: shape (B, N)
    initial_shape=q0.shape
    if q0.ndim == 1:
        q0 = q0.reshape(1, -1)
    else:

        q0 = q0.reshape((-1, initial_shape[-1]))
    B, N = q0.shape

    # Initialize the iterate array and auxiliary storage.
    qk = q0.copy()  # Current iterates, shape (B, N)
    f1_arr = np.empty(B, dtype=float)
    f2_arr = np.empty(B, dtype=float)
    g1_arr = np.empty((B, N), dtype=float)
    g2_arr = np.empty((B, N), dtype=float)

    # active[i] is True if point i is still iterating.
    active = np.ones(B, dtype=bool)
    # success[i] remains True only if point i converges.
    success = np.ones(B, dtype=bool)

    iter_count = 0
    while np.any(active) and iter_count < max_iter:
        active_idx = np.where(active)[0]
        # Evaluate surf1, surf2 and their gradients for active points.
        for i in active_idx:
            f1_arr[i] = surf1(qk[i])
            f2_arr[i] = surf2(qk[i])
            g1_arr[i] = grad1(qk[i])
            g2_arr[i] = grad2(qk[i])

        # For active points, build the 2x2 Jacobian matrices and right-hand sides.
        g1_active = g1_arr[active_idx]  # (M, N)
        g2_active = g2_arr[active_idx]  # (M, N)
        f1_active = f1_arr[active_idx]  # (M,)
        f2_active = f2_arr[active_idx]  # (M,)

        # Compute the entries of the 2x2 matrices:
        a = np.sum(g1_active * g1_active, axis=1)  # (M,)
        b = np.sum(g2_active * g1_active, axis=1)  # (M,)
        c = np.sum(g1_active * g2_active, axis=1)  # (M,)
        d_comp = np.sum(g2_active * g2_active, axis=1)  # (M,)

        # Assemble the Jacobian for each active point.
        J_active = np.stack([np.stack([a, b], axis=1),
                             np.stack([c, d_comp], axis=1)], axis=1)  # (M, 2, 2)

        # Right-hand side: r = -[f1, f2]
        r_active = np.stack([-f1_active, -f2_active], axis=1)  # (M, 2)

        # Solve each 2x2 system.
        sol_success, sol = solve2x2_batch(J_active, r_active)
        # Update overall success for these points.
        success[active_idx] = success[active_idx] & sol_success

        # Compute the Newton update delta = alpha * g1 + beta * g2.
        delta_active = np.zeros((len(active_idx), N), dtype=float)
        for j, idx in enumerate(active_idx):
            if sol_success[j]:
                delta_active[j] = sol[j, 0] * g1_active[j] + sol[j, 1] * g2_active[j]
            else:
                delta_active[j] = np.zeros(N, dtype=float)

        # Compute the norm of the update for each active point.
        norms = np.linalg.norm(delta_active, axis=1)

        # Update the iterates and check convergence.
        for j, idx in enumerate(active_idx):
            if sol_success[j]:
                qk[idx] = qk[idx] + delta_active[j]
                if norms[j] <= tol:
                    active[idx] = False  # Converged.
            else:
                # If the 2x2 system could not be solved, mark this point as done.
                active[idx] = False

        iter_count += 1

    # After max_iter iterations, if some points are still active then handle according to no_err.
    if np.any(active):
        if not no_err:
            raise ValueError("Maximum iterations exceeded for some points.")
        else:
            # Mark the remaining (nonconverged) points as failures.
            failed_idx = np.where(active)[0]
            success[failed_idx] = False
            active[failed_idx] = False

    # Return the result.
    if return_grads:
        return (success.reshape(initial_shape[:-1]),
                qk.reshape(initial_shape),
                f1_arr.reshape(initial_shape[:-1]), f2_arr.reshape(initial_shape[:-1]),
                g1_arr.reshape(initial_shape), g2_arr.reshape(initial_shape))
    else:
        return success.reshape(initial_shape[:-1]), qk.reshape(initial_shape)



def _resolve_grad(func, grad=None):
    return grad if grad is not None else Grad(func)


def _curve_point_newton_step(func: callable, x: np.ndarray, grad: callable):
    fi = func(x)
    g = grad(x)
    cc = math.pow(g[0], 2) + math.pow(g[1], 2)
    if cc > 0:
        t = -fi / cc
    else:
        return x

    return x + t * g


def _curve_point2(func, x0, tol=1e-5, grad=None):
    grad = _resolve_grad(func, grad)
    x0 = np.copy(x0)
    delta = 1.0
    while delta >= tol:
        xi1, yi1 = _curve_point_newton_step(func, x0, grad=grad)
        delta = abs(x0[0] - xi1) + abs(x0[1] - yi1)
        x0[0] = xi1
        x0[1] = yi1
    return x0


def _normalize3d(v):
    norm = scalar_norm(v)
    if norm == 0:
        return v
    return v / norm


def _linear_combination_3d(a, v1, b, v2):
    return a * v1 + b * v2


def _implicit_tangent(d1, d2):
    return unit(make_perpendicular(d1, d2))


def surface_point(fun, p0, grad=None, tol=1e-8):
    p_i = p0
    grad = _resolve_grad(fun, grad)
    while True:
        fi, gradfi = (fun(p_i), grad(p_i))
        cc = scalar_dot(gradfi, gradfi)
        if cc > 0:
            t = -fi / cc
        else:
            t = 0
            print(f"{cc} WARNING tri (surface_point...): newton")

        p_i1 = _linear_combination_3d(1, p_i, t, gradfi)
        dv = p_i1 - p_i
        delta = scalar_norm(dv)

        if delta < tol:
            break

        p_i = p_i1

    #fi, gradfi = fun(p_i), grad(p_i)
    return p_i1


def surface_plane(fun, p_start, grad, tol=1e-5):
    p, nv = surface_point(fun, p_start, grad, tol)
    nv = _normalize3d(nv)

    if abs(nv[0]) > 0.5 or abs(nv[1]) > 0.5:
        tv1 = np.array([nv[1], -nv[0], 0])
    else:
        tv1 = np.array([-nv[2], 0, nv[0]])

    tv1 = _normalize3d(tv1)
    tv2 = scalar_cross(nv, tv1)
    return p, nv, tv1, tv2

# =========================
# Example usage:
# =========================
if __name__ == "__main__":
    # Define two implicit surfaces. For example, two spheres.
    def sphere1(point):
        # Sphere centered at (0, 0, 0) with radius 1: x^2 + y^2 + z^2 - 1 = 0
        return point[0] ** 2 + point[1] ** 2 + point[2] ** 2 - 1


    def sphere2(point):
        # Sphere centered at (0.5, 0.5, 0.5) with radius 1:
        # (x-0.5)^2 + (y-0.5)^2 + (z-0.5)^2 - 1 = 0
        return (point[0] - 0.5) ** 2 + (point[1] - 0.5) ** 2 + (point[2] - 0.5) ** 2 - 1


    # Optionally, provide explicit gradient functions.
    def sphere1_grad(point):
        return np.array([2 * point[0], 2 * point[1], 2 * point[2]])


    def sphere2_grad(point):
        return np.array([2 * (point[0] - 0.5),
                         2 * (point[1] - 0.5),
                         2 * (point[2] - 0.5)])


    # Create a batch of initial points in 3D (shape (B, 3)).
    q0_batch = np.array([
        [0.8, 0.1, 0.9],
        [0.2, 0.3, 1.0],
        [1.0, 0.5, 0.5],
        [0.0, 0.0, 0.0]  # This point may be far from the intersection.
    ])

    # Compute the intersection points, returning extra gradient information.
    success, qk, f1_vals, f2_vals, g1_vals, g2_vals = intersection_curve_point(
        sphere1, sphere2, q0_batch,
        grad1=sphere1_grad, grad2=sphere2_grad,
        tol=1e-6, max_iter=100, return_grads=True, no_err=True
    )

    print("Convergence success flags:")
    print(success)
    print("\nIntersection points (qk):")
    print(qk)
    print("\nsphere1 evaluations (f1):")
    print(f1_vals)
    print("\nsphere2 evaluations (f2):")
    print(f2_vals)
    print("\nGradient of sphere1 (g1):")
    print(g1_vals)
    print("\nGradient of sphere2 (g2):")
    print(g2_vals)
