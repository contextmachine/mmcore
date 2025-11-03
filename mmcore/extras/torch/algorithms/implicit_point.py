import functools
from typing import Callable

import torch
from torch import Tensor,FloatTensor
import numpy as np








def _resolve_grad_surf(surf, grad):
    """
    Resolves the gradient function for a given implicit surface function.
    If grad is None, a default finite-difference estimator is returned.

    :param surf: Implicit surface function.
                 Callable accepting a point (torch.Tensor of shape (N,)) and returning a scalar.
    :param grad: Gradient function for the surface.
                 Callable accepting a point (torch.Tensor of shape (N,)) and returning a torch.Tensor of shape (N,).
                 If None, a central finite difference estimator is used.
    :return: A callable that computes the gradient.
    """
    if grad is None:
        def grad_default(q, h=1e-5):
            # Create a tensor to store the estimated gradient.
            grad_est = torch.zeros_like(q, dtype=q.dtype, device=q.device)
            # Finite difference: perturb each dimension.
            for i in range(q.size(0)):
                q1 = q.clone()
                q2 = q.clone()
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

    :param J: torch.Tensor of shape (M, 2, 2) representing M Jacobian matrices.
    :param r: torch.Tensor of shape (M, 2) representing the right-hand side vectors.
    :param eps: Tolerance for the determinant to be considered nonzero.
    :return: A tuple (success, sol) where
             success is a boolean tensor of shape (M,) indicating whether each system was solvable,
             and sol is a tensor of shape (M, 2) containing the solutions (undefined where success is False).
    """
    M = J.shape[0]
    sol = torch.zeros((M, 2), dtype=J.dtype, device=J.device)
    # Compute the determinant for each system.
    det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    success = (det.abs() > eps)
    idx = torch.nonzero(success, as_tuple=False).squeeze()
    if idx.numel() > 0:
        # Ensure idx is always a 1D tensor.
        if idx.dim() == 0:
            idx = idx.unsqueeze(0)
        sol[idx, 0] = (J[idx, 1, 1] * r[idx, 0] - J[idx, 0, 1] * r[idx, 1]) / det[idx]
        sol[idx, 1] = (-J[idx, 1, 0] * r[idx, 0] + J[idx, 0, 0] * r[idx, 1]) / det[idx]
    return success, sol


def intersection_curve_point_batch(surf1, surf2, q0, grad1=None, grad2=None, tol=1e-6, max_iter=100,
                                   return_grads=False, no_err=True):
    """
    Computes the intersection point on the curve defined by the intersection of two implicit surfaces,
    using a Newton-based iterative method. This batched version accepts a set of initial points.

    :param surf1: The first surface implicit function.
                  Callable that accepts a point (torch.Tensor of shape (N,)) and returns a scalar.
    :param surf2: The second surface implicit function.
                  Callable that accepts a point (torch.Tensor of shape (N,)) and returns a scalar.
    :param q0: Initial point(s) on the intersection curve.
               torch.Tensor of shape (N,) for a single point or (B, N) for a batch of B points.
    :param grad1: Gradient function for surf1.
                  Callable accepting a point (torch.Tensor of shape (N,)) and returning a torch.Tensor of shape (N,).
                  If None, a finite-difference estimator is used.
    :param grad2: Gradient function for surf2.
                  Callable accepting a point (torch.Tensor of shape (N,)) and returning a torch.Tensor of shape (N,).
                  If None, a finite-difference estimator is used.
    :param tol: Tolerance for convergence (default 1e-6).
    :param max_iter: Maximum number of iterations allowed (default 100).
    :param return_grads: If True, the function returns additional information (f1, f2, g1, g2).
                         See return section below.
    :param no_err: If True (default), nonconvergence is handled leniently (the corresponding points are marked as failures).
                   If False, a ValueError is raised for points that do not converge within max_iter iterations.
    :return: Depending on return_grads:
             - If return_grads is True, returns a tuple of six tensors:
                 success  : (B,) boolean tensor indicating convergence success for each point.
                 qk       : (B, N) tensor of the final intersection points.
                 f1       : (B,) tensor with surf1 evaluated at the final points.
                 f2       : (B,) tensor with surf2 evaluated at the final points.
                 g1       : (B, N) tensor with grad1 evaluated at the final points.
                 g2       : (B, N) tensor with grad2 evaluated at the final points.
             - Otherwise, returns a tuple of two tensors:
                 success  : (B,) boolean tensor.
                 qk       : (B, N) tensor of the final intersection points.
    """
    # Resolve gradient functions for the surfaces.
    grad1 = _resolve_grad_surf(surf1, grad1)
    grad2 = _resolve_grad_surf(surf2, grad2)

    # Ensure q0 is two-dimensional: shape (B, N)
    initial_shape = q0.shape
    if q0.dim() == 1:
        q0 = q0.reshape(1, -1)
    else:
        q0 = q0.reshape(-1, q0.shape[-1])
    B, N = q0.shape

    # Initialize the iterate array and auxiliary storage.
    qk = q0.clone()  # Current iterates, shape (B, N)
    f1_arr = torch.empty(B, dtype=q0.dtype, device=q0.device)
    f2_arr = torch.empty(B, dtype=q0.dtype, device=q0.device)
    g1_arr = torch.empty((B, N), dtype=q0.dtype, device=q0.device)
    g2_arr = torch.empty((B, N), dtype=q0.dtype, device=q0.device)

    # active[i] is True if point i is still iterating.
    active = torch.ones(B, dtype=torch.bool, device=q0.device)
    # success[i] remains True only if point i converges.
    success = torch.ones(B, dtype=torch.bool, device=q0.device)

    iter_count = 0
    while active.any() and iter_count < max_iter:
        # Get indices of active points.
        active_idx = torch.nonzero(active, as_tuple=False).squeeze()
        if active_idx.dim() == 0:
            active_idx = active_idx.unsqueeze(0)

        # Evaluate surf1, surf2 and their gradients for active points.
        for i in active_idx.tolist():
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
        a = torch.sum(g1_active * g1_active, dim=1)      # (M,)
        b = torch.sum(g2_active * g1_active, dim=1)         # (M,)
        c = torch.sum(g1_active * g2_active, dim=1)         # (M,)
        d_comp = torch.sum(g2_active * g2_active, dim=1)    # (M,)

        # Assemble the Jacobian for each active point.
        J_active = torch.stack([torch.stack([a, b], dim=1),
                                torch.stack([c, d_comp], dim=1)], dim=1)  # (M, 2, 2)

        # Right-hand side: r = -[f1, f2]
        r_active = torch.stack([-f1_active, -f2_active], dim=1)  # (M, 2)

        # Solve each 2x2 system.
        sol_success, sol = solve2x2_batch(J_active, r_active)
        # Update overall success for these points.
        success[active_idx] = success[active_idx] & sol_success

        # Compute the Newton update delta = alpha * g1 + beta * g2.
        delta_active = torch.zeros((active_idx.size(0), N), dtype=q0.dtype, device=q0.device)
        for j, idx in enumerate(active_idx.tolist()):
            if sol_success[j]:
                delta_active[j] = sol[j, 0] * g1_active[j] + sol[j, 1] * g2_active[j]
            else:
                delta_active[j] = torch.zeros(N, dtype=q0.dtype, device=q0.device)

        # Compute the norm of the update for each active point.
        norms = torch.norm(delta_active, dim=1)

        # Update the iterates and check convergence.
        for j, idx in enumerate(active_idx.tolist()):
            if sol_success[j]:
                qk[idx] = qk[idx] + delta_active[j]
                if norms[j] <= tol:
                    active[idx] = False  # Converged.
            else:
                # If the 2x2 system could not be solved, mark this point as done.
                active[idx] = False

        iter_count += 1

    # After max_iter iterations, if some points are still active then handle according to no_err.
    if active.any():
        if not no_err:
            raise ValueError("Maximum iterations exceeded for some points.")
        else:
            failed_idx = torch.nonzero(active, as_tuple=False).squeeze()
            if failed_idx.dim() == 0:
                failed_idx = failed_idx.unsqueeze(0)
            success[failed_idx] = False
            active[failed_idx] = False

    # Reshape outputs to match the original q0 shape.
    if return_grads:
        return (success.view(initial_shape[:-1]),
                qk.view(initial_shape),
                f1_arr.view(initial_shape[:-1]),
                f2_arr.view(initial_shape[:-1]),
                g1_arr.view(initial_shape),
                g2_arr.view(initial_shape))
    else:
        return success.view(initial_shape[:-1]), qk.view(initial_shape)


# =========================
# Example usage:
# =========================
if __name__ == "__main__":
    from mmcore.numeric.algorithms import implicit_point as _implicit_point
    # Define two implicit surfaces. For example, two spheres.
    def sphere1(point):
        # Sphere centered at (0, 0, 0) with radius 1: x^2 + y^2 + z^2 - 1 = 0
        return point[0] ** 2 + point[1] ** 2 + point[2] ** 2 - 1


    def sphere2(point):
        # Sphere centered at (0.5, 0.5, 0.5) with radius 1:
        # (x-0.5)^2 + (y-0.5)^2 + (z-0.5)^2 - 1 = 0
        return (point[0] - 0.5) ** 2 + (point[1] - 0.5) ** 2 + (point[2] - 0.5) ** 2 - 1


    # Create a batch of initial points in 3D (shape (B, 3)).
    q0_batch = np.array(
        [
            [0.8, 0.1, 0.9],
            [0.2, 0.3, 1.0],
            [1.0, 0.5, 0.5],
            [0.0, 0.0, 0.0],  # This point may be far from the intersection.
        ]
    )

    # Compute the intersection points, returning extra gradient information.
    success, qk, f1_vals, f2_vals, g1_vals, g2_vals = _implicit_point.intersection_curve_point_batch(
        sphere1,
        sphere2,
        q0_batch,
        tol=1e-6,
        max_iter=100,
        return_grads=True,
        no_err=True,
    )
    # Create a batch of initial points in 3D (shape (B, 3)).
    q0_batch = torch.tensor(
        [
            [0.8, 0.1, 0.9],
            [0.2, 0.3, 1.0],
            [1.0, 0.5, 0.5],
            [0.0, 0.0, 0.0],  # This point may be far from the intersection.
        ], dtype=torch.float64
    )
    print('\n\nnumpy version\n---------------------------------------------------------------------------------------------\n')


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
    print('\n\n\ntorch version\n---------------------------------------------------------------------------------------------\n')
    successt, qkt, f1_valst, f2_valst, g1_valst, g2_valst = intersection_curve_point_batch(
        sphere1,
        sphere2,
        q0_batch,
        tol=1e-6,
        max_iter=100,
        return_grads=True,
        no_err=True,
    )

    print("Convergence success flags:")
    print(successt)
    print("\nIntersection points (qk):")
    print(qkt)
    print("\nsphere1 evaluations (f1):")
    print(f1_valst)
    print("\nsphere2 evaluations (f2):")
    print(f2_valst)
    print("\nGradient of sphere1 (g1):")
    print(g1_valst)
    print("\nGradient of sphere2 (g2):")
    print(g2_valst)
    print(
        '\n\n\ncomparison\n---------------------------------------------------------------------------------------------\n')

    print(">>> torch.allclose(torch.tensor(qk),qkt) # final check\n",torch.allclose(torch.tensor(qk),qkt))


