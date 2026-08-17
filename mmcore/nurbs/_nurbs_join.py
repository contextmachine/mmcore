from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from math import comb
from typing import Iterable, NamedTuple, Optional, Sequence

import numpy as np


# =============================================================================
# Optional fallback tuple definitions
# =============================================================================
# If your project already defines NURBSCurveTuple / KnotRemovalResult, these
# fallback definitions will not override them.

from ._nurbs_eval import NURBSCurveTuple

from ._nurbs_knots import KnotRemovalResult



# =============================================================================
# Join operation specifications
# =============================================================================

@dataclass(frozen=True)
class JoinSegmentSpec:
    source_index: int
    is_reversed: bool
    degree_elevations: int

    # Final parameter length assigned to this segment during linking.
    # This is what makes the reparameterisation replayable on another curve set.
    parameter_length: float


@dataclass(frozen=True)
class JoinInterfaceSpec:
    """
    Describes one interior join between segment local index i and i + 1.
    """
    left_local_index: int
    right_local_index: int

    # Knot value of this join in the linked curve before/during knot removal.
    knot: float

    # True when tangent directions matched at the join.
    is_c1_directional: bool

    # Multiplicative parameter scale applied to the right segment.
    # 1.0 means no C1 reparameterisation was applied at this interface.
    right_parameter_scale: float

    # Knot-removal result for this interface.
    knot_removed: bool = False
    knot_removal_error: float = float("inf")


@dataclass(frozen=True)
class JoinChainSpec:
    segments: tuple[JoinSegmentSpec, ...]
    target_order: int

    is_cycle: bool = False
    is_singleton: bool = False
    was_closed_input: bool = False

    interfaces: tuple[JoinInterfaceSpec, ...] = ()
    removed_knots: tuple[float, ...] = ()

    # For cycles, the last->first connection is geometric but not an interior
    # knot in the clamped linked curve.
    cycle_closure_is_c1_directional: bool = False

    @property
    def source_indices(self) -> tuple[int, ...]:
        return tuple(s.source_index for s in self.segments)

    @property
    def reversals(self) -> tuple[bool, ...]:
        return tuple(s.is_reversed for s in self.segments)

    @property
    def degree_elevations(self) -> tuple[int, ...]:
        return tuple(s.degree_elevations for s in self.segments)

    @property
    def parameter_lengths(self) -> tuple[float, ...]:
        return tuple(s.parameter_length for s in self.segments)


# =============================================================================
# Basic Bezier / NURBS helpers
# =============================================================================

def degree_elevation(degree, ctrlpts, num=1, **kwargs):
    """
    Compute control points after degree elevation for a Bezier shape.

    This is Eq. 5.36 of The NURBS Book by Piegl & Tiller, 2nd ed., p.205.

    Parameters
    ----------
    degree:
        Current Bezier degree.
    ctrlpts:
        Control points. Can also be homogeneous control points.
    num:
        Number of degree elevations.

    Returns
    -------
    list
        Degree-elevated control points.
    """
    if num < 0:
        raise ValueError("num must be >= 0")

    if num == 0:
        return [list(p) for p in ctrlpts]

    num_pts_elev = degree + 1 + num
    pts_elev = [[0.0 for _ in range(len(ctrlpts[0]))] for _ in range(num_pts_elev)]

    for i in range(num_pts_elev):
        start = max(0, i - num)
        end = min(degree, i)

        for j in range(start, end + 1):
            coeff = comb(degree, j) * comb(num, i - j)
            coeff /= comb(degree + num, i)
            pts_elev[i] = [
                p1 + coeff * p2
                for p1, p2 in zip(pts_elev[i], ctrlpts[j])
            ]

    return pts_elev


def _bezier_knots(order: int, interval: tuple[float, float]):
    start, end = interval
    return [start] * order + [end] * order


def to_homogeneous(control_points, weights):
    """
    Convert Euclidean rational control points P_i, w_i to homogeneous P_i*w_i, w_i.
    """
    P = np.asarray(control_points, dtype=float)
    W = np.asarray(weights, dtype=float)

    if P.ndim != 2:
        raise ValueError("control_points must be a 2D array")

    if W.ndim != 1:
        raise ValueError("weights must be a 1D array")

    if len(P) != len(W):
        raise ValueError("len(control_points) must equal len(weights)")

    return np.hstack([P * W[:, None], W[:, None]])


def to_euclidean(P_hom):
    """
    Project homogeneous rational control points back to Euclidean points and weights.
    """
    P_hom = np.asarray(P_hom, dtype=float)

    if P_hom.ndim != 2 or P_hom.shape[1] < 2:
        raise ValueError("P_hom must be a 2D array with at least 2 columns")

    W = P_hom[:, -1].copy()

    if np.any(np.abs(W) < 1e-15):
        raise ValueError("Cannot project homogeneous control point with near-zero weight")

    P = P_hom[:, :-1] / W[:, None]
    return P, W


def _copy_curve_as_tuple(crv):
    """
    Defensive copy as NURBSCurveTuple with numpy arrays.
    """
    return NURBSCurveTuple(
        order=int(crv.order),
        knot=np.asarray(crv.knot, dtype=float).copy(),
        control_points=np.asarray(crv.control_points, dtype=float).copy(),
        weights=np.asarray(crv.weights, dtype=float).copy(),
    )


def _curve_domain(crv):
    """
    Active clamped curve domain.

    This follows the convention used by the linker:
      start = first knot
      end   = first trailing clamp knot = knot[-order]
    """
    k = np.asarray(crv.knot, dtype=float)
    order = int(crv.order)

    if len(k) < order * 2:
        raise ValueError("Curve has an unexpectedly short knot vector")

    return float(k[0]), float(k[-order])


def _curve_domain_length(crv) -> float:
    a, b = _curve_domain(crv)
    length = float(b - a)

    if length <= 0.0:
        raise ValueError(f"Curve has non-positive parameter length: {length}")

    return length


def _reverse_curve(crv):
    """
    Reverse a clamped NURBS curve orientation: u -> u0 + u1 - u.

    This preserves geometry and flips:
      - control point order
      - weight order
      - knot vector mirrored into the same [u0, u1] domain
    """
    k = np.asarray(crv.knot, dtype=float)
    cp = np.asarray(crv.control_points, dtype=float)
    w = np.asarray(crv.weights, dtype=float)

    u0, u1 = float(k[0]), float(k[-1])
    k_rev = (u0 + u1) - k[::-1]

    return NURBSCurveTuple(
        order=int(crv.order),
        knot=k_rev,
        control_points=cp[::-1].copy(),
        weights=w[::-1].copy(),
    )


def _is_c0_closed(crv, tol: float) -> bool:
    """
    Return True if curve start/end control points coincide within tol.
    """
    cp = np.asarray(crv.control_points, dtype=float)
    d = cp[0] - cp[-1]
    return float(np.dot(d, d)) <= float(tol) * float(tol)


def _snap_pair(a, a_idx: int, b, b_idx: int):
    """
    Snap two endpoints exactly together using average point and average weight.
    """
    p = 0.5 * (a.control_points[a_idx] + b.control_points[b_idx])
    w = 0.5 * (float(a.weights[a_idx]) + float(b.weights[b_idx]))

    a.control_points[a_idx] = p
    b.control_points[b_idx] = p
    a.weights[a_idx] = w
    b.weights[b_idx] = w


def _snap_join_endpoints(pieces, is_cycle: bool):
    """
    Snap consecutive piece endpoints, and also last->first for cycles.
    """
    for i in range(len(pieces) - 1):
        _snap_pair(pieces[i], -1, pieces[i + 1], 0)

    if is_cycle and len(pieces) >= 2:
        _snap_pair(pieces[-1], -1, pieces[0], 0)


def _degree_elevate_bezier_curve(crv, num: int):
    """
    Fallback rational Bezier degree elevation.

    If your project already provides degree_elevate_curve(), that function is used
    instead by _degree_elevate_curve_impl().

    This fallback assumes the input is a single Bezier segment:
      len(control_points) == order
      knot == [u0] * order + [u1] * order
    """
    num = int(num)

    if num < 0:
        raise ValueError("degree elevation count must be >= 0")

    crv = _copy_curve_as_tuple(crv)

    if num == 0:
        return crv

    order = int(crv.order)
    degree = order - 1

    if len(crv.control_points) != order:
        raise ValueError(
            "Fallback degree elevation only supports single Bezier segments. "
            "Provide your project's general degree_elevate_curve() for non-Bezier curves."
        )

    P_hom = to_homogeneous(crv.control_points, crv.weights)
    P_hom_elev = np.asarray(
        degree_elevation(degree, P_hom.tolist(), num=num),
        dtype=float,
    )

    P_new, W_new = to_euclidean(P_hom_elev)

    u0, u1 = _curve_domain(crv)
    new_order = order + num
    U_new = np.asarray(_bezier_knots(new_order, (u0, u1)), dtype=float)

    return NURBSCurveTuple(
        order=new_order,
        knot=U_new,
        control_points=P_new,
        weights=W_new,
    )


def _degree_elevate_curve_impl(crv, num: int):
    """
    Use project-level degree_elevate_curve() if it exists; otherwise use the
    rational Bezier fallback above.
    """
    num = int(num)

    if num < 0:
        raise ValueError("degree elevation count must be >= 0")

    if num == 0:
        return _copy_curve_as_tuple(crv)

    fn = globals().get("degree_elevate_curve", None)

    if callable(fn):
        return _copy_curve_as_tuple(fn(crv, num))

    return _degree_elevate_bezier_curve(crv, num)


# =============================================================================
# Basis/evaluation helpers used by derivative checks and knot removal
# =============================================================================

def find_span(n: int, p: int, u: float, U):
    """
    Standard B-spline knot span search.

    n = number of control points - 1
    p = degree
    """
    U = np.asarray(U, dtype=float)
    u = float(u)

    if u >= U[n + 1]:
        return n

    if u <= U[p]:
        return p

    low = p
    high = n + 1
    mid = (low + high) // 2

    while u < U[mid] or u >= U[mid + 1]:
        if u < U[mid]:
            high = mid
        else:
            low = mid

        mid = (low + high) // 2

    return mid


def ders_basis_funs(i: int, u: float, p: int, U, n_ders: int = 1):
    """
    Derivatives of nonzero basis functions N_{i-p}, ..., N_i.

    Returns array of shape (n_ders + 1, p + 1).
    """
    U = np.asarray(U, dtype=float)
    u = float(u)

    requested_ders = int(n_ders)
    n_ders = min(requested_ders, p)

    ndu = np.zeros((p + 1, p + 1), dtype=float)
    left = np.zeros(p + 1, dtype=float)
    right = np.zeros(p + 1, dtype=float)

    ndu[0, 0] = 1.0

    for j in range(1, p + 1):
        left[j] = u - U[i + 1 - j]
        right[j] = U[i + j] - u
        saved = 0.0

        for r in range(j):
            ndu[j, r] = right[r + 1] + left[j - r]

            if abs(ndu[j, r]) < 1e-15:
                temp = 0.0
            else:
                temp = ndu[r, j - 1] / ndu[j, r]

            ndu[r, j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp

        ndu[j, j] = saved

    ders = np.zeros((requested_ders + 1, p + 1), dtype=float)

    for j in range(p + 1):
        ders[0, j] = ndu[j, p]

    a = np.zeros((2, p + 1), dtype=float)

    for r in range(p + 1):
        s1 = 0
        s2 = 1
        a[0, 0] = 1.0

        for k in range(1, n_ders + 1):
            d = 0.0
            rk = r - k
            pk = p - k

            if r >= k:
                denom = ndu[pk + 1, rk]
                a[s2, 0] = a[s1, 0] / denom if abs(denom) > 1e-15 else 0.0
                d = a[s2, 0] * ndu[rk, pk]

            if rk >= -1:
                j1 = 1
            else:
                j1 = -rk

            if r - 1 <= pk:
                j2 = k - 1
            else:
                j2 = p - r

            for j in range(j1, j2 + 1):
                denom = ndu[pk + 1, rk + j]
                a[s2, j] = (
                    (a[s1, j] - a[s1, j - 1]) / denom
                    if abs(denom) > 1e-15
                    else 0.0
                )
                d += a[s2, j] * ndu[rk + j, pk]

            if r <= pk:
                denom = ndu[pk + 1, r]
                a[s2, k] = -a[s1, k - 1] / denom if abs(denom) > 1e-15 else 0.0
                d += a[s2, k] * ndu[r, pk]

            ders[k, r] = d
            s1, s2 = s2, s1

    factor = p

    for k in range(1, n_ders + 1):
        for j in range(p + 1):
            ders[k, j] *= factor

        factor *= p - k

    return ders


def evaluate_b_spline_derivs(U, P, p: int, u: float, n_ders: int = 1):
    """
    Evaluate derivatives of a non-rational B-spline in the control-point space P.

    For rational curves, pass homogeneous control points.
    """
    U = np.asarray(U, dtype=float)
    P = np.asarray(P, dtype=float)

    n = len(P) - 1
    n_ders = int(n_ders)

    span = find_span(n, p, u, U)
    ders = ders_basis_funs(span, u, p, U, n_ders=n_ders)

    CK = np.zeros((n_ders + 1, P.shape[1]), dtype=float)

    for k in range(min(n_ders, p) + 1):
        for j in range(p + 1):
            cp_idx = span - p + j
            CK[k] += ders[k, j] * P[cp_idx]

    return CK


def _nurbs_first_derivative(crv, u: float):
    """
    Euclidean first derivative of a rational NURBS curve at parameter u.
    """
    p = int(crv.order) - 1
    U = np.asarray(crv.knot, dtype=float)
    P = np.asarray(crv.control_points, dtype=float)
    W = np.asarray(crv.weights, dtype=float)

    P_hom = to_homogeneous(P, W)
    ders_hom = evaluate_b_spline_derivs(U, P_hom, p, float(u), n_ders=1)

    Cw = ders_hom[0]
    Dw = ders_hom[1]

    w = float(Cw[-1])
    dw = float(Dw[-1])

    if abs(w) < 1e-15:
        raise ValueError("Cannot evaluate rational derivative: zero homogeneous weight")

    return (Dw[:-1] * w - Cw[:-1] * dw) / (w * w)


def _endpoint_first_derivative(crv, side: int):
    """
    side=0 -> start derivative
    side=1 -> end derivative
    """
    u0, u1 = _curve_domain(crv)
    return _nurbs_first_derivative(crv, u0 if side == 0 else u1)


def _same_derivative_direction(
    v1,
    v2,
    *,
    direction_tol: float = 1e-6,
    derivative_tol: float = 1e-12,
):
    """
    Return True when v1 and v2 point in the same direction.

    Uses normalized vector distance:
        ||v1 / ||v1|| - v2 / ||v2||||

    This works for 2D, 3D, and higher dimensions.
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)

    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))

    if n1 <= derivative_tol or n2 <= derivative_tol:
        return False

    u1 = v1 / n1
    u2 = v2 / n2

    # Same direction, not opposite direction.
    if float(np.dot(u1, u2)) <= 0.0:
        return False

    return float(np.linalg.norm(u1 - u2)) <= float(direction_tol)


def _compute_c1_reparameterisation(
    pieces,
    *,
    is_cycle: bool,
    enabled: bool = True,
    direction_tol: float = 1e-6,
    derivative_tol: float = 1e-12,
):
    """
    Compute final parameter lengths for an already oriented and snapped chain.

    First segment keeps its native parameter length. Each next segment may be
    scaled if the join has matching tangent directions.

    For affine parameter scaling:
        C_new(u_new) = C_old(u_old)
        u_new = offset + scale * (u_old - u0)

    Therefore:
        dC_new / du_new = dC_old / du_old / scale

    To make derivatives equal in magnitude:
        scale_right = ||D_right_start_native|| / ||D_left_end_global||

    Returns
    -------
    segment_lengths:
        Final parameter length assigned to each segment.
    interfaces:
        JoinInterfaceSpec objects without knot values filled in yet.
    cycle_closure_is_c1_directional:
        Whether the last->first closure has matching tangent directions.
    """
    n = len(pieces)

    native_lengths = [_curve_domain_length(c) for c in pieces]
    segment_lengths = list(native_lengths)

    # scale[i] = segment_lengths[i] / native_lengths[i]
    scales = [1.0 for _ in pieces]

    interfaces = []

    for i in range(n - 1):
        left = pieces[i]
        right = pieces[i + 1]

        d_left_native = _endpoint_first_derivative(left, side=1)
        d_left_global = d_left_native / scales[i]

        d_right_native = _endpoint_first_derivative(right, side=0)

        is_c1_dir = _same_derivative_direction(
            d_left_global,
            d_right_native,
            direction_tol=direction_tol,
            derivative_tol=derivative_tol,
        )

        right_scale = 1.0

        if enabled and is_c1_dir:
            n_left = float(np.linalg.norm(d_left_global))
            n_right = float(np.linalg.norm(d_right_native))

            if n_left > derivative_tol and n_right > derivative_tol:
                right_scale = n_right / n_left
                scales[i + 1] = right_scale
                segment_lengths[i + 1] = native_lengths[i + 1] * right_scale

        interfaces.append(
            JoinInterfaceSpec(
                left_local_index=i,
                right_local_index=i + 1,
                knot=float("nan"),
                is_c1_directional=is_c1_dir,
                right_parameter_scale=float(right_scale),
            )
        )

    cycle_closure_is_c1_directional = False

    if is_cycle and n >= 2:
        d_last_native = _endpoint_first_derivative(pieces[-1], side=1)
        d_first_native = _endpoint_first_derivative(pieces[0], side=0)

        d_last_global = d_last_native / scales[-1]
        d_first_global = d_first_native / scales[0]

        cycle_closure_is_c1_directional = _same_derivative_direction(
            d_last_global,
            d_first_global,
            direction_tol=direction_tol,
            derivative_tol=derivative_tol,
        )

    return (
        tuple(float(x) for x in segment_lengths),
        tuple(interfaces),
        bool(cycle_closure_is_c1_directional),
    )


# =============================================================================
# Knot removal
# =============================================================================

def remove_knot(
    curve: NURBSCurveTuple,
    u_remove: float,
    tolerance: float = 1e-4,
) -> KnotRemovalResult:
    """
    Removes one occurrence of a knot from a rational NURBS curve using
    homogeneous Hermite-Birkhoff interpolation.

    Steps
    -----
    1. Convert control points to homogeneous space.
    2. Identify the local knot span and affected control points.
    3. Sample the original curve's homogeneous position / derivative at boundaries.
    4. Solve least squares for new homogeneous control points.
    5. Project back to Euclidean space.
    6. Measure Euclidean point error at the removed knot.
    """
    p = int(curve.order) - 1
    U_old = np.asarray(curve.knot, dtype=float)

    P_hom_old = to_homogeneous(curve.control_points, curve.weights)
    hom_dim = P_hom_old.shape[1]

    matches = np.where(np.abs(U_old - float(u_remove)) < 1e-12)[0]

    if len(matches) == 0:
        return KnotRemovalResult(None, False, float("inf"), float(u_remove))

    r = int(matches[-1])

    # Safety check for boundary knots.
    if r >= len(U_old) - p - 1:
        if len(matches) > 1:
            r = int(matches[-2])
        else:
            return KnotRemovalResult(None, False, float("inf"), float(u_remove))

    U_new = np.delete(U_old, r)

    # Recalculate Q_{r-p} ... Q_{r-1}
    start_idx = r - p
    end_idx = r - 1
    num_unknowns = p

    if start_idx < 0:
        return KnotRemovalResult(None, False, float("inf"), float(u_remove))

    u_a = U_new[r - 1]
    u_b = U_new[r]

    samples = [
        (u_a, 0),  # position
        (u_a, 1),  # derivative
        (u_b, 0),  # position
        (u_b, 1),  # derivative
    ]

    # Additional internal position constraints for degree > 3.
    if p > 3:
        num_internal = max(0, p - 4)

        if num_internal > 0:
            t_internal = np.linspace(u_a, u_b, num_internal + 2)[1:-1]

            for t in t_internal:
                samples.append((float(t), 0))

    num_constraints = len(samples)

    A = np.zeros((num_constraints, num_unknowns), dtype=float)
    B = np.zeros((num_constraints, hom_dim), dtype=float)

    n_new = len(P_hom_old) - 2

    for row_i, (u, mode) in enumerate(samples):
        span_new = find_span(n_new, p, u, U_new)
        ders_new = ders_basis_funs(span_new, u, p, U_new, n_ders=1)
        basis_vals = ders_new[mode, :]

        target_val_hom = evaluate_b_spline_derivs(
            U_old,
            P_hom_old,
            p,
            u,
            n_ders=1,
        )[mode]

        rhs_contrib = target_val_hom.copy()

        for j in range(p + 1):
            cp_idx = span_new - p + j
            basis = basis_vals[j]

            if start_idx <= cp_idx <= end_idx:
                col_idx = cp_idx - start_idx

                if 0 <= col_idx < num_unknowns:
                    A[row_i, col_idx] += basis
            else:
                if cp_idx < start_idx:
                    fixed_cp = P_hom_old[cp_idx]
                else:
                    fixed_cp = P_hom_old[cp_idx + 1]

                rhs_contrib -= basis * fixed_cp

        B[row_i] = rhs_contrib

    try:
        Q_hom_local, _residuals, _rank, _s = np.linalg.lstsq(A, B, rcond=None)
    except np.linalg.LinAlgError:
        return KnotRemovalResult(None, False, float("inf"), float(u_remove))

    P_hom_new = np.zeros((len(P_hom_old) - 1, hom_dim), dtype=float)
    P_hom_new[:start_idx] = P_hom_old[:start_idx]
    P_hom_new[start_idx : end_idx + 1] = Q_hom_local
    P_hom_new[end_idx + 1 :] = P_hom_old[end_idx + 2 :]

    try:
        P_new, W_new = to_euclidean(P_hom_new)
    except ValueError:
        return KnotRemovalResult(None, False, float("inf"), float(u_remove))

    new_curve = NURBSCurveTuple(
        p + 1,
        np.asarray(U_new, dtype=float),
        np.asarray(P_new, dtype=float),
        np.asarray(W_new, dtype=float),
    )

    pt_old_hom = evaluate_b_spline_derivs(U_old, P_hom_old, p, u_remove, 0)[0]
    pt_old_euc = pt_old_hom[:-1] / pt_old_hom[-1]

    pt_new_hom = evaluate_b_spline_derivs(U_new, P_hom_new, p, u_remove, 0)[0]
    pt_new_euc = pt_new_hom[:-1] / pt_new_hom[-1]

    dist = float(np.linalg.norm(pt_old_euc - pt_new_euc))
    success = dist <= float(tolerance)

    return KnotRemovalResult(new_curve, bool(success), dist, float(u_remove))


def _kr_curve(result):
    if hasattr(result, "curve"):
        return result.curve

    if hasattr(result, "new_curve"):
        return result.new_curve

    return result[0]


def _kr_success(result) -> bool:
    if hasattr(result, "success"):
        return bool(result.success)

    return bool(result[1])


def _kr_error(result) -> float:
    if hasattr(result, "error"):
        return float(result.error)

    if hasattr(result, "distance"):
        return float(result.distance)

    return float(result[2])


def _get_default_knot_remover(knot_remover):
    if knot_remover is not None:
        return knot_remover

    return remove_knot


def _try_remove_c1_join_knots(
    joined_curve,
    interfaces,
    *,
    remove_c1_knots: bool,
    knot_removal_tolerance: float,
    knot_remover=None,
):
    """
    Attempt one knot removal at each C1-directional interior join.

    Only accepted removals are recorded.
    """
    curve = joined_curve
    updated = []

    if not remove_c1_knots:
        return curve, tuple(interfaces), ()

    knot_remover = _get_default_knot_remover(knot_remover)

    for iface in interfaces:
        if not iface.is_c1_directional:
            updated.append(iface)
            continue

        res = knot_remover(
            curve,
            float(iface.knot),
            tolerance=float(knot_removal_tolerance),
        )

        err = _kr_error(res)

        if _kr_success(res) and _kr_curve(res) is not None:
            curve = _kr_curve(res)
            updated.append(
                replace(
                    iface,
                    knot_removed=True,
                    knot_removal_error=float(err),
                )
            )
        else:
            updated.append(
                replace(
                    iface,
                    knot_removed=False,
                    knot_removal_error=float(err),
                )
            )

    removed_knots = tuple(
        float(iface.knot)
        for iface in updated
        if iface.knot_removed
    )

    return curve, tuple(updated), removed_knots


# =============================================================================
# Low-level curve linking
# =============================================================================

def _elevate_pieces_by_counts(
    pieces,
    degree_counts,
    target_order: Optional[int] = None,
):
    """
    Elevate each piece by an explicit count.

    This is intentionally explicit because the returned JoinChainSpec must be
    replayable on another corresponding curve set.
    """
    if len(pieces) != len(degree_counts):
        raise ValueError("pieces and degree_counts must have the same length")

    out = []

    for i, (crv, cnt) in enumerate(zip(pieces, degree_counts)):
        cnt = int(cnt)

        if cnt < 0:
            raise ValueError(f"Negative degree elevation count for piece {i}: {cnt}")

        crv2 = _copy_curve_as_tuple(crv)

        if cnt:
            crv2 = _copy_curve_as_tuple(_degree_elevate_curve_impl(crv2, cnt))

        if target_order is not None and int(crv2.order) != int(target_order):
            raise ValueError(
                f"Piece {i} has order {crv2.order} after {cnt} degree elevations; "
                f"expected target order {target_order}."
            )

        out.append(crv2)

    if target_order is None and out:
        order = int(out[0].order)

        for i, crv in enumerate(out):
            if int(crv.order) != order:
                raise ValueError(
                    "Specified degree elevations did not produce a common order: "
                    f"piece 0 has order {order}; piece {i} has order {crv.order}."
                )

    return out


def _link_curves_same_order(
    curves,
    segment_lengths: Optional[Sequence[float]] = None,
):
    """
    Concatenate already degree-compatible, already oriented pieces.

    If segment_lengths is supplied, each curve's local knot vector is affinely
    scaled so its active domain has exactly that length in the joined curve.
    """
    curves = [_copy_curve_as_tuple(c) for c in curves]

    if not curves:
        raise ValueError("Empty input list")

    order = int(curves[0].order)

    for i, c in enumerate(curves):
        if int(c.order) != order:
            raise ValueError(
                f"All curves must already have the same order. "
                f"Curve 0 has order {order}; curve {i} has order {c.order}."
            )

    if segment_lengths is None:
        segment_lengths = [_curve_domain_length(c) for c in curves]
    else:
        if len(segment_lengths) != len(curves):
            raise ValueError("segment_lengths must have the same length as curves")

        segment_lengths = [float(x) for x in segment_lengths]

        for i, length in enumerate(segment_lengths):
            if length <= 0:
                raise ValueError(f"segment_lengths[{i}] must be > 0; got {length}")

    kv = []
    cpts = []
    wgts = []

    interior_knots = []
    offset = 0.0

    for i, crv in enumerate(curves):
        k = np.asarray(crv.knot, dtype=float).copy()
        cp = np.asarray(crv.control_points, dtype=float)
        w = np.asarray(crv.weights, dtype=float)

        u0, u1 = _curve_domain(crv)
        native_length = float(u1 - u0)

        if native_length <= 0:
            raise ValueError(f"Curve {i} has non-positive native parameter length")

        target_length = float(segment_lengths[i])
        scale = target_length / native_length

        # Affine parameter mapping:
        # old u0 -> offset
        # old u1 -> offset + target_length
        k = offset + (k - u0) * scale

        if i == 0:
            kv.extend(k[:-order])
            cpts.extend(cp)
            wgts.extend(w)
        else:
            # Skip duplicate first knot and duplicate first control point.
            kv.extend(k[1:-order])
            cpts.extend(cp[1:])
            wgts.extend(w[1:])

        offset = float(k[-order])
        interior_knots.append(offset)

    kv.extend([offset] * order)

    if interior_knots:
        interior_knots.pop()

    return (
        NURBSCurveTuple(
            order=order,
            knot=np.asarray(kv, dtype=float),
            control_points=np.asarray(cpts, dtype=float),
            weights=np.asarray(wgts, dtype=float),
        ),
        interior_knots,
    )


def _link_curves_with_degree_counts(
    pieces,
    degree_counts,
    target_order: Optional[int] = None,
    segment_lengths: Optional[Sequence[float]] = None,
):
    elevated = _elevate_pieces_by_counts(
        pieces,
        degree_counts,
        target_order=target_order,
    )

    return _link_curves_same_order(
        elevated,
        segment_lengths=segment_lengths,
    )


def link_curves(
    curves,
    segment_lengths: Optional[Sequence[float]] = None,
):
    """
    Backwards-compatible public linker.

    It computes the common max order, degree-elevates each piece as needed, and
    optionally applies affine parameter lengths before concatenation.
    """
    curves = [_copy_curve_as_tuple(c) for c in curves]

    if not curves:
        raise ValueError("Empty input list")

    target_order = max(int(c.order) for c in curves)
    degree_counts = [target_order - int(c.order) for c in curves]

    return _link_curves_with_degree_counts(
        curves,
        degree_counts,
        target_order=target_order,
        segment_lengths=segment_lengths,
    )


# =============================================================================
# Endpoint matching / topology
# =============================================================================

def _nearest_endpoint_nodes(base, curve_indices, tol: float):
    """
    Build graph nodes by greedy nearest-neighbor endpoint matching.

    Each endpoint may be matched to at most one other endpoint.

    If several endpoints lie within tol, candidate endpoint pairs are sorted by
    geometric distance and the closest still-unmatched pair is accepted. Ties are
    broken deterministically by endpoint index.
    """
    endpoint_curve = []
    endpoint_side = []
    endpoint_point = []
    endpoint_index = {}

    for ci in sorted(curve_indices):
        c = base[ci]

        for side, cp_idx in ((0, 0), (1, -1)):
            endpoint_index[(ci, side)] = len(endpoint_curve)
            endpoint_curve.append(ci)
            endpoint_side.append(side)
            endpoint_point.append(np.asarray(c.control_points[cp_idx], dtype=float))

    pts = np.asarray(endpoint_point, dtype=float)
    n = len(pts)

    tol2 = float(tol) * float(tol)
    candidates = []

    for a in range(n):
        for b in range(a + 1, n):
            # Never match one curve endpoint to the other endpoint of the same
            # curve. Closed curves are filtered before this function, but this
            # also protects nearly closed open curves.
            if endpoint_curve[a] == endpoint_curve[b]:
                continue

            d = pts[a] - pts[b]
            d2 = float(np.dot(d, d))

            if d2 <= tol2:
                candidates.append((d2, a, b))

    candidates.sort(key=lambda x: (x[0], x[1], x[2]))

    matched = {}
    selected_pairs = []

    for d2, a, b in candidates:
        if a not in matched and b not in matched:
            matched[a] = b
            matched[b] = a
            selected_pairs.append((min(a, b), max(a, b), d2))

    selected_pairs.sort(key=lambda x: (x[0], x[1]))

    endpoint_node = np.full(n, -1, dtype=int)
    next_node = 0

    for a, b, _d2 in selected_pairs:
        endpoint_node[a] = next_node
        endpoint_node[b] = next_node
        next_node += 1

    for e in range(n):
        if endpoint_node[e] < 0:
            endpoint_node[e] = next_node
            next_node += 1

    start_node = {}
    end_node = {}
    node_incident = defaultdict(list)

    for ci in sorted(curve_indices):
        a = int(endpoint_node[endpoint_index[(ci, 0)]])
        b = int(endpoint_node[endpoint_index[(ci, 1)]])

        start_node[ci] = a
        end_node[ci] = b

        node_incident[a].append((ci, 0))
        node_incident[b].append((ci, 1))

    return start_node, end_node, node_incident, selected_pairs


# =============================================================================
# Public join functions
# =============================================================================

def join_curves(
    curves: Iterable[NURBSCurveTuple],
    tol=1e-6,
    *,
    closed_segment_tol: Optional[float] = None,
    reparameterize_c1: bool = True,
    c1_direction_tol: float = 1e-6,
    derivative_tol: float = 1e-12,
    remove_c1_knots: bool = True,
    knot_removal_tolerance: Optional[float] = None,
    knot_remover=None,
):
    """
    Join an arbitrary set of NURBS curves into maximal C0-connected chains.

    Returns
    -------
    joined_curves:
        list[NURBSCurveTuple]

    join_specs:
        list[JoinChainSpec]

    The spec records:
      - source segment indices in join order
      - whether each source segment was reversed
      - how many degree elevations were applied
      - final parameter length of each segment after C1 reparameterisation
      - which interior knots were successfully removed

    Behavior
    --------
    1. C0-closed input segments are filtered out of the join graph and returned
       as singleton outputs/specs.
    2. Endpoint clusters with more than two possible endpoints are handled by
       greedy closest-pair matching instead of raising immediately.
    3. If two consecutive pieces have matching tangent directions, the right
       piece's parameter length is scaled so first derivative magnitudes match.
    4. After linking, one knot-removal attempt is made at each C1-directional
       interior join. If removal satisfies tolerance, it is accepted and recorded.

    Notes
    -----
    closed_segment_tol defaults to tol. If you have very short open segments whose
    endpoints are closer than tol, pass a smaller closed_segment_tol.
    """
    curves = list(curves)

    if not curves:
        raise ValueError("Empty input list")

    tol = float(tol)

    if tol <= 0:
        raise ValueError("tol must be > 0")

    if closed_segment_tol is None:
        closed_segment_tol = tol

    closed_segment_tol = float(closed_segment_tol)

    if closed_segment_tol < 0:
        raise ValueError("closed_segment_tol must be >= 0")

    if knot_removal_tolerance is None:
        knot_removal_tolerance = tol

    base = [_copy_curve_as_tuple(c) for c in curves]

    dim = base[0].control_points.shape[1] if base[0].control_points.ndim == 2 else None

    if dim is None:
        raise ValueError("control_points must be a 2D array of shape (n, dim)")

    for i, c in enumerate(base):
        if c.control_points.ndim != 2 or c.control_points.shape[1] != dim:
            raise ValueError(f"Curve {i} has inconsistent control point dimension")

        if len(c.control_points) != len(c.weights):
            raise ValueError(f"Curve {i} has len(control_points) != len(weights)")

        if len(c.knot) < c.order * 2:
            raise ValueError(f"Curve {i} has an unexpectedly short knot vector")

    result_items = []

    closed_indices = []
    open_indices = []

    for i, c in enumerate(base):
        if closed_segment_tol > 0 and _is_c0_closed(c, closed_segment_tol):
            closed_indices.append(i)
        else:
            open_indices.append(i)

    # Closed input curves pass through as singleton outputs.
    for ci in closed_indices:
        native_length = _curve_domain_length(base[ci])

        spec = JoinChainSpec(
            segments=(
                JoinSegmentSpec(
                    source_index=ci,
                    is_reversed=False,
                    degree_elevations=0,
                    parameter_length=native_length,
                ),
            ),
            target_order=int(base[ci].order),
            is_cycle=True,
            is_singleton=True,
            was_closed_input=True,
        )

        result_items.append((ci, _copy_curve_as_tuple(base[ci]), spec))

    if open_indices:
        start_node, end_node, node_incident, _selected_pairs = _nearest_endpoint_nodes(
            base,
            open_indices,
            tol,
        )

        unused = set(open_indices)

        def component_from_curve(seed):
            comp = set()
            stack = [seed]

            while stack:
                ci = stack.pop()

                if ci in comp:
                    continue

                comp.add(ci)

                for nd in (start_node[ci], end_node[ci]):
                    for cj, _side in node_incident[nd]:
                        if cj not in comp:
                            stack.append(cj)

            return comp

        def node_degrees_for_component(comp):
            deg = defaultdict(int)

            for ci in comp:
                a = int(start_node[ci])
                b = int(end_node[ci])

                if a == b:
                    deg[a] += 2
                else:
                    deg[a] += 1
                    deg[b] += 1

            return deg

        def build_ordered_chain(comp):
            """
            Return:
                seq, is_cycle

            where seq is list[(curve_index, forward_flag)].

            forward_flag=True means use the curve as-is.
            forward_flag=False means reverse it.
            """
            comp = set(comp)

            if len(comp) == 1:
                ci = next(iter(comp))
                return [(ci, True)], False

            deg = node_degrees_for_component(comp)
            end_nodes = sorted([nd for nd, d in deg.items() if d == 1])

            if len(end_nodes) == 2:
                # Simple path.
                is_cycle = False
                chain_start = end_nodes[0]

                used_local = set()
                seq = []
                cur_node = chain_start

                while True:
                    candidates = [
                        cj
                        for cj, _side in node_incident[cur_node]
                        if cj in comp and cj not in used_local
                    ]

                    if not candidates:
                        break

                    candidates.sort()
                    cj = candidates[0]

                    fwd = int(start_node[cj]) == int(cur_node)

                    seq.append((cj, fwd))
                    used_local.add(cj)

                    cur_node = int(end_node[cj]) if fwd else int(start_node[cj])

                if len(used_local) != len(comp):
                    raise ValueError(
                        "Topology error while building path chain: did not consume all curves. "
                        "This typically indicates inconsistent tolerance or endpoint matching."
                    )

                return seq, is_cycle

            if len(end_nodes) == 0:
                # Closed cycle made from multiple open segments.
                is_cycle = True
                start_curve = min(comp)

                a = int(start_node[start_curve])
                b = int(end_node[start_curve])

                # Deterministic orientation choice for the first segment.
                fwd0 = (a, b) <= (b, a)

                start_node_chain = a if fwd0 else b
                cur_node = b if fwd0 else a

                used_local = {start_curve}
                seq = [(start_curve, fwd0)]

                while len(used_local) < len(comp):
                    candidates = [
                        cj
                        for cj, _side in node_incident[cur_node]
                        if cj in comp and cj not in used_local
                    ]

                    if not candidates:
                        raise ValueError(
                            "Topology error while building cycle chain: got stuck before consuming all curves. "
                            "This typically indicates inconsistent endpoint matching."
                        )

                    candidates.sort()
                    cj = candidates[0]

                    fwd = int(start_node[cj]) == int(cur_node)

                    seq.append((cj, fwd))
                    used_local.add(cj)

                    cur_node = int(end_node[cj]) if fwd else int(start_node[cj])

                if cur_node != start_node_chain:
                    raise ValueError(
                        "Topology error while building cycle chain: traversal did not return to the start node."
                    )

                return seq, is_cycle

            raise ValueError(
                f"Non-chain connectivity after nearest endpoint matching: component has "
                f"{len(end_nodes)} degree-1 nodes; expected 0 for a cycle or 2 for a path."
            )

        def oriented_curve(ci, forward):
            return base[ci] if forward else _reverse_curve(base[ci])

        while unused:
            seed = min(unused)
            comp = component_from_curve(seed)
            unused -= comp

            seq, is_cycle = build_ordered_chain(comp)
            min_source_index = min(ci for ci, _fwd in seq)

            if len(seq) == 1 and len(comp) == 1:
                ci, fwd = seq[0]
                native_length = _curve_domain_length(base[ci])

                spec = JoinChainSpec(
                    segments=(
                        JoinSegmentSpec(
                            source_index=ci,
                            is_reversed=not fwd,
                            degree_elevations=0,
                            parameter_length=native_length,
                        ),
                    ),
                    target_order=int(base[ci].order),
                    is_cycle=False,
                    is_singleton=True,
                    was_closed_input=False,
                )

                result_items.append(
                    (
                        min_source_index,
                        _copy_curve_as_tuple(oriented_curve(ci, fwd)),
                        spec,
                    )
                )

                continue

            target_order = max(int(base[ci].order) for ci, _fwd in seq)

            pieces = [
                _copy_curve_as_tuple(oriented_curve(ci, fwd))
                for ci, fwd in seq
            ]

            # Snap before derivative testing and before degree elevation.
            _snap_join_endpoints(pieces, is_cycle=is_cycle)

            (
                segment_lengths,
                interfaces,
                cycle_closure_is_c1_directional,
            ) = _compute_c1_reparameterisation(
                pieces,
                is_cycle=is_cycle,
                enabled=bool(reparameterize_c1),
                direction_tol=float(c1_direction_tol),
                derivative_tol=float(derivative_tol),
            )

            spec_segments = tuple(
                JoinSegmentSpec(
                    source_index=ci,
                    is_reversed=not fwd,
                    degree_elevations=target_order - int(base[ci].order),
                    parameter_length=float(segment_lengths[j]),
                )
                for j, (ci, fwd) in enumerate(seq)
            )

            joined, interior = _link_curves_with_degree_counts(
                pieces,
                [s.degree_elevations for s in spec_segments],
                target_order=target_order,
                segment_lengths=segment_lengths,
            )

            if len(interior) != len(interfaces):
                raise RuntimeError(
                    f"Internal error: got {len(interior)} interior knots but "
                    f"{len(interfaces)} interfaces."
                )

            interfaces = tuple(
                replace(iface, knot=float(knot))
                for iface, knot in zip(interfaces, interior)
            )

            joined, interfaces, removed_knots = _try_remove_c1_join_knots(
                joined,
                interfaces,
                remove_c1_knots=bool(remove_c1_knots),
                knot_removal_tolerance=float(knot_removal_tolerance),
                knot_remover=knot_remover,
            )

            spec = JoinChainSpec(
                segments=spec_segments,
                target_order=target_order,
                is_cycle=is_cycle,
                is_singleton=False,
                was_closed_input=False,
                interfaces=interfaces,
                removed_knots=removed_knots,
                cycle_closure_is_c1_directional=cycle_closure_is_c1_directional,
            )

            result_items.append((min_source_index, joined, spec))

    result_items.sort(key=lambda x: x[0])

    joined_curves = [x[1] for x in result_items]
    join_specs = [x[2] for x in result_items]

    return joined_curves, join_specs


def join_curves_by_spec(
    curves: Sequence[NURBSCurveTuple],
    join_specs: Sequence[JoinChainSpec],
    *,
    snap: bool = True,
    strict_order: bool = True,
    apply_recorded_knot_removals: bool = True,
    validate_replayed_knot_removal: bool = False,
    replay_knot_removal_tolerance: float = 1e-4,
    knot_remover=None,
):
    """
    Replay a previously computed join operation on another corresponding curve set.

    This applies:
      - source segment order
      - reversal flags
      - explicit degree-elevation counts
      - recorded parameter lengths
      - recorded successful knot removals

    Example
    -------
    joined_xyz, specs = join_curves(xyz_curves, tol=1e-6)

    joined_uv = join_curves_by_spec(
        uv_curves,
        specs,
        validate_replayed_knot_removal=False,
    )

    If validate_replayed_knot_removal=False, recorded knot removals are applied
    even if the replay curve does not satisfy the supplied geometric tolerance.
    This is useful for corresponding curves in UV / projected / auxiliary spaces.
    """
    curves = list(curves)
    out = []

    if apply_recorded_knot_removals:
        knot_remover = _get_default_knot_remover(knot_remover)

    for si, spec in enumerate(join_specs):
        if not spec.segments:
            raise ValueError(f"Join spec {si} has no segments")

        pieces = []
        degree_counts = []
        segment_lengths = []

        for seg in spec.segments:
            ci = int(seg.source_index)

            if ci < 0 or ci >= len(curves):
                raise IndexError(
                    f"Join spec {si} references source curve {ci}, "
                    f"but only {len(curves)} curves were provided."
                )

            crv = _copy_curve_as_tuple(curves[ci])

            if seg.is_reversed:
                crv = _reverse_curve(crv)

            pieces.append(crv)
            degree_counts.append(int(seg.degree_elevations))
            segment_lengths.append(float(seg.parameter_length))

        if snap:
            _snap_join_endpoints(pieces, is_cycle=bool(spec.is_cycle))

        target_order = int(spec.target_order) if strict_order else None

        elevated = _elevate_pieces_by_counts(
            pieces,
            degree_counts,
            target_order=target_order,
        )

        if len(elevated) == 1 and spec.is_singleton:
            curve = elevated[0]
        else:
            curve, _interior = _link_curves_same_order(
                elevated,
                segment_lengths=segment_lengths,
            )

        if apply_recorded_knot_removals and spec.removed_knots:
            for u in spec.removed_knots:
                res = knot_remover(
                    curve,
                    float(u),
                    tolerance=float(replay_knot_removal_tolerance),
                )

                new_curve = _kr_curve(res)

                if new_curve is None:
                    raise ValueError(
                        f"Could not replay knot removal at u={u} for join spec {si}"
                    )

                if validate_replayed_knot_removal and not _kr_success(res):
                    raise ValueError(
                        f"Replay knot removal at u={u} for join spec {si} failed "
                        f"with error {_kr_error(res)}"
                    )

                curve = new_curve

        out.append(curve)

    return out