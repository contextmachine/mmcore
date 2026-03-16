"""
Squared-distance Bernstein net classification engine.

Analyzes N-dimensional Bernstein control nets of ||Delta||^2
(from mmcore.numeric.bern_sq_dist) to classify intersection type.
"""
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from mmcore.numeric.bern import (
    bernstein_partial_derivative_coeffs,
    bernstein_all_boundaries_nd,
    bernstein_boundary_nd,
    de_casteljau_restrict_nd,
    bernstein_eval_nd,
)

# ---------------------------------------------------------------------------
# Classification constants
# ---------------------------------------------------------------------------
NO_INTERSECTION = 0
UNIQUE_ISOLATED = 1
OVERLAP = 2
INDETERMINATE = 3


@dataclass
class Classification:
    kind: int
    boundary_zeros: list = field(default_factory=list)  # (axis, side) pairs
    overlap_endpoints: list = field(default_factory=list)
    notes: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _weight_max_product(Pw: NDArray, Qw: NDArray) -> float:
    """max(|Pw|) * max(|Qw|) — the weight scaling denominator."""
    return float(np.max(np.abs(Pw)) * np.max(np.abs(Qw)))


def _add_value_dim(F: NDArray) -> NDArray:
    """Add trailing value dimension required by bern.py functions."""
    return F[..., np.newaxis]


def _squeeze_value_dim(arr: NDArray) -> NDArray:
    """Remove trailing value dimension."""
    return arr[..., 0]


# ---------------------------------------------------------------------------
# Check 1: Min-of-net positive (weight-corrected)
# ---------------------------------------------------------------------------

def _check_min_of_net(F: NDArray, atol: float, w_scale: float) -> bool:
    """Return True if net proves NO_INTERSECTION."""
    lb = float(np.min(F)) / (w_scale ** 2)
    return lb > atol * atol


# ---------------------------------------------------------------------------
# Check 2: Lipschitz tightening
# ---------------------------------------------------------------------------

def _check_lipschitz(F: NDArray, atol: float, w_scale: float) -> bool:
    """Return True if Lipschitz bound proves NO_INTERSECTION."""
    ndim = F.ndim
    if ndim < 1:
        return False

    # Evaluate F at midpoint
    Fv = _add_value_dim(F)
    mid_params = tuple(0.5 for _ in range(ndim))
    f_mid = float(bernstein_eval_nd(Fv, mid_params).squeeze())

    # Compute sup-norms of partial derivative nets
    lip_sum = 0.0
    for ax in range(ndim):
        dF = bernstein_partial_derivative_coeffs(Fv, axis=ax)
        dF_scalar = _squeeze_value_dim(dF)
        lip_sum += float(np.max(np.abs(dF_scalar)))

    lower = f_mid - 0.5 * lip_sum
    lb = lower / (w_scale ** 2)
    return lb > atol * atol


# ---------------------------------------------------------------------------
# Check 3: Boundary zero analysis
# ---------------------------------------------------------------------------

def _find_boundary_zeros(F: NDArray, atol: float, w_scale: float) -> list:
    """Return list of (axis, side) pairs where boundary may touch zero."""
    threshold = (atol * w_scale) ** 2
    Fv = _add_value_dim(F)
    boundaries = bernstein_all_boundaries_nd(Fv)
    zeros = []
    for axis, side, bnd_grid in boundaries:
        bnd_scalar = _squeeze_value_dim(bnd_grid)
        if float(np.min(bnd_scalar)) < threshold:
            zeros.append((axis, side))
    return zeros


# ---------------------------------------------------------------------------
# Check 4: Uniqueness certificate (2D only — CCX)
# ---------------------------------------------------------------------------

def _check_uniqueness_2d(F: NDArray) -> bool:
    """
    For a 2D net (CCX case), check whether the zero is unique-isolated.

    Conditions:
    1. Each partial derivative net has both positive AND negative coefficients
       (sign change in each direction).
    2. Hessian positive-definite: min(Fuu) > 0, min(Fvv) > 0,
       min(Fuu)*min(Fvv) - max(|Fuv|)^2 > 0.
    """
    if F.ndim != 2:
        return False

    Fv = _add_value_dim(F)

    # Partial derivative nets
    Fu = _squeeze_value_dim(bernstein_partial_derivative_coeffs(Fv, axis=0))
    Fv_d = _squeeze_value_dim(bernstein_partial_derivative_coeffs(Fv, axis=1))

    # Condition 1: sign changes in both partial derivatives
    fu_has_pos = np.any(Fu > 0)
    fu_has_neg = np.any(Fu < 0)
    fv_has_pos = np.any(Fv_d > 0)
    fv_has_neg = np.any(Fv_d < 0)

    if not (fu_has_pos and fu_has_neg and fv_has_pos and fv_has_neg):
        return False

    # Condition 2: Hessian positive-definite
    # Second partial derivatives
    Fuu = _squeeze_value_dim(bernstein_partial_derivative_coeffs(_add_value_dim(Fu), axis=0))
    Fvv = _squeeze_value_dim(bernstein_partial_derivative_coeffs(_add_value_dim(Fv_d), axis=1))

    # Mixed partial: differentiate Fu w.r.t. axis 1
    Fuv = _squeeze_value_dim(bernstein_partial_derivative_coeffs(_add_value_dim(Fu), axis=1))

    min_fuu = float(np.min(Fuu))
    min_fvv = float(np.min(Fvv))
    max_abs_fuv = float(np.max(np.abs(Fuv)))

    if min_fuu <= 0 or min_fvv <= 0:
        return False

    if min_fuu * min_fvv - max_abs_fuv ** 2 <= 0:
        return False

    return True


# ---------------------------------------------------------------------------
# Check 5: Overlap certificate
# ---------------------------------------------------------------------------

def _check_overlap(F: NDArray, atol: float, w_scale: float, boundary_zeros: list) -> tuple:
    """
    Check for overlap: opposite boundaries on same axis both touch zero,
    and the function values along the connecting path are near zero.
    Returns (is_overlap, endpoints).
    """
    threshold = (atol * w_scale) ** 2
    ndim = F.ndim

    # Group boundary zeros by axis
    axis_sides = {}
    for axis, side in boundary_zeros:
        axis_sides.setdefault(axis, set()).add(side)

    for axis, sides in axis_sides.items():
        if 0 in sides and 1 in sides:
            # Both boundaries on this axis touch zero.
            Fv = _add_value_dim(F)

            bnd0 = _squeeze_value_dim(bernstein_boundary_nd(Fv, axis=axis, side=0))
            bnd1 = _squeeze_value_dim(bernstein_boundary_nd(Fv, axis=axis, side=1))

            remaining_shapes = [F.shape[a] for a in range(ndim) if a != axis]
            remaining_orig = [a for a in range(ndim) if a != axis]

            # Collect ALL near-zero multi-indices on each boundary as
            # candidate start/end points for the connecting path.
            def _near_zero_params(bnd_scalar, shapes):
                flat = bnd_scalar.ravel()
                candidates = []
                for fi in range(len(flat)):
                    if abs(float(flat[fi])) < threshold:
                        idx = np.unravel_index(fi, bnd_scalar.shape)
                        params = []
                        for ii, s in zip(idx, shapes):
                            params.append(ii / max(s - 1, 1))
                        candidates.append(params)
                if not candidates:
                    # Fallback: use argmin
                    idx = np.unravel_index(np.argmin(bnd_scalar), bnd_scalar.shape)
                    params = [ii / max(s - 1, 1) for ii, s in zip(idx, shapes)]
                    candidates.append(params)
                return candidates

            cands0 = _near_zero_params(bnd0, remaining_shapes)
            cands1 = _near_zero_params(bnd1, remaining_shapes)

            def _build_full_point(axis_val, other_params):
                pt = [0.0] * ndim
                pt[axis] = axis_val
                for k, orig_ax in enumerate(remaining_orig):
                    pt[orig_ax] = other_params[k]
                return tuple(pt)

            n_samples = max(2 * max(F.shape), 5)
            tol_sq = atol * atol

            # Try each pairing of start/end candidates
            for p0 in cands0:
                for p1 in cands1:
                    start_pt = _build_full_point(0.0, p0)
                    end_pt = _build_full_point(1.0, p1)

                    all_near_zero = True
                    for k in range(n_samples + 1):
                        alpha = k / n_samples
                        pt = tuple(
                            s * (1.0 - alpha) + e * alpha
                            for s, e in zip(start_pt, end_pt)
                        )
                        val = float(bernstein_eval_nd(Fv, pt).squeeze())
                        if abs(val) / (w_scale ** 2) > tol_sq:
                            all_near_zero = False
                            break

                    if all_near_zero:
                        avg_params = [
                            (a + b) / 2.0 for a, b in zip(p0, p1)
                        ]
                        return True, avg_params

    return False, []


# ---------------------------------------------------------------------------
# Main classification function
# ---------------------------------------------------------------------------

def classify_sq_dist_net(
    F: NDArray,
    atol: float,
    Pw: NDArray,
    Qw: NDArray,
) -> Classification:
    """
    Classify the intersection type from a squared-distance Bernstein net.

    Parameters
    ----------
    F : NDArray
        Scalar Bernstein net (no trailing value dimension).
    atol : float
        Geometric tolerance.
    Pw : NDArray
        Weight array for the first geometry.
    Qw : NDArray
        Weight array for the second geometry.

    Returns
    -------
    Classification
        Result with kind in {NO_INTERSECTION, UNIQUE_ISOLATED, OVERLAP, INDETERMINATE}.
    """
    Pw = np.asarray(Pw, dtype=float)
    Qw = np.asarray(Qw, dtype=float)
    F = np.asarray(F, dtype=float)

    w_scale = _weight_max_product(Pw, Qw)

    # Check 1: Min-of-net positive
    if _check_min_of_net(F, atol, w_scale):
        return Classification(
            kind=NO_INTERSECTION,
            notes="min-of-net positive (weight-corrected)",
        )

    # Check 2: Lipschitz tightening
    if _check_lipschitz(F, atol, w_scale):
        return Classification(
            kind=NO_INTERSECTION,
            notes="Lipschitz lower bound positive",
        )

    # Check 3: Boundary zero analysis
    boundary_zeros = _find_boundary_zeros(F, atol, w_scale)

    # Check 4: Uniqueness certificate (2D only)
    if F.ndim == 2 and _check_uniqueness_2d(F):
        return Classification(
            kind=UNIQUE_ISOLATED,
            boundary_zeros=boundary_zeros,
            notes="uniqueness certificate (Hessian PD + sign changes)",
        )

    # Check 5: Overlap certificate
    is_overlap, endpoints = _check_overlap(F, atol, w_scale, boundary_zeros)
    if is_overlap:
        return Classification(
            kind=OVERLAP,
            boundary_zeros=boundary_zeros,
            overlap_endpoints=endpoints,
            notes="opposite boundaries both zero, connecting net all below tolerance",
        )

    # Fallback
    return Classification(
        kind=INDETERMINATE,
        boundary_zeros=boundary_zeros,
        notes="no conclusive classification",
    )
