"""Bezier curve-surface intersection using squared-distance Bernstein net classification.

This module implements a subdivision-based CSX algorithm that uses the
squared-distance net ``||C(t) - S(u,v)||^2`` in trivariate Bernstein form
to classify cells as NO_INTERSECTION, UNIQUE_ISOLATED, OVERLAP,
BOUNDARY_ZERO, or INDETERMINATE.

Structurally identical to the CCX algorithm in ``_bez_ccx4.py`` but one
dimension larger (trivariate sq-dist net).
"""
from __future__ import annotations

from math import comb

import math
import numpy as np

from mmcore.numeric._bezier_common import _compute_remaining_intervals
from mmcore.numeric.aabb import aabb_offset
from mmcore.numeric._aabb import aabb_intersect, aabb
from mmcore.numeric._work_budget import DownCounter
from mmcore.numeric.bern import (
    de_casteljau_split_nd, bernstein_boundary_nd,
    bernstein_partial_derivative_coeffs,
)
from mmcore.numeric.bern_sq_dist import curve_surface_distance_squared_net_homog
from mmcore.numeric._bezier_common import (
    extract_weights, eval_curve, eval_surface, eval_curve_d1, eval_surface_d1,
    newton_csx, bernstein_product_1d, subdivide_curve, subdivide_sq_dist_net,
    restrict_net_axis, restrict_net_axis_v, geometry_collapsed,
)
from mmcore.numeric.intersection.ccx._bez_ccx4 import bez_ccx as bez_ccx_v4
# Shared with ccx's twin: both centerings run the same four roundings.
from mmcore.numeric.intersection.ccx._bez_ccx4 import (
    _CENTERING_OPS as _CCX_CENTERING_OPS,
)
from mmcore.numeric.intersection._sq_dist_classify import (
    BoundaryZero,
    _boundary_zero_to_param_point,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# L52 slice 7: shared implementation in _bezier_common (verbatim move).
_subdivide_curve = subdivide_curve


def _surface_fibers_at_point(surface, u, v, point, atol, rational):
    """Free surface axes whose entire control hull fits the CAD tolerance.

    At a pinched isoline, changing its free parameter produces more
    representatives of the same geometric hit. Keep the boundary witnesses
    but reuse their coverage while searching the remaining parameter box.
    Positive rational weights give the same Cartesian convex-hull bound.
    """
    def fits(net):
        if not np.all(np.isfinite(net)):
            return False
        if rational:
            if np.any(net[..., -1] <= 0.):
                return False
            points = net[..., :-1]/net[..., -1:]
        else:
            points = net
        return bool(np.all(np.linalg.norm(points-point, axis=-1) <= atol))

    if fits(surface):
        return ((1, 2),)
    fibers = []
    for fixed_axis, value in ((0, u), (1, v)):
        left, _ = _subdivide_surface(surface, fixed_axis, float(value))
        isoline = left[-1, :, :] if fixed_axis == 0 else left[:, -1, :]
        if fits(isoline):
            fibers.append((2-fixed_axis,))
    return tuple(fibers)


def _matching_surface_fiber(entry, root, point, atol, tolerances):
    if np.linalg.norm(point-entry['point']) > atol:
        return ()
    parameters = (entry['t'], entry['u'], entry['v'])
    for free_axes in entry.get('_surface_fibers', ()):
        if all(axis in free_axes or abs(a-b) <= 2.*tol
               for axis, (a, b, tol) in enumerate(zip(parameters, root, tolerances))):
            return free_axes
    return ()


def _subdivide_surface(ctrl, axis, t=0.5):
    n = ctrl.shape[axis] - 1
    tmp = np.moveaxis(ctrl.copy(), axis, 0)
    left = [tmp[0].copy()]
    right_rev = [tmp[n].copy()]
    for r in range(1, n + 1):
        tmp[:n + 1 - r] = (1.0 - t) * tmp[:n + 1 - r] + t * tmp[1:n + 2 - r]
        left.append(tmp[0].copy())
        right_rev.append(tmp[n - r].copy())
    left = np.moveaxis(np.array(left), 0, axis)
    right = np.moveaxis(np.array(right_rev[::-1]), 0, axis)
    return left, right


# L52 slice 7: shared implementation in _bezier_common (verbatim move).
_subdivide_sq_dist_net = subdivide_sq_dist_net


def _subdivide_surface_weights(sw, axis, t=0.5):
    n = sw.shape[axis] - 1
    tmp = np.moveaxis(sw.copy(), axis, 0)
    left = [tmp[0].copy()]
    right_rev = [tmp[n].copy()]
    for r in range(1, n + 1):
        tmp[:n + 1 - r] = (1.0 - t) * tmp[:n + 1 - r] + t * tmp[1:n + 2 - r]
        left.append(tmp[0].copy())
        right_rev.append(tmp[n - r].copy())
    left = np.moveaxis(np.array(left), 0, axis)
    right = np.moveaxis(np.array(right_rev[::-1]), 0, axis)
    return left, right


def _residual_vec_net(C, S, rational):
    """Bernstein coefficient net of the homogeneous vector residual

        G(t,u,v) = C(t)·w_S(u,v) − S(u,v)·w_C(t),   shape (p+1, m+1, n+1, 3).

    t and (u,v) are disjoint variables, so the products are plain outer
    products — no degree elevation. For non-rational input the weights are
    1 and this is the difference net C[i] − S[j,k].

    Unlike the squared-distance net (which is sign-blind: it must resolve
    |G|² > 0 to prune, an O(h²) hull race against d²), a single component
    whose coefficient hull excludes zero proves G ≠ 0 in the cell. Near
    transversal roots the components change sign cleanly, so this prune
    terminates the near-contact band orders of magnitude earlier.
    """
    if rational:
        Cp, Cw = C[:, :-1], C[:, -1]
        Sp, Sw = S[..., :-1], S[..., -1]
    else:
        Cp = C
        Cw = np.ones(C.shape[0], dtype=np.float64)
        Sp = S
        Sw = np.ones(S.shape[:2], dtype=np.float64)
    return (Cp[:, None, None, :] * Sw[None, :, :, None]
            - Sp[None, :, :, :] * Cw[:, None, None, None])


def _residual_excludes_zero(G_cell, roundoff=None):
    """True if some component's Bernstein hull excludes 0 → no zero of G.

    L52 slice 6c: exclusion only beyond the L1 roundoff margin
    ``128·ε·max|coeff|`` (§4 invariant — margins make exclusion stricter,
    the sound direction; ccx's identity-side twin carries a depth-scaled
    margin, this was the one unmargined sign prune left). Fixture-first
    measurement: 240 exact-Fraction restriction-chain comparisons on
    tangential-graze nets produced 0 wrongful exclusions and 0 exclusions
    within 1e-12·scale of the boundary, so the margin costs nothing in
    practice — it is insurance for restriction chains outside that family.
    """
    eps = float(np.finfo(np.float64).eps)
    for c in range(3):
        comp = G_cell[..., c]
        margin = 128.0 * eps * float(np.abs(comp).max())
        if roundoff is not None:
            margin = max(margin, float(roundoff[c]))
        if float(comp.min()) > margin or float(comp.max()) < -margin:
            return True
    return False


def _residual_aligned_excludes_zero(G_cell, roundoff=None):
    """Geometry-ALIGNED zero exclusion (ledger L60).

    The axis-component test above converges linearly in clearance but only
    along world axes: when the residual direction is diagonal, every axis
    component mixes the LARGE tangential variation with the SMALL
    clearance and nothing excludes until cells shrink to clearance scale
    (measured: 28,961 cells proving one near-band pair empty).  Projecting
    the net onto its own mean direction asks the same question in the
    geometry's frame: ``dot(G, n_hat)`` is an EXACT Bernstein net (a fixed
    linear combination of the exactly-built component nets), so a
    margined hull exclusion on it proves ``dot(G, n_hat) != 0`` and hence
    ``G != 0`` everywhere in the cell — sound for ANY fixed direction; the
    mean maximizes the chance of clearing (same pair: 511 nodes, ~1000x).
    The 128·ε·max|coeff| L1 margin covers the build, combination and
    restriction roundoff (the 6c insurance class; margins make exclusion
    stricter, the sound direction).
    """
    flat = G_cell.reshape(-1, 3)
    mean = flat.mean(axis=0)

    def excludes(direction):
        n_len = float(np.linalg.norm(direction))
        if n_len <= 0.0 or not np.isfinite(n_len):
            return False
        n_hat = direction / n_len
        scalar = (G_cell[..., 0] * n_hat[0]
                  + G_cell[..., 1] * n_hat[1]
                  + G_cell[..., 2] * n_hat[2])
        eps = float(np.finfo(np.float64).eps)
        margin = 128.0 * eps * float(np.abs(scalar).max())
        if roundoff is not None:
            operands = float(np.dot(
                np.abs(n_hat), np.max(np.abs(flat), axis=0)))
            margin = max(margin, float(np.dot(np.abs(n_hat), roundoff))
                         + (5.0*eps/(1.0-5.0*eps))*operands)
        return bool(float(scalar.min()) > margin
                    or float(scalar.max()) < -margin)

    if excludes(mean):
        return True
    # The mean can contain a large tangential displacement. The direction
    # of least coefficient variation exposes the constant normal gap of
    # nearby planar/grazing geometry. Eigenanalysis only proposes a fixed
    # direction; the same bounded Bernstein hull test certifies rejection.
    centered = flat - mean
    try:
        _, directions = np.linalg.eigh(centered.T @ centered)
    except np.linalg.LinAlgError:
        return False
    return excludes(directions[:, 0])


# L52 slice 7: shared implementation in _bezier_common (verbatim move).
_restrict_net_axis_v = restrict_net_axis_v


def _compute_param_tols_csx(C, S, atol, rational):
    """Compute parametric tolerances for curve and surface.

    Returns (ptol_t, ptol_u, ptol_v).
    """
    from mmcore.nurbs._nurbs_param_tol import bez_curve_param_tolerance, bez_surface_param_tolerance
    ptol_t = float(bez_curve_param_tolerance(C, tol=atol, rational=rational))
    ptol_u, ptol_v = bez_surface_param_tolerance(S, tol=atol, rational=rational)
    return ptol_t, float(ptol_u), float(ptol_v)


from mmcore.numeric.intersection._root_box_certificate import (
    unique_root_box as _unique_csx_root_box,
    residual_roundoff_bound as _csx_residual_roundoff_bound,
    jacobian_is_injective as _jacobian_is_injective,
    root_existence_certificate as _root_existence_certificate,
    affine_residual_is_zero as _affine_residual_is_zero,
    root_boxes_have_same_root as _root_boxes_have_same_root,
    root_box_enclosure as _root_box_enclosure,
)


def _certify_csx_root_existence(entries, C, S, G, radii, rational, source_scale):
    """Discharge only roots with existence evidence; retain unknown boxes."""
    resolved, unresolved = [], []
    for entry in entries:
        parameters = tuple(entry[key] for key in ('t', 'u', 'v'))
        box = entry.get('_unique_box') or tuple(
            (max(0., t-r), min(1., t+r)) for t, r in zip(parameters, radii))
        certificate = (entry['root_existence_certification'] if entry.get('_existence_established')
                       else 'exact_boundary_curve_intersection' if entry.get('_boundary_root_proof')
                       else _root_existence_certificate(
                           C, S, parameters, box, G, rational, source_scale))
        if certificate is not None:
            entry['root_existence_certification'] = certificate
            resolved.append(entry)
        else:
            reservation = {
                't_range': box[0], 'u_range': box[1], 'v_range': box[2],
                'reason': 'root_existence', 'candidate': parameters,
            }
            unique_box = entry.get('_unique_box') or _unique_csx_root_box(
                G, parameters, radii, source_scale)
            if unique_box is not None:
                reservation['parameter_uniqueness_box'] = unique_box
            unresolved.append(reservation)
    return resolved, unresolved


def _in_csx_root_box(root, box):
    return all(lo <= x <= hi for x, (lo, hi) in zip(root, box))


def _certify_csx_owned_cell(net, cell_box, known, source_scale=None, charge=None,
                            coefficient_error=None):
    """Prove a cell contains no root other than an already published root.

    The cell need not contain that root or fit its existing isolating box.
    Source existence in the known enclosure and injectivity on their whole
    product-box union exclude every unreported root. None denotes denied
    work; False denotes an unproved cell. Neither result discards geometry.
    """
    units = max(1,(4*net.size+127)//128)
    for entry in known:
        enclosure = entry.get('parameter_root_box')
        if enclosure is None or not entry.get('root_existence_certification'):
            continue
        if charge is not None and not charge(units):
            return None
        # The shared helper proves union injectivity; this use supplies
        # existence only for the known root, not for the current cell.
        if _root_boxes_have_same_root(net,enclosure,cell_box,source_scale,coefficient_error):
            entry['_unique_box'] = tuple((min(a[0],b[0]),max(a[1],b[1]))
                                          for a,b in zip(enclosure,cell_box))
            return True
    return False


def _csx_residual_source_scale(C, S, rational):
    if not rational:
        return None
    products = (np.abs(C[:, None, None, :-1]) * np.abs(S[None, ..., -1:])
                + np.abs(S[None, ..., :-1]) * np.abs(C[:, None, None, -1:]))
    return np.max(products, axis=(0, 1, 2))




# Floor on sin(angle) below which the angle-aware tolerance stops tightening.
# 1e-3 corresponds to ≈ 0.06° between the curve tangent and the surface tangent
# plane; tighter than that we treat the configuration as effectively tangent
# (a different topology) and accept residuals at the floor scale.
_CSX_SIN_ANG_FLOOR = 1e-3


def _csx_eff_atol(C, S, t, u, v, atol, rational):
    """Angle-aware geometric tolerance for accepting a CSX root candidate.

    A pure ``||C(t) - S(u, v)|| < atol`` test is misleading when the curve
    grazes the surface: the residual stays small in a "thick gap" of width
    ``atol / sin(angle)`` around the true intersection, so Newton can settle
    at a stationary point of the squared distance that is *far* from any
    real root.

    The xyz distance from the true intersection at the candidate is
    approximately ``residual / sin(angle)`` where ``angle`` is the angle
    between the curve tangent ``C'(t)`` and the surface tangent plane.
    Requiring ``residual < atol * sin(angle)`` keeps the implied xyz error
    below ``atol`` regardless of grazing. The floor avoids collapsing the
    tolerance to zero in the exactly-tangent limit.
    """
    _pt_c, c_d = eval_curve_d1(C, float(t), rational=rational)
    _pt_s, s_du, s_dv = eval_surface_d1(S, float(u), float(v), rational=rational)
    N = np.cross(s_du, s_dv)

    n_norm = float(np.linalg.norm(N))
    c_norm = float(np.linalg.norm(c_d))

    if n_norm > 1e-30 and c_norm > 1e-30:
        sin_ang = abs(float(np.dot(c_d, N))) / (c_norm * n_norm)
    else:
        sin_ang = 0.0
    return atol * max(sin_ang, _CSX_SIN_ANG_FLOOR)


def _cartesian_controls_for_exactness(ctrl, rational):
    """Return finite Cartesian controls, or ``None`` for invalid weights."""
    ctrl = np.asarray(ctrl, dtype=np.float64)
    if rational:
        weights = ctrl[..., -1:]
        if (not np.all(np.isfinite(weights))
                or np.any(weights == 0.0)):
            return None
        points = ctrl[..., :-1] / weights
    else:
        points = ctrl
    if not np.all(np.isfinite(points)):
        return None
    return points


def _center_homogeneous_for_exactness(ctrl, rational, origin,
                                      with_source_scale=False):
    """Translate homogeneous controls by a common Cartesian origin.

    Normalize once before the homogeneous translation to avoid overflow in
    ``origin * weight``, then normalize the translated net again for stable
    evaluation.  Both normalizations are one common nonzero homogeneous
    factor and therefore leave the represented geometry unchanged.

    With ``with_source_scale`` this also returns the magnitude of the two
    operands the translation subtracts, carried through the same final
    normalization.  Because the division by ``scale`` happens BEFORE the
    subtraction, a coordinate whose exact translated value is zero arrives
    as cancellation noise rather than as an exact zero; any envelope built
    from the cancelled value alone is then circular.  Twin of ccx's
    `_center_curve_homogeneous_for_exactness`.
    """
    ctrl = np.asarray(ctrl, dtype=np.float64)
    if rational:
        homog = ctrl.copy()
    else:
        homog = np.concatenate(
            [ctrl, np.ones(ctrl.shape[:-1] + (1,), dtype=np.float64)],
            axis=-1)
    scale = float(np.max(np.abs(homog)))
    if not np.isfinite(scale) or scale <= 0.0:
        return (None, None) if with_source_scale else None
    homog /= scale
    shift = np.asarray(origin, dtype=np.float64) * homog[..., -1:]
    operand_mag = np.abs(homog[..., :-1]) + np.abs(shift)
    homog[..., :-1] -= shift
    local_scale = float(np.max(np.abs(homog)))
    if not np.isfinite(local_scale) or local_scale <= 0.0:
        return (None, None) if with_source_scale else None
    out = np.ascontiguousarray(homog / local_scale)
    if not with_source_scale:
        return out
    return out, np.ascontiguousarray(operand_mag / local_scale)


def _strict_csx_root_tol(C, S, rational):
    """Translation-invariant context for exact root certification."""
    c_pts = _cartesian_controls_for_exactness(C, rational)
    s_pts = _cartesian_controls_for_exactness(S, rational)
    if c_pts is None or s_pts is None:
        return None
    origin = np.asarray(c_pts).reshape(-1, 3)[0].copy()
    c_centered, c_src = _center_homogeneous_for_exactness(
        C, rational, origin, with_source_scale=True)
    s_centered, s_src = _center_homogeneous_for_exactness(
        S, rational, origin, with_source_scale=True)
    if c_centered is None or s_centered is None:
        return None
    c_local = c_centered[..., :-1] / c_centered[..., -1:]
    s_local = s_centered[..., :-1] / s_centered[..., -1:]
    points = np.vstack([
        np.asarray(c_local).reshape(-1, 3),
        np.asarray(s_local).reshape(-1, 3),
    ])
    component_scale = np.max(np.abs(points), axis=0)
    # Per-axis envelope of the centering's own cancellation, in the same
    # dehomogenized units as `component_scale`.  An axis that is genuinely
    # absent from both operands keeps a zero envelope, so this can never
    # hide a real offset; an axis that is only zero up to the translation's
    # roundoff gets the margin it actually needs.  Factor matches the
    # certificate's own `32 * degree_factor` family below.
    c_w = np.abs(c_centered[..., -1:])
    s_w = np.abs(s_centered[..., -1:])
    if np.any(c_w == 0.0) or np.any(s_w == 0.0):
        return None
    # Factor: this envelope prices ONE subtraction (the common-origin
    # centering), not a degree-n accumulation — see `_CENTERING_OPS` in
    # ccx's twin for the per-rounding derivation.  Using the certificate's
    # own `32*degree_factor` family here was ~256x too loose (measured), and
    # on an ACCEPT path that surplus is the false-certification window: a
    # curve lifted 1e-13 off the surface certified as 'exact' at 195/200
    # random world positions of magnitude 1e2.
    centering_noise = (_CCX_CENTERING_OPS * np.finfo(np.float64).eps
                       * np.maximum(
                           np.max(np.asarray(c_src / c_w).reshape(-1, 3),
                                  axis=0),
                           np.max(np.asarray(s_src / s_w).reshape(-1, 3),
                                  axis=0)))
    component_scale = np.where(
        component_scale > centering_noise, component_scale, 0.0)
    return c_centered, s_centered, component_scale, centering_noise


def _strict_csx_residual_ok(C, S, t, u, v, rational, strict_context):
    """Componentwise floating envelope for exact curve/surface equality."""
    if strict_context is None:
        pc = eval_curve(C, float(t), rational=rational)
        ps = eval_surface(S, float(u), float(v), rational=rational)
        return False, pc - ps
    c_centered, s_centered, component_scale = strict_context[:3]
    centering_noise = strict_context[3]
    pc = eval_curve(c_centered, float(t), rational=True)
    ps = eval_surface(s_centered, float(u), float(v), rational=True)
    if not (np.all(np.isfinite(pc)) and np.all(np.isfinite(ps))):
        return False, pc - ps
    scale = np.maximum(
        np.asarray(component_scale, dtype=np.float64),
        np.maximum(np.abs(pc), np.abs(ps)))
    degree_factor = max(
        1, len(C) + int(S.shape[0]) + int(S.shape[1]))
    bound = (32.0 * degree_factor * np.finfo(np.float64).eps) * scale
    # DEGENERATE AXES ONLY.  An axis with real content keeps the relative
    # bound above untouched: widening it there costs resolution where it
    # matters most, because near a tangency the parameter error grows as
    # the square root of the residual envelope (a blanket second term moved
    # this module's near-tangent root off 0.5 by 1.7e-7 against a 1e-7 bar).
    # An axis whose entire content is the common-origin centering's own
    # cancellation has no scale to be relative to — `component_scale` is 0
    # there by construction — so its bound is that cancellation envelope.
    # Sources genuinely zero on an axis give envelope 0 and the pre-existing
    # exact-zero behaviour is preserved bit-for-bit; a planar pair merely
    # translated off its own plane now certifies at every world position
    # instead of ~70% of them.
    absent = np.asarray(component_scale, dtype=np.float64) == 0.0
    bound = np.where(
        absent,
        np.maximum(bound, np.asarray(centering_noise, dtype=np.float64)),
        bound)
    residual = pc - ps
    return bool(np.all(np.abs(residual) <= bound)), residual


def _polish_csx_root(C, S, t, u, v, rational, strict_tol):
    """Unbounded polish plus a strict zero certificate.

    A cell-bounded Newton may stop against a cutout wall with a zero step
    while merely lying inside the geometric tolerance valley.  Such a stall
    is useful for subdivision guidance but is not a distinct root.  Re-polish
    in the full parameter domain and only type the result when the actual
    residual reaches the roundoff-scale root tolerance.
    """
    root_ok, G0 = _strict_csx_residual_ok(
        C, S, t, u, v, rational, strict_tol)
    if root_ok:
        return float(t), float(u), float(v), G0, True

    if strict_tol is None:
        polish_C, polish_S, polish_rational = C, S, rational
    else:
        polish_C, polish_S = strict_tol[0], strict_tol[1]
        polish_rational = True
        # Solve in the same component scales used by the root certificate.
        # A single absolute LM residual floor cannot polish a short axis:
        # it stops before that axis meets its relative equality envelope.
        scale = np.asarray(strict_tol[2], dtype=float)
        scale = np.where(scale > 0.0, scale, 1.0)
        polish_C = np.array(polish_C, copy=True)
        polish_S = np.array(polish_S, copy=True)
        polish_C[..., :-1] /= scale
        polish_S[..., :-1] /= scale
    t, u, v, G, _ = newton_csx(
        polish_C, polish_S, t, u, v,
        rational=polish_rational, bounds=None)
    ok, G = _strict_csx_residual_ok(
        C, S, t, u, v, rational, strict_tol)
    return float(t), float(u), float(v), G, bool(ok)


def _polish_csx_boundary_root(
    C, S, bz, t, u, v, rational, strict_tol, max_iter=48,
):
    """Polish a face candidate while preserving its certified boundary.

    A boundary zero has one parameter fixed exactly at 0 or 1.  Letting the
    ordinary three-variable Newton move that coordinate can turn an exterior
    root into an in-domain endpoint (the former case13 false positive), and
    it perturbs exact overlap endpoints just far enough to defeat their
    coefficient identity.  Solve the overdetermined three-residual/two-free-
    parameter problem instead, then apply the same strict equality test.
    """
    params = np.array([t, u, v], dtype=np.float64)
    params[bz.axis] = 0.0 if bz.side == 0 else 1.0
    free = [axis for axis in (0, 1, 2) if axis != bz.axis]
    if strict_tol is None:
        polish_C, polish_S, polish_rational = C, S, rational
    else:
        polish_C, polish_S = strict_tol[0], strict_tol[1]
        polish_rational = True

    for _ in range(max(0, int(max_iter))):
        ok, G = _strict_csx_residual_ok(
            C, S, *params, rational, strict_tol)
        if ok:
            return (*map(float, params), G, True)

        _pc, cd = eval_curve_d1(
            polish_C, params[0], rational=polish_rational)
        _ps, su, sv = eval_surface_d1(
            polish_S, params[1], params[2], rational=polish_rational)
        J = np.column_stack([cd, -su, -sv])[:, free]
        try:
            delta = np.linalg.lstsq(J, -G, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(delta)):
            break

        old_norm = float(np.linalg.norm(G))
        accepted = False
        step = 1.0
        for _ls in range(16):
            candidate = params.copy()
            candidate[free] = np.clip(
                candidate[free] + step * delta, 0.0, 1.0)
            candidate[bz.axis] = 0.0 if bz.side == 0 else 1.0
            pc = eval_curve(
                polish_C, candidate[0], rational=polish_rational)
            ps = eval_surface(
                polish_S, candidate[1], candidate[2],
                rational=polish_rational)
            if float(np.linalg.norm(pc - ps)) < old_norm:
                params = candidate
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break

    ok, G = _strict_csx_residual_ok(
        C, S, *params, rational, strict_tol)
    return (*map(float, params), G, bool(ok))


def _restrict_bezier_axis_ordered(ctrl, axis, a, b):
    """Restrict a tensor-product Bezier net to ordered interval a -> b."""
    a = float(np.clip(a, 0.0, 1.0))
    b = float(np.clip(b, 0.0, 1.0))
    reverse = b < a
    lo, hi = (b, a) if reverse else (a, b)
    out = np.asarray(ctrl, dtype=np.float64)
    if hi < 1.0:
        out, _ = de_casteljau_split_nd(out, axis=axis, t=hi)
    if lo > 0.0:
        rel = lo / max(hi, np.finfo(np.float64).tiny)
        _, out = de_casteljau_split_nd(out, axis=axis, t=rel)
    if reverse:
        out = np.flip(out, axis=axis).copy()
    return out


def _surface_affine_path_homogeneous(S_h, u0, v0, u1, v1):
    """Homogeneous Bezier curve S(u(lambda), v(lambda)), affine in lambda."""
    patch = _restrict_bezier_axis_ordered(S_h, 0, u0, u1)
    patch = _restrict_bezier_axis_ordered(patch, 1, v0, v1)
    p = patch.shape[0] - 1
    q = patch.shape[1] - 1
    degree = p + q
    out = np.zeros((degree + 1, patch.shape[-1]), dtype=np.float64)
    for i in range(p + 1):
        for j in range(q + 1):
            k = i + j
            out[k] += (comb(p, i) * comb(q, j) / comb(degree, k)) * patch[i, j]
    return out


# L52 slice 6a: the shared exact product. NOTE: the shared version computes
# the comb factor fully in longdouble (the old copy here rounded it through
# a Python float64 first) — identical on macOS (longdouble aliases float64),
# a last-ulp factor upgrade on true-80-bit platforms, absorbed by the
# certificate envelope.
_bernstein_product_1d = bernstein_product_1d


def _certify_affine_csx_overlap(C, S, a, b, rational):
    """Prove the affine endpoint correspondence is an exact overlap.

    The public overlap schema ships only endpoint ranges, so its implied
    correspondence is affine in one common parameter.  Certify that exact
    path by checking every coefficient of the homogeneous residual
    C_xyz*W_S - S_xyz*W_C, not a finite set of tolerance samples.
    """
    c_pts = _cartesian_controls_for_exactness(C, rational)
    s_pts = _cartesian_controls_for_exactness(S, rational)
    if c_pts is None or s_pts is None:
        return False
    origin = np.asarray(c_pts).reshape(-1, 3)[0]
    C_h, c_src = _center_homogeneous_for_exactness(
        C, rational, origin, with_source_scale=True)
    S_h, s_src = _center_homogeneous_for_exactness(
        S, rational, origin, with_source_scale=True)
    if C_h is None or S_h is None:
        return False
    # The restricted affine path can contain an exactly-zero coordinate
    # produced by cancellation of nonzero source controls (for example the
    # y=0 line through a plane whose y controls are -0.5 and 1.5).  A
    # roundoff bound based only on the already-cancelled product values is
    # then identically zero and rejects the arithmetic residue of the
    # restriction itself.  Retain a per-coordinate source-operation scale;
    # genuinely absent coordinates still keep a zero floor, so a one-sided
    # offset cannot hide here.
    c_xyz_scale = np.max(np.abs(C_h[:, :-1]), axis=0)
    s_xyz_scale = np.max(np.abs(S_h[..., :-1]), axis=(0, 1))
    c_weight_scale = float(np.max(np.abs(C_h[:, -1])))
    s_weight_scale = float(np.max(np.abs(S_h[..., -1])))
    source_product_scale = (
        c_xyz_scale * s_weight_scale
        + s_xyz_scale * c_weight_scale)
    # Centering term — twin of ccx's `_overlap_mapping_is_identity`.  The
    # two scales above come from the already-centered nets, so they measure
    # what survived the common-origin cancellation rather than what was
    # consumed by it; a world translation then destroys precision the
    # envelope never accounts for.  Priced with `_CCX_CENTERING_OPS` (one
    # subtraction), so it is ~eps at the model's own origin and grows only
    # with the precision actually lost.
    centering_product_scale = (
        np.max(np.asarray(c_src).reshape(-1, 3), axis=0) * s_weight_scale
        + np.max(np.asarray(s_src).reshape(-1, 3), axis=0) * c_weight_scale)
    ta, ua, va = (float(x) for x in a)
    tb, ub, vb = (float(x) for x in b)
    curve_h = _restrict_bezier_axis_ordered(C_h, 0, ta, tb)
    surf_h = _surface_affine_path_homogeneous(S_h, ua, va, ub, vb)
    left = _bernstein_product_1d(
        curve_h[:, :-1], surf_h[:, -1:])
    right = _bernstein_product_1d(
        surf_h[:, :-1], curve_h[:, -1:])
    residual = np.abs(left - right)
    # L52 slice 6b — the EXPLICIT envelope reconciliation (review §10 /
    # ledger): this certificate and ccx's `_overlap_mapping_is_identity`
    # certify the same class ("residual explainable by roundoff of
    # exactly-coincident sources") and now share ccx's derived two-term
    # structure instead of this module's former folded
    # `4096·n₁n₂·ε_f64·(|l|+|r|+src)` bound:
    # - the OPERATOR term scales with the eps of the dtype the products
    #   actually run in (longdouble; aliases float64 on macOS, where the
    #   computation genuinely is float64 — platform-consistent either way)
    #   and with n₁·n₂ accumulation terms;
    # - the SOURCE term scales with ε_float64 (sources are float64 on
    #   every platform) and n₁+n₂ restriction/degree-reduction steps; the
    #   8192 constant is the family ccx's float-built-subcurve fixture
    #   calibrated from below.
    # Net effect is a measured tightening at gate degrees (in-axis
    # acceptance boundary (3.0, 3.5]e-11 -> (2.6, 3.0]e-11 on the
    # cubic/bilinear probe; single-axis offsets were and remain rejected
    # at every magnitude by the per-coordinate source scale above) — the
    # SOUND direction: borderline candidates fall to the L42 typed
    # uncertified_overlap_span fallback instead of certifying.
    op_factor = (np.longdouble(64)
                 * np.longdouble(max(1, len(curve_h)))
                 * np.longdouble(max(1, len(surf_h)))
                 * np.longdouble(np.finfo(np.longdouble).eps))
    source_factor = (np.longdouble(8192)
                     * np.longdouble(max(1, len(curve_h) + len(surf_h)))
                     * np.longdouble(np.finfo(np.float64).eps))
    centering_factor = (np.longdouble(_CCX_CENTERING_OPS)
                        * np.longdouble(np.finfo(np.float64).eps))
    roundoff = (op_factor * (np.abs(left) + np.abs(right))
                + source_factor * np.asarray(
                    source_product_scale, dtype=np.longdouble)[None, :]
                + centering_factor * np.asarray(
                    centering_product_scale, dtype=np.longdouble)[None, :])
    return bool(np.all(np.isfinite(residual))
                and np.all(residual <= roundoff))


def _tolerance_csx_overlap_certificate(C, S, atol, rational, ptol_t,
                                        strict_context, n_samples=65,
                                        on_dense=None):
    """Modeling-tolerance curve-on-surface span classification.

    Endpoint evidence, domain-pinned projections and sampled interior
    witnesses classify within-atol spans. A sampled normal-side sign flip
    refuses promotion, but finite sampling cannot guarantee separation of
    every exact crossing. The legacy 'exact' label here describes the
    sampled residual tier, not an exact polynomial identity certificate.
    Exact-topology callers bypass this helper and use source identities.

    Returns overlap dictionaries (possibly multiple clipped spans), or
    None when no modeling span can be classified.
    """
    diag = float(np.linalg.norm(
        np.max(S.reshape(-1, S.shape[-1]), axis=0)
        - np.min(S.reshape(-1, S.shape[-1]), axis=0))) or 1.0
    tiny = max(4096.0 * float(np.finfo(np.float64).eps), 1e-12) * max(1.0, diag)
    edge_pad = 1e-9

    def _member(t, seed_uv):
        p = eval_curve(C, float(t), rational=rational)
        u, v, dist = _project_point_on_surface(
            p, S, seed_uv[0], seed_uv[1], atol, rational)
        u_cl = min(max(u, 0.0), 1.0)
        v_cl = min(max(v, 0.0), 1.0)
        if (u_cl, v_cl) != (u, v):
            s_pt = eval_surface(
                np.asarray(S, dtype=np.float64), u_cl, v_cl,
                rational=rational)
            dist = float(np.linalg.norm(p - s_pt))
        return p, u_cl, v_cl, float(dist)

    # --- (1) arming: coarse scan for ANY on-surface stretch -------------
    coarse = np.linspace(0.0, 1.0, 17)
    seed = (0.5, 0.5)
    hits = []
    for t in coarse:
        _p, u, v, d = _member(float(t), seed)
        if d <= atol:
            hits.append(float(t))
            seed = (u, v)
    if not hits:
        return None

    # A no-hit arming scan costs 17 projections; the dense pass below is
    # the expensive part and is billed separately by the caller (a flat
    # combined price tripped the §11.5 work-drift gate on nested cut-face
    # calls, which routinely arm-and-miss).
    if on_dense is not None:
        on_dense()

    # --- (2) dense membership + witnesses --------------------------------
    ts = np.linspace(0.0, 1.0, int(n_samples))
    inside = np.zeros(len(ts), dtype=bool)
    res = np.zeros(len(ts))
    uvs = np.zeros((len(ts), 2))
    signed = np.zeros(len(ts))
    seed = (0.5, 0.5)
    S64 = np.asarray(S, dtype=np.float64)
    for k, t in enumerate(ts):
        p, u, v, d = _member(float(t), seed)
        inside[k] = d <= atol
        res[k] = d
        uvs[k] = (u, v)
        if inside[k]:
            seed = (u, v)
        s_pt, s_du, s_dv = eval_surface_d1(S64, u, v, rational=rational)
        n_vec = np.cross(s_du, s_dv)
        n_len = float(np.linalg.norm(n_vec))
        signed[k] = (float(np.dot(p - s_pt, n_vec / n_len))
                     if n_len > 0.0 else 0.0)

    # contiguous in-band runs
    runs = []
    k = 0
    while k < len(ts):
        if inside[k]:
            j = k
            while j + 1 < len(ts) and inside[j + 1]:
                j += 1
            runs.append((k, j))
            k = j + 1
        else:
            k += 1

    def _refine(t_in, t_out, seed_uv):
        """Bisect the tolerance boundary between an in-band and an
        out-of-band sample."""
        for _ in range(40):
            tm = 0.5 * (t_in + t_out)
            _p, u, v, d = _member(tm, seed_uv)
            if d <= atol:
                t_in, seed_uv = tm, (u, v)
            else:
                t_out = tm
            if abs(t_out - t_in) <= max(1e-12, 0.01 * ptol_t):
                break
        return t_in, seed_uv

    overlaps = []
    for k0, k1 in runs:
        if k1 == k0:
            # Single-sample runs are the CORNER-CONTACT signature (the
            # CCX L47 guard's analog: a graze at one grid node whose
            # bisection fringe can exceed 4*ptol_t is still not band
            # evidence — measured: a ~4*atol stub at a shared corner
            # shipped as a junk overlap branch). A genuine span must be
            # in-band across at least one full grid interval.
            continue
        t_lo, t_hi = float(ts[k0]), float(ts[k1])
        if k0 > 0:
            t_lo, _ = _refine(t_lo, float(ts[k0 - 1]), tuple(uvs[k0]))
        if k1 < len(ts) - 1:
            t_hi, _ = _refine(t_hi, float(ts[k1 + 1]), tuple(uvs[k1]))
        if t_hi - t_lo < 4.0 * ptol_t:
            continue                    # too short to claim a 1-D overlap
        # (2b) domain pinning: each end must sit at the CURVE's domain end
        # or at the uv-domain edge of the surface (projected pin) — an
        # interior fade-out on both sides is the offset-twin signature and
        # is never promoted (the CCX L47 rule).
        def _pinned(t_end, outward):
            if t_end <= edge_pad or t_end >= 1.0 - edge_pad:
                return True
            _p, u, v, _d = _member(t_end, (0.5, 0.5))
            if min(u, 1.0 - u) <= 1e-6 or min(v, 1.0 - v) <= 1e-6:
                return True
            # The tolerance boundary and the uv-domain exit can COINCIDE
            # (measured on user data: d crosses atol at almost the same t
            # where the projected path leaves through u=1, so the
            # projection AT the refined boundary is still interior).
            # Probe one grid step OUTWARD: a domain-clipped span clamps to
            # an edge there; a genuine interior fade-out (the offset-twin
            # signature, which must stay refused) does not.
            t_probe = min(1.0, max(0.0, t_end + outward / (len(ts) - 1.0)))
            _p, u, v, _d = _member(t_probe, (u, v))
            return (min(u, 1.0 - u) <= 1e-6 or min(v, 1.0 - v) <= 1e-6)
        if not (_pinned(t_lo, -1.0) and _pinned(t_hi, +1.0)):
            continue
        # (3) flip guard on the gap samples INSIDE the span: root-like
        # samples (res <= tiny) are bridged; consecutive gap samples with
        # opposite normal-side signs = crossing structure -> refuse.
        # END-ADJACENT flips are exempt: a genuine touch AT a pinned span
        # end (real-world-inexact data sits above the roundoff floor, so
        # bridging cannot cover it) is the span's own endpoint root — the
        # theorem terminates the overlap there anyway. INTERIOR flips
        # still refuse (the never-merge-crossings invariant).
        idx = [k for k in range(k0, k1 + 1) if res[k] > tiny]
        flip = any(signed[a] * signed[b] < 0.0
                   for a, b in zip(idx, idx[1:])
                   if a != k0 and b != k1)
        if flip:
            continue
        span_res = float(res[k0:k1 + 1].max())
        cert = "exact" if span_res <= tiny else "tolerance"
        u_rng = (float(uvs[k0:k1 + 1, 0].min()),
                 float(uvs[k0:k1 + 1, 0].max()))
        v_rng = (float(uvs[k0:k1 + 1, 1].min()),
                 float(uvs[k0:k1 + 1, 1].max()))
        overlaps.append({
            "boundary_zeros": [],
            "overlap_endpoints": [],
            "t_range": (t_lo, t_hi),
            "u_range": u_rng,
            "v_range": v_rng,
            "certification": cert,
            "residual_max": span_res,
        })
    return overlaps or None


def _boundary_zero_to_tuv(bz: BoundaryZero,
                           t0: float, t1: float,
                           u0: float, u1: float,
                           v0: float, v1: float) -> tuple[float, float, float]:
    """Convert a BoundaryZero to global (t, u, v) parameters.

    For a 3D CSX net with axes [0=t, 1=u, 2=v]:
    - axis=0 fixed: t = t0 or t1, (param, param2) map to (u, v)
    - axis=1 fixed: u = u0 or u1, (param, param2) map to (t, v)
    - axis=2 fixed: v = v0 or v1, (param, param2) map to (t, u)

    The free axes for a given fixed axis are [i for i in (0,1,2) if i != axis],
    and param maps to the first free axis, param2 to the second.
    """
    ranges = [(t0, t1), (u0, u1), (v0, v1)]
    pt = [0.0, 0.0, 0.0]

    # Fixed axis
    pt[bz.axis] = ranges[bz.axis][0] if bz.side == 0 else ranges[bz.axis][1]

    # Free axes
    free = [i for i in range(3) if i != bz.axis]
    pt[free[0]] = ranges[free[0]][0] + bz.param * (ranges[free[0]][1] - ranges[free[0]][0])
    if bz.param2 is not None:
        pt[free[1]] = ranges[free[1]][0] + bz.param2 * (ranges[free[1]][1] - ranges[free[1]][0])
    else:
        # Fallback: midpoint (shouldn't happen for properly constructed BoundaryZeros)
        pt[free[1]] = 0.5 * (ranges[free[1]][0] + ranges[free[1]][1])

    return pt[0], pt[1], pt[2]


def _boundary_zero_from_tuv_like(bz: BoundaryZero, t, u, v) -> BoundaryZero:
    """Rebuild a boundary record with strict-polished free parameters."""
    values = [float(t), float(u), float(v)]
    free = [i for i in (0, 1, 2) if i != bz.axis]
    return BoundaryZero(
        axis=bz.axis, side=bz.side,
        param=values[free[0]], param2=values[free[1]])


# ---------------------------------------------------------------------------
# CSX boundary analysis: find zeros on the 6 faces of [0,1]^3
# ---------------------------------------------------------------------------

def _find_csx_boundary_zeros(
    F_3d, C, S, atol, ptol_t, ptol_u, ptol_v, rational,
    *, max_cells=50_000, max_results=4_096, diagnostics=None,
    tolerance_tier=False,
):
    """Find precise intersection points on the boundary faces of the CSX domain.

    The 6 faces of [0,1]^3 decompose into two types:

    Type 1 — Curve endpoints (t=0, t=1):
      Extract the 2D slice D(u,v) = ||C(t_fixed) - S(u,v)||^2 from the
      trivariate net. This is a point-on-surface problem. We use the
      2D classifier (same as CCX) to find zeros.

    Type 2 — Surface boundaries (u=0, u=1, v=0, v=1):
      Extract the surface boundary isocurve (a Bezier curve in 3D) and
      call bez_ccx(C, iso_curve) to find intersections.

    Returns ``(zeros, budget_exhausted, cells_processed)``.  When the flag
    is true, ``zeros`` is diagnostic-only: the caller must not use a partial
    face set to infer overlap topology or cut the Phase-2 domain.
    """
    zeros = []
    cells = DownCounter(max_cells)
    exhausted = False
    structural_incomplete = False
    boundary_obligations_complete = True
    F_3d_v = F_3d[..., np.newaxis]

    # --- Type 1: Curve endpoints (t=0, t=1) ---
    # Restrict the 3D net to t=0 and t=1 → 2D nets for point-on-surface
    for t_side in (0, 1):
        if cells.remaining <= 0:
            exhausted = True
            break
        # Charge the fixed face analysis/Newton probe itself.
        cells.spend(1)
        face_2d = bernstein_boundary_nd(F_3d_v, axis=0, side=t_side)[..., 0]

        # Use the 2D classifier (same as CCX) on this face
        _, Sw = extract_weights(S, rational=rational)
        sw_flat = Sw.ravel()
        # Weight for the point side is just C's weight at t=0 or t=1
        _, Cw = extract_weights(C, rational=rational)
        pw_point = np.array([Cw[0 if t_side == 0 else -1]])

        w_scale = _weight_max_product(pw_point, sw_flat)

        # Quick check: if min of 2D face > 0, no intersection at this endpoint
        if _check_min_of_net(face_2d, atol, w_scale):
            continue

        # Find precise zeros on the 2D face using the 1D solver
        # (finds zeros on edges of the face)
        from mmcore.numeric._bern_zero_1d import bernstein_zero_budget
        with bernstein_zero_budget(
            cells.remaining, max(0, max_results - len(zeros)),
        ) as zero_budget:
            face_zeros = _find_precise_bz_2d(face_2d, atol, w_scale)
        cells.spend(zero_budget.nodes)
        if zero_budget.exhausted:
            exhausted = True
            break
        for bz_2d in face_zeros:
            if not isinstance(bz_2d, BoundaryZero):
                exhausted = True
                break
            if len(zeros) >= max_results:
                exhausted = True
                break
            # Map 2D face zero to 3D BoundaryZero
            # bz_2d has axis (0 or 1 in the 2D face) and param
            # In the 3D net, axis=0 is t, so the face's axes map to u (1) and v (2)
            zeros.append(BoundaryZero(
                axis=0, side=t_side,
                param=bz_2d.param,
                param2=bz_2d.param2 if bz_2d.param2 is not None else (
                    float(bz_2d.side) if bz_2d.axis == 1 else None
                ),
            ))
        if exhausted:
            break

        # Also check if the 2D face has a minimum near zero in its interior
        # (not just on its edges). Use Newton on the point-on-surface problem.
        pt = eval_curve(C, float(t_side), rational=rational)
        # Simple Newton: project point onto surface
        from mmcore.numeric._bezier_common import eval_surface_d1
        u_s, v_s = 0.5, 0.5  # seed from center
        for _it in range(20):
            s_pt, s_du, s_dv = eval_surface_d1(S, u_s, v_s, rational=rational)
            G = s_pt - pt
            g2 = float(np.dot(G, G))
            if g2 < atol * atol:
                break
            J = np.column_stack([s_du, s_dv])
            JTJ = J.T @ J + 1e-12 * np.eye(2)
            JTG = J.T @ G
            try:
                delta = -np.linalg.solve(JTJ, JTG)
            except np.linalg.LinAlgError:
                break
            step = 1.0
            for _ls in range(8):
                un = max(0.0, min(1.0, u_s + step * delta[0]))
                vn = max(0.0, min(1.0, v_s + step * delta[1]))
                Gn = eval_surface(S, un, vn, rational=rational) - pt
                if float(np.dot(Gn, Gn)) <= g2:
                    u_s, v_s = un, vn
                    break
                step *= 0.5
            else:
                break
        G_final = eval_surface(S, u_s, v_s, rational=rational) - pt
        if not exhausted and float(np.linalg.norm(G_final)) < atol:
            # Found a point-on-surface intersection — add as BoundaryZero
            bz = BoundaryZero(axis=0, side=t_side, param=u_s, param2=v_s)
            # Check not duplicate
            is_dup = any(
                abs(z.param - u_s) < ptol_u and z.param2 is not None and abs(z.param2 - v_s) < ptol_v
                for z in zeros if z.axis == 0 and z.side == t_side
            )
            if not is_dup:
                if len(zeros) >= max_results:
                    exhausted = True
                else:
                    zeros.append(bz)

    if exhausted:
        return zeros, True, cells.processed

    # --- Type 2: Surface boundaries (u=0, u=1, v=0, v=1) ---
    # Extract boundary isocurves and run CCX
    for surf_axis, surf_side in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        # Extract isocurve: S at axis=surf_axis, side=surf_side
        # For S of shape (m+1, n+1, D): axis=0 → S[side, :, :], axis=1 → S[:, side, :]
        if surf_axis == 0:
            iso_curve = S[0 if surf_side == 0 else -1, :, :]  # shape (n+1, D)
        else:
            iso_curve = S[:, 0 if surf_side == 0 else -1, :]  # shape (m+1, D)

        # AABB pre-filter: skip isocurves whose bounding box doesn't overlap the curve's

        if rational:
            c_pts = C[:, :-1] / C[:, -1:]
            iso_pts = iso_curve[:, :-1] / iso_curve[:, -1:]
        else:
            c_pts = C
            iso_pts = iso_curve
        c_bb = np.array(aabb(c_pts))
        c_bb[0] -= atol; c_bb[1] += atol
        iso_bb = np.array(aabb(iso_pts))
        iso_bb[0] -= atol; iso_bb[1] += atol
        if not aabb_intersect(c_bb, iso_bb):
            continue

        # Run CCX between the original curve and this isocurve
        if cells.remaining <= 0:
            exhausted = True
            break
        # Keep the nested solver on the caller's contract. CAD-tolerance
        # CSX must not stop on an optional exact-root existence obligation;
        # its candidates still pass the boundary polishing below.
        ccx_result = bez_ccx_v4(
            C, iso_curve, atol=atol, rational=rational,
            max_cells=cells.remaining,
            max_results=max(0, max_results - len(zeros)),
            tolerance_tier=tolerance_tier,
        )
        ccx_cells = int(ccx_result.get("cells_processed", 0))
        ccx_cells = min(ccx_cells, cells.remaining)
        cells.spend(ccx_cells)
        if diagnostics is not None:
            for unresolved in ccx_result.get('unresolved_parameter_boxes', []):
                t_range = tuple(unresolved.get('u_range', (0., 1.)))
                edge_range = tuple(unresolved.get('v_range', (0., 1.)))
                fixed_range = (float(surf_side), float(surf_side))
                ranges = ((t_range, fixed_range, edge_range) if surf_axis == 0
                          else (t_range, edge_range, fixed_range))
                mapped = dict(zip(('t_range', 'u_range', 'v_range'), ranges))
                mapped['reason'] = unresolved.get('reason', 'boundary_root')
                if 'candidate' in unresolved:
                    t_candidate, edge_candidate = unresolved['candidate']
                    mapped['candidate'] = ((t_candidate, float(surf_side), edge_candidate)
                                           if surf_axis == 0 else
                                           (t_candidate, edge_candidate, float(surf_side)))
                for field in ('parameter_uniqueness_box', 'parameter_root_box'):
                    if field in unresolved:
                        curve_box, edge_box = unresolved[field]
                        mapped[field] = ((curve_box, fixed_range, edge_box)
                                         if surf_axis == 0 else
                                         (curve_box, edge_box, fixed_range))
                diagnostics.setdefault('unresolved_parameter_boxes', []).append(mapped)
        if (ccx_result.get("budget_exhausted", False)
                or not ccx_result.get("boundary_topology_complete", True)):
            if ccx_result.get('truncation_cause') == 'resolution':
                structural_incomplete = True
                boundary_obligations_complete &= bool(ccx_result.get('unresolved_obligations_complete', False))
                if diagnostics is not None:
                    diagnostics['truncation_cause'] = 'resolution'
                # Preserve individually polished witnesses and keep
                # analyzing independent surface edges within the budget.
            else:
                exhausted = True
                if diagnostics is not None:
                    diagnostics['truncation_cause'] = ccx_result.get('truncation_cause') or 'boundary'
                break

        for iso in ccx_result['isolated']:
            if len(zeros) >= max_results:
                exhausted = True
                break
            t_val = iso['u']  # parameter on C
            s_val = iso['v']  # parameter along the isocurve

            # Map isocurve parameter to surface (u, v)
            # The net axis in the 3D CSX problem:
            # surf_axis=0 → u is fixed, s_val runs along v → csx_axis=1
            # surf_axis=1 → v is fixed, s_val runs along u → csx_axis=2
            csx_axis = surf_axis + 1  # 0→1 (u-axis in CSX), 1→2 (v-axis in CSX)

            if surf_axis == 0:
                # u is fixed, s_val is v-parameter
                bz = BoundaryZero(axis=csx_axis, side=surf_side,
                                  param=t_val, param2=s_val)
            else:
                # v is fixed, s_val is u-parameter
                bz = BoundaryZero(axis=csx_axis, side=surf_side,
                                  param=t_val, param2=s_val)
            if (iso.get('parameter_root_certification') in
                    ('exact_linear_elimination', 'exact_curve_line_sturm')
                    and 'parameter_root_box' in iso):
                curve_box, edge_box = iso['parameter_root_box']
                fixed_box = (float(surf_side), float(surf_side))
                root_box = ((curve_box, fixed_box, edge_box) if surf_axis == 0
                            else (curve_box, edge_box, fixed_box))
                parameters = ((t_val, float(surf_side), s_val) if surf_axis == 0
                              else (t_val, s_val, float(surf_side)))
                bz._root_proof = {'parameter_root_box': root_box,
                                  'parameters': parameters, 'point': iso['point']}
            zeros.append(bz)

        if exhausted:
            break

        # Overlaps from CCX also produce boundary information
        for ovl in ccx_result['overlaps']:
            # Overlap endpoints are boundary zeros too
            ur = ovl.get('u_range', (0.0, 1.0))
            vr = ovl.get('v_range', (0.0, 1.0))
            csx_axis = surf_axis + 1
            for t_val, s_val in [(ur[0], vr[0]), (ur[1], vr[1])]:
                if len(zeros) >= max_results:
                    exhausted = True
                    break
                bz = BoundaryZero(axis=csx_axis, side=surf_side,
                                  param=t_val, param2=s_val)
                zeros.append(bz)
            if exhausted:
                break

        if exhausted:
            break

    if diagnostics is not None:
        diagnostics['unresolved_obligations_complete'] = bool(
            boundary_obligations_complete and not exhausted)
    return zeros, exhausted or structural_incomplete, cells.processed


def _project_point_on_surface(pt, S, u_seed, v_seed, atol, rational, max_it=20):
    """Project a 3D point onto a Bezier surface. Returns (u, v, dist)."""
    from mmcore.numeric._bezier_common import eval_surface_d1
    u, v = float(u_seed), float(v_seed)
    for _ in range(max_it):
        s_pt, s_du, s_dv = eval_surface_d1(S, u, v, rational=rational)
        G = s_pt - pt
        g2 = float(np.dot(G, G))
        if g2 < 1e-20:
            break
        J = np.column_stack([s_du, s_dv])
        A = J.T @ J + 1e-12 * np.eye(2)
        b = -J.T @ G
        try:
            delta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break
        step = 1.0
        for _ in range(8):
            un = max(0.0, min(1.0, u + step * delta[0]))
            vn = max(0.0, min(1.0, v + step * delta[1]))
            Gn = eval_surface(S, un, vn, rational=rational) - pt
            if float(np.dot(Gn, Gn)) <= g2:
                u, v = un, vn
                break
            step *= 0.5
        else:
            break
    dist = float(np.linalg.norm(eval_surface(S, u, v, rational=rational) - pt))
    return u, v, dist


def _collapsed_point_surface_membership(
    C, S, t, u, v, atol, rational, strict_context,
):
    """Certify one representative of a collapsed curve parameter fiber.

    General CSX roots and overlaps use the roundoff-scale certificate in
    :func:`_strict_csx_residual_ok`.  Collapsed fibers need one additional,
    deliberately narrower notion of numerical identity: independently
    rounded CAD parameterizations can describe the same singular point with
    a few ulps of the *source decimal data* between them (case 14's two
    eight-decimal cone nets differ by 1.36e-9 at the common apex).

    The fallback is capped by both the public geometric tolerance and a
    2e-10 relative local-coordinate envelope.  It therefore cannot turn an
    ordinary tolerance-valley point (for example a 5e-4 offset at
    ``atol=1e-3``) into a positive-dimensional solution.
    """
    point = eval_curve(C, float(t), rational=rational)
    u, v, _dist = _project_point_on_surface(
        point, S, u, v, atol, rational, max_it=64)
    strict, residual = _strict_csx_residual_ok(
        C, S, t, u, v, rational, strict_context)
    if strict:
        return True, float(u), float(v), residual

    if strict_context is None:
        return False, float(u), float(v), residual
    c_centered, s_centered, component_scale = strict_context[:3]
    point_local = eval_curve(c_centered, float(t), rational=True)
    surface_local = eval_surface(
        s_centered, float(u), float(v), rational=True)
    if not (np.all(np.isfinite(point_local)) and
            np.all(np.isfinite(surface_local))):
        return False, float(u), float(v), residual
    scale = np.maximum(1.0, np.asarray(component_scale, dtype=np.float64))
    identity_bound = np.minimum(float(atol), 2.0e-10 * scale)
    residual = point_local - surface_local
    return (bool(np.all(np.abs(residual) <= identity_bound)),
            float(u), float(v), residual)


def _check_csx_overlap_valley(C, S, boundary_zeros, atol, rational):
    """Valley check for CSX: step from each endpoint along the valley.

    For each pair of boundary zeros on different faces:
    1. From each boundary zero, step inward by 2*ptol_t along the curve
    2. Evaluate C(t_step) and project onto S to find (u, v)
    3. If dist < atol AND (u,v) moved away from the seed → valley continues
    4. If both sides confirm → overlap

    This does NOT assume the overlap is linear in parameter space.
    """
    if len(boundary_zeros) < 2:
        return None

    ptol_t, ptol_u, ptol_v = _compute_param_tols_csx(C, S, atol, rational)

    for i in range(len(boundary_zeros)):
        bz_a = boundary_zeros[i]
        for j in range(i + 1, len(boundary_zeros)):
            bz_b = boundary_zeros[j]
            if (bz_a.axis, bz_a.side) == (bz_b.axis, bz_b.side):
                continue

            pt_a = _boundary_zero_to_param_point(bz_a, 3)
            pt_b = _boundary_zero_to_param_point(bz_b, 3)
            t_a, u_a, v_a = pt_a[0], pt_a[1], pt_a[2]
            t_b, u_b, v_b = pt_b[0], pt_b[1], pt_b[2]

            # Check the t-separation is large enough for a meaningful overlap
            if abs(t_b - t_a) < ptol_t * 4:
                continue

            # Step inward from endpoint A
            step_t = 2 * ptol_t
            t_step_a = t_a + step_t if t_a < t_b else t_a - step_t
            t_step_a = max(0.0, min(1.0, t_step_a))

            pt_c_a = eval_curve(C, t_step_a, rational=rational)
            u_proj_a, v_proj_a, dist_a = _project_point_on_surface(
                pt_c_a, S, u_a, v_a, atol, rational,
            )
            moved_a = (abs(u_proj_a - u_a) > ptol_u or
                       abs(v_proj_a - v_a) > ptol_v)

            if dist_a >= atol or not moved_a:
                continue

            # Step inward from endpoint B
            t_step_b = t_b - step_t if t_b > t_a else t_b + step_t
            t_step_b = max(0.0, min(1.0, t_step_b))

            pt_c_b = eval_curve(C, t_step_b, rational=rational)
            u_proj_b, v_proj_b, dist_b = _project_point_on_surface(
                pt_c_b, S, u_b, v_b, atol, rational,
            )
            moved_b = (abs(u_proj_b - u_b) > ptol_u or
                       abs(v_proj_b - v_b) > ptol_v)

            if dist_b >= atol or not moved_b:
                continue

            # Both sides confirmed: valley continues inward from both endpoints
            return [bz_a, bz_b]

    return None


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------


def _restrict_net_t(F, t_lo, t_hi):
    """Restrict a trivariate sq-dist net to a sub-interval along axis 0 (t)."""
    Fv = F[..., np.newaxis]
    if t_lo > 0.0:
        _, Fv = de_casteljau_split_nd(Fv, axis=0, t=t_lo)
    if t_hi < 1.0:
        t_hi_rescaled = (t_hi - t_lo) / (1.0 - t_lo) if t_lo > 0.0 else t_hi
        Fv, _ = de_casteljau_split_nd(Fv, axis=0, t=t_hi_rescaled)
    return Fv[..., 0]


def _restrict_curve(C, t_lo, t_hi):
    """Restrict a Bezier curve to a sub-interval [t_lo, t_hi]."""
    if t_lo > 0.0:
        _, C = _subdivide_curve(C, t_lo)
    if t_hi < 1.0:
        t_hi_rescaled = (t_hi - t_lo) / (1.0 - t_lo) if t_lo > 0.0 else t_hi
        C, _ = _subdivide_curve(C, t_hi_rescaled)
    return C


def _split_intervals(cut, lo, hi, ptol):
    """Split [lo, hi] into up to 3 sub-intervals around cut ± ptol.

    Returns list of (sub_lo, sub_hi) intervals. The center interval
    [cut-ptol, cut+ptol] is the "excluded" zone (included in the list
    but marked by being the center).
    Returns all 3 intervals (left, center, right), filtering empties.
    """
    cut_lo = min(hi, max(cut - ptol, lo))
    cut_hi = max(lo, min(cut + ptol, hi))
    intervals = []
    if lo < cut_lo:
        intervals.append((lo, cut_lo))
    intervals.append((cut_lo, cut_hi))  # center (to be excluded)
    if cut_hi < hi:
        intervals.append((cut_hi, hi))
    return intervals


# L52 slice 7: shared implementation in _bezier_common (verbatim move).
_restrict_net_axis = restrict_net_axis


def _cutout_3d(F_cell, G_cell, seg_c, pw, sw, t0, t1, u0, u1, v0, v1, depth,
               t_cut, u_cut, v_cut, ptol_t, ptol_u, ptol_v, rational):
    """Cut out a ptol-neighborhood around (t_cut, u_cut, v_cut) from a cell.

    Splits along each axis at cut ± ptol, producing 3×3×3 = 27 boxes.
    The center box (ptol-neighborhood) is discarded. Remaining 26 are
    returned with restricted nets.
    """
    t_intervals = _split_intervals(t_cut, t0, t1, ptol_t)
    u_intervals = _split_intervals(u_cut, u0, u1, ptol_u)
    v_intervals = _split_intervals(v_cut, v0, v1, ptol_v)

    # Identify the clipped box by bounds. A domain-edge root leaves two
    # intervals on that axis, so list length cannot identify its center.
    t_center = (max(t_cut-ptol_t, t0), min(t_cut+ptol_t, t1))
    u_center = (max(u_cut-ptol_u, u0), min(u_cut+ptol_u, u1))
    v_center = (max(v_cut-ptol_v, v0), min(v_cut+ptol_v, v1))

    sub_cells = []

    for ti, (t_lo, t_hi) in enumerate(t_intervals):
        for ui, (u_lo, u_hi) in enumerate(u_intervals):
            for vi, (v_lo, v_hi) in enumerate(v_intervals):
                # Skip the center box
                if ((t_lo, t_hi) == t_center and (u_lo, u_hi) == u_center
                        and (v_lo, v_hi) == v_center):
                    continue

                # Skip empty intervals
                if t_hi <= t_lo or u_hi <= u_lo or v_hi <= v_lo:
                    continue

                # Restrict the nets to this sub-box
                F_sub = _restrict_net_axis(F_cell, 0, t_lo, t_hi, t0, t1)
                F_sub = _restrict_net_axis(F_sub, 1, u_lo, u_hi, u0, u1)
                F_sub = _restrict_net_axis(F_sub, 2, v_lo, v_hi, v0, v1)
                G_sub = _restrict_net_axis_v(G_cell, 0, t_lo, t_hi, t0, t1)
                G_sub = _restrict_net_axis_v(G_sub, 1, u_lo, u_hi, u0, u1)
                G_sub = _restrict_net_axis_v(G_sub, 2, v_lo, v_hi, v0, v1)

                # Restrict curve to t sub-interval
                C_sub = _restrict_curve(seg_c,
                    (t_lo - t0) / (t1 - t0),
                    (t_hi - t0) / (t1 - t0))
                pw_sub = C_sub[:, -1].copy() if rational else np.ones(C_sub.shape[0])

                # Restrict surface weights to u,v sub-interval
                if rational:
                    sw_sub = sw  # TODO: proper weight restriction
                else:
                    sw_sub = sw.copy()

                sub_cells.append((F_sub, G_sub, C_sub, pw_sub, sw_sub,
                                  t_lo, t_hi, u_lo, u_hi, v_lo, v_hi, depth + 1))

    return sub_cells


# Float64 ceiling on subdivision: a parameter interval narrower than the
# spacing between adjacent doubles cannot be halved again, so depth beyond
# ~3*53 (three axes) can never execute.  Any "safe" default above this is
# not safety, it is dead levels.
_CSX_DEPTH_FLOAT64_CEILING = 3 * 53

# Safety factor on the derived requirement (dimensionless).  The estimate
# below assumes the subdivision spends its halvings evenly on the axis that
# needs them; real cells wander, so allow half again.
_CSX_DEPTH_SAFETY = 1.5


def _derived_max_depth(ptol_t, ptol_u, ptol_v):
    """Subdivision ceiling implied by the parameter tolerances.

    Isolating a root to a per-axis tolerance takes log2(span/ptol) halvings
    on that axis, and `_phase2_isolated_search` subdivides ONE axis per
    level, so the requirement is the SUM over axes.  A constant cannot track
    this because ptol follows the caller's atol: measured on harness case 11,
    the requirement is 56.8 at atol=1e-3, 76.7 at 1e-5 and 82.7 at 2.5e-6,
    while the former constant was 64 -- which is exactly why depth
    truncations were never observed at 1e-3 (0 in 1,284 CSX calls) and
    appeared below it, aborting the SSX search with a `work_budget` reason
    it could not act on.
    """
    need = 0.0
    for ptol in (ptol_t, ptol_u, ptol_v):
        ptol = float(ptol)
        if not np.isfinite(ptol) or ptol <= 0.0:
            return _CSX_DEPTH_FLOAT64_CEILING
        need += math.log2(1.0 / min(ptol, 1.0)) if ptol < 1.0 else 0.0
    return int(min(_CSX_DEPTH_FLOAT64_CEILING,
                   max(16, math.ceil(_CSX_DEPTH_SAFETY * need))))


def _phase2_isolated_search(
    F_sub, G_sub, C_sub, S, C_orig, S_orig,
    t_lo, t_hi, atol, rational, ptol_t, ptol_u, ptol_v,
    known_points=None,
    max_depth=64, max_cells=50_000, max_results=4_096,
    exact_topology=False,
    unresolved_root_boxes=None,
    obligation_diagnostics=None,
    source_residual=None,
    source_exact_root=None,
    source_exact_root_work=0,
):
    """Phase 2: find isolated intersections via subdivision + Newton + cutout.

    When Newton finds an intersection, the ptol-neighborhood is cut out
    from the cell, producing sub-cells that are guaranteed root-free at
    that location. This prevents re-convergence and enables efficient
    discovery of multiple intersections.
    """
    from mmcore.numeric.intersection._sq_dist_classify import (
        _check_min_of_net, _check_lipschitz, _weight_max_product,
    )

    _, Pw = extract_weights(C_sub, rational=rational)
    _, Sw = extract_weights(S, rational=rational)
    strict_root_tol = _strict_csx_root_tol(C_orig, S_orig, rational)
    G_original = (_residual_vec_net(C_orig, S_orig, rational)
                  if source_residual is None else source_residual[0])
    source_scale = (_csx_residual_source_scale(C_orig, S_orig, rational)
                    if source_residual is None else None)
    coefficient_error = None if source_residual is None else source_residual[1]

    # Start with known points from Phase 1 so pre-Newton dedup can skip them
    isolated = known_points if known_points is not None else []
    initial_results = len(isolated)

    cells = 0
    exhausted = False
    # WHY the search stopped, when it did.  A bare boolean cannot
    # distinguish "ran out of the caller's cells" from "hit my own internal
    # depth ceiling", and the caller needs that distinction: the first is a
    # knob it can raise, the second is not.  See `bez_csx`'s
    # ``truncation_cause``.
    cause = None
    if obligation_diagnostics is not None:
        obligation_diagnostics['unresolved_obligations_complete'] = True

    stack = [(F_sub, G_sub, C_sub, Pw.copy(), Sw.copy(),
              t_lo, t_hi, 0.0, 1.0, 0.0, 1.0, 0)]
    terminal_boxes = []
    unresolved_start = len(unresolved_root_boxes) if unresolved_root_boxes is not None else 0

    def charge_union(amount):
        nonlocal cells
        if cells+amount > max_cells:
            return False
        cells += amount
        return True

    def exact_source_candidate(root, box, cell_box):
        # Endpoints are proposals, never tolerance-based replacements.
        # Only the source predicate can establish one, and it must belong
        # to both the certified uniqueness box and the owned search cell.
        from itertools import product
        choices = []
        for x, (lo, hi), (cell_lo, cell_hi) in zip(root, box, cell_box):
            values = [x]
            for endpoint in (0., 1.):
                if lo <= endpoint <= hi and cell_lo <= endpoint <= cell_hi and endpoint != x:
                    values.append(endpoint)
            choices.append(values)
        for candidate in product(*choices):
            if not charge_union(source_exact_root_work):
                return None, True
            if source_exact_root(candidate) is True:
                return candidate, False
        return None, False

    while stack:
        if cells >= max_cells:
            exhausted = True
            cause = cause or "cells"
            if obligation_diagnostics is not None:
                obligation_diagnostics['unresolved_obligations_complete'] = False
            break
        if len(isolated) - initial_results >= max_results:
            exhausted = True
            cause = cause or "results"
            if obligation_diagnostics is not None:
                obligation_diagnostics['unresolved_obligations_complete'] = False
            break
        cells += 1

        F_cell, G_cell, seg_c, pw, sw, t0, t1, u0, u1, v0, v1, depth = stack.pop()
        residual_error = _csx_residual_roundoff_bound(
            G_original, depth=(depth if source_residual is None else 6*(depth+1)),
            source_scale=source_scale)
        if coefficient_error is not None:
            residual_error = np.nextafter(residual_error+coefficient_error, np.inf)

        # Vector-residual sign prune: one component of G = C·w_S − S·w_C
        # whose coefficient hull excludes 0 proves no zero in the cell.
        # Cheapest and by far the most decisive prune near transversal
        # roots — run it first.
        if _residual_excludes_zero(G_cell, residual_error):
            continue

        # L60: the geometry-aligned exclusion — decisive on near-band
        # cells whose residual direction is diagonal to the axes (the
        # axis test converges only at clearance-scale cells there).
        if _residual_aligned_excludes_zero(G_cell, residual_error):
            continue

        # Quick prune: min-of-net
        sw_flat = sw.ravel()
        w_sc = _weight_max_product(pw, sw_flat)
        if not exact_topology and _check_min_of_net(F_cell, atol, w_sc):
            continue

        # A positive *floating* squared-distance coefficient is not a
        # strict zero-exclusion certificate. Gram-product cancellation
        # and subdivision can lift a genuine zero above zero. The
        # margined vector-residual exclusions above remain valid for
        # exact-root pruning even inside the modeling-tolerance band.

        # Resolution-floor prune: a cell that lies entirely inside the
        # ±2·ptol box of a known root cannot contain an additional zero
        # distinguishable from that root at the algorithm's parametric
        # resolution (ptol is the same indistinguishability scale the
        # cutout and the dedup already assume). Without this, the cells
        # hugging a cutout hole — where F genuinely dips toward zero — can
        # never certify positivity and grind down to ptol one level at a
        # time.
        _near_known = False
        for _e in isolated:
            if exact_topology:
                box = _e.get('_unique_box')
                enclosure = _e.get('parameter_root_box')
                if (box is not None and enclosure is not None
                        and _e.get('root_existence_certification')
                        and all(lo <= a <= b <= hi for (a,b),(lo,hi)
                                in zip(enclosure,box)) and all(
                        blo <= lo and hi <= bhi
                        for (lo, hi), (blo, bhi) in zip(
                            ((t0, t1), (u0, u1), (v0, v1)), box))):
                    _near_known = True
                    break
                continue
            if (t0 >= _e["t"] - 2.0 * ptol_t and t1 <= _e["t"] + 2.0 * ptol_t
                    and u0 >= _e["u"] - 2.0 * ptol_u and u1 <= _e["u"] + 2.0 * ptol_u
                    and v0 >= _e["v"] - 2.0 * ptol_v and v1 <= _e["v"] + 2.0 * ptol_v):
                _near_known = True
                break
            for free_axes in _e.get('_surface_fibers', ()):
                if all(axis in free_axes or (lo >= p-2.*tol and hi <= p+2.*tol)
                       for axis, (lo, hi, p, tol) in enumerate(zip(
                           (t0, u0, v0), (t1, u1, v1),
                           (_e['t'], _e['u'], _e['v']),
                           (ptol_t, ptol_u, ptol_v)))):
                    _near_known = True
                    break
            if _near_known:
                break
        if exact_topology and not _near_known and depth >= max_depth:
            ownership = _certify_csx_owned_cell(
                G_original,((t0,t1),(u0,u1),(v0,v1)),isolated,source_scale,
                charge=charge_union, coefficient_error=coefficient_error)
            if ownership is None:
                exhausted,cause = True,cause or 'cells'
                if obligation_diagnostics is not None:
                    obligation_diagnostics['unresolved_obligations_complete'] = False
                break
            _near_known = ownership
        if _near_known:
            continue

        # Lipschitz prune
        if not exact_topology and _check_lipschitz(F_cell, atol, w_sc):
            continue

        # Derivative sign check: if any partial derivative has all
        # coefficients of the same sign, no stationary point exists
        can_have_stationary = True
        if not exact_topology:
            Fv = F_cell[..., np.newaxis]
            for ax in range(3):
                dF = bernstein_partial_derivative_coeffs(Fv, axis=ax)
                coeffs = dF[..., 0]
                # A weak sign proves monotonicity only. Roots on a shared
                # subdivision face have dF=0 there and must survive in at
                # least one child. Exact topology uses the source-bounded
                # vector residual instead of unbounded floating F gates.
                if np.min(coeffs) > 0 or np.max(coeffs) < 0:
                    can_have_stationary = False
                    break
        if not can_have_stationary:
            continue

        # Resolve the full product domain: a short curve interval can
        # still have several well-separated surface-parameter preimages.
        t_span = t1 - t0
        u_span = u1 - u0
        v_span = v1 - v0
        if (not exact_topology and t_span <= ptol_t
                and u_span <= ptol_u and v_span <= ptol_v):
            # Micro-fragment: verify cell center is near a root, then
            # Newton-refine before reporting. Refinement collapses
            # near-duplicates (parametrically close to a known root but
            # geometrically just over atol) onto the exact known root,
            # so the dedup catches them.
            t_mid = 0.5 * (t0 + t1)
            u_mid = 0.5 * (u0 + u1)
            v_mid = 0.5 * (v0 + v1)
            pt_c = eval_curve(C_orig, t_mid, rational=rational)
            pt_s = eval_surface(S_orig, u_mid, v_mid, rational=rational)

            if float(np.linalg.norm(pt_c - pt_s)) < _csx_eff_atol(
                    C_orig, S_orig, t_mid, u_mid, v_mid, atol, rational):
                t_r, u_r, v_r, G_r, root_ok = _polish_csx_root(
                    C_orig, S_orig, t_mid, u_mid, v_mid,
                    rational, strict_root_tol)
                if root_ok:
                    pt_r = eval_curve(C_orig, t_r, rational=rational)
                    if not _is_duplicate(isolated,t_r, u_r, v_r, pt_r, atol,ptol_t,ptol_u,ptol_v):
                        isolated.append({
                            "t": float(t_r), "u": float(u_r), "v": float(v_r),
                            "point": pt_r, "_micro": True,
                        })
            continue

        # Try Newton from cell center
        t_mid = 0.5 * (t0 + t1)
        u_mid = 0.5 * (u0 + u1)
        v_mid = 0.5 * (v0 + v1)
        # Newton is BOUNDED to the cell: an unbounded trajectory draining
        # into an attractor outside the cell proves nothing and (before
        # this was bounded) forced a full subdivision cascade along the
        # boundary of every excluded root neighborhood.
        t_sol, u_sol, v_sol, G, last_step = newton_csx(
            C_orig, S_orig, t_mid, u_mid, v_mid, rational=rational,
            bounds=(t0, t1, u0, u1, v0, v1),
        )

        residual_ok = float(np.linalg.norm(G)) < _csx_eff_atol(
            C_orig, S_orig, t_sol, u_sol, v_sol, atol, rational)

        if residual_ok:
            t_sol, u_sol, v_sol, G, residual_ok = _polish_csx_root(
                C_orig, S_orig, t_sol, u_sol, v_sol,
                rational, strict_root_tol)

        # Exact ownership uses the closed representable cell. A fixed
        # slack can exceed a narrow cell's entire width and repeatedly
        # cut around an attractor which lies outside its product domain.
        _fp_slack = 0. if exact_topology else 1e-12
        in_cell = (
            t0 - _fp_slack <= t_sol <= t1 + _fp_slack
            and u0 - _fp_slack <= u_sol <= u1 + _fp_slack
            and v0 - _fp_slack <= v_sol <= v1 + _fp_slack
        )

        # An outside-cell Newton attractor is only a proposal. Publishing
        # it here bypasses the owned-box identity certificate below and
        # can fill max_results with ulp-shifting copies of one known root.
        # Its owning cell is responsible for existence and enumeration;
        # this cell continues searching its own parameter product.
        if residual_ok and (in_cell or not exact_topology):
            pt = eval_curve(C_orig, t_sol, rational=rational)
            root = (float(t_sol), float(u_sol), float(v_sol))
            cutout_center = root
            radii = (ptol_t, ptol_u, ptol_v)
            unique_box = None
            existence_certificate = None
            inherited_root_box = None
            matching = []
            if exact_topology and in_cell:
                radii = tuple(min(tol, span) for tol, span in zip(
                    radii, (t_span, u_span, v_span)))
                unique_box = _unique_csx_root_box(
                    G_original, root, radii, source_scale, coefficient_error)
                if unique_box is not None:
                    if source_residual is None:
                        existence_certificate = _root_existence_certificate(
                            C_orig, S_orig, root, unique_box, G_original, rational, source_scale)
                    else:
                        if source_exact_root is not None:
                            source_candidate, denied = exact_source_candidate(
                                root, unique_box, ((t0,t1),(u0,u1),(v0,v1)))
                            if denied:
                                exhausted, cause = True, 'cells'
                                if obligation_diagnostics is not None:
                                    obligation_diagnostics['unresolved_obligations_complete'] = False
                                break
                            if source_candidate is not None:
                                existence_certificate = 'exact_parameter_identity'
                                root = source_candidate
                                pt = eval_curve(C_orig,root[0],rational=rational)
                        if existence_certificate is None:
                            source_enclosure = _root_box_enclosure(
                                G_original, unique_box, coefficient_error=coefficient_error)
                            if source_enclosure is not None:
                                existence_certificate = 'source_krawczyk_inclusion'
                                inherited_root_box = source_enclosure
                    if existence_certificate is None:
                        for known in isolated:
                            root_enclosure = known.get('parameter_root_box')
                            if (root_enclosure is not None
                                    and known.get('root_existence_certification')
                                    and all(lo <= a <= b <= hi for (a, b), (lo, hi)
                                            in zip(root_enclosure, unique_box))):
                                existence_certificate = 'known_root_enclosure'
                                inherited_root_box = root_enclosure
                                break
                if unique_box is not None and existence_certificate is None:
                    ownership = _certify_csx_owned_cell(
                        G_original, unique_box, isolated, source_scale, charge=charge_union,
                        coefficient_error=coefficient_error)
                    if ownership is None:
                        exhausted, cause = True, 'cells'
                        if obligation_diagnostics is not None:
                            obligation_diagnostics['unresolved_obligations_complete'] = False
                        break
                    if ownership:
                        # The candidate need not establish another root:
                        # union injectivity proves this whole reservation
                        # contains no root besides the published source root.
                        stack.extend(_cutout_3d(
                            F_cell, G_cell, seg_c, pw, sw, t0, t1, u0, u1, v0, v1, depth,
                            *root, *radii, rational))
                        continue
                    # This region is reserved as unresolved, not consumed
                    # as a root. Search its complement within the budget.
                    if unresolved_root_boxes is not None:
                        unresolved_root_boxes.append({
                            't_range': unique_box[0], 'u_range': unique_box[1],
                            'v_range': unique_box[2], 'reason': 'root_existence',
                            'candidate': root, 'parameter_uniqueness_box': unique_box})
                    exhausted, cause = True, 'resolution'
                    stack.extend(_cutout_3d(
                        F_cell, G_cell, seg_c, pw, sw, t0, t1, u0, u1, v0, v1, depth,
                        *root, *radii, rational))
                    continue
                if unique_box is not None:
                    # Each box has an independently established root.
                    # Uniqueness on their union identifies those roots.
                    matching = [e for e in isolated if
                                (e.get('_unique_box') is None
                                 and (e['t'], e['u'], e['v']) == root) or
                                (e.get('parameter_root_box') is not None
                                 and e.get('root_existence_certification')
                                 and all(lo <= a <= b <= hi for (a, b), (lo, hi)
                                         in zip(e['parameter_root_box'], unique_box))) or
                                (e.get('_unique_box') is not None
                                 and _root_boxes_have_same_root(
                                     G_original, e['_unique_box'], unique_box, source_scale,
                                     coefficient_error))]
                    if matching:
                        matching[0]['_unique_box'] = unique_box
                        matching[0]['root_existence_certification'] = existence_certificate
                        matching[0]['_existence_established'] = True
                        for redundant in matching[1:]:
                            isolated[:] = [e for e in isolated if e is not redundant]
            if exact_topology and unique_box is not None:
                is_new = not matching
            elif exact_topology:
                # A Newton proposal without an ownership certificate
                # consumes no cell: subdivision below still searches its
                # entire domain. Publishing it here would later fabricate
                # an unresolved reservation even after that search proves
                # the domain empty or owned by another certified root.
                is_new = False
            else:
                is_new = not _is_duplicate(isolated, t_sol, u_sol, v_sol, pt,
                                           atol, ptol_t, ptol_u, ptol_v,
                                           exact_topology=exact_topology)
                for known in isolated:
                    free_axes = _matching_surface_fiber(
                        known, root, pt, atol, (ptol_t, ptol_u, ptol_v))
                    if free_axes:
                        radii = tuple(1. if axis in free_axes else radius
                                      for axis, radius in enumerate(radii))
                        break
            if is_new:
                # A genuine root. Record it regardless of which cell it
                # lies in — _is_duplicate protects against a double add
                # when the root's home cell converges to it later.
                entry = {
                    "t": root[0], "u": root[1], "v": root[2],
                    "point": pt,
                }
                if unique_box is not None:
                    entry['_unique_box'] = unique_box
                    entry['root_existence_certification'] = existence_certificate
                    entry['_existence_established'] = True
                    if inherited_root_box is not None:
                        entry['parameter_root_box'] = inherited_root_box
                    elif existence_certificate == 'exact_parameter_identity':
                        entry['parameter_root_box'] = tuple((t, t) for t in root)
                    else:
                        entry['parameter_root_box'] = unique_box
                isolated.append(entry)
            if in_cell and (not exact_topology or unique_box is not None):
                # Cut the root's ptol-neighborhood out of THIS cell so the
                # remaining sub-cells cannot re-converge to the same
                # attractor. Valid for newly found and already-known roots
                # alike — the cutout is what guarantees progress.
                sub_cells = _cutout_3d(
                    F_cell, G_cell, seg_c, pw, sw, t0, t1, u0, u1, v0, v1, depth,
                    *cutout_center,
                    *radii, rational,
                )
                stack.extend(sub_cells)
                continue
            # Root outside this cell: one Newton trajectory draining to an
            # outside attractor says NOTHING about other roots inside the
            # cell (regression: bez_ssx case 10 lost the branch segment
            # s in [0.57, 0.75] to this prune). Fall through to
            # subdivision; the strict-positivity / min-of-net / Lipschitz /
            # derivative-sign prunes terminate root-free sub-cells.
        # not residual_ok: Newton stalled or exhausted its iterations. A
        # single failed trajectory is likewise not evidence that the cell
        # is root-free — fall through to subdivision. (Near-miss "valleys"
        # where ||C-S|| dips below atol without a zero are terminated by
        # the strict-positivity certificate at the top of the loop, NOT by
        # trusting the stall: reporting or excising sub-atol valleys here
        # destroys sub-tolerance topology that the SSX layer depends on —
        # e.g. near-tangent loops whose paired crossings are connected by
        # a sub-atol valley.)


        if depth >= max_depth:
            # This cell survived every exclusion certificate and Newton did
            # not resolve it.  The depth guard is therefore a soft-budget
            # exhaustion, not evidence that the cell is root-free.
            #
            # It is ALSO not a resource shortfall: raising the caller's cell
            # allowance cannot buy more depth.  Reporting it as one made SSX
            # hard-stop the whole search and blame `work_budget` at 1.2%
            # ledger utilization (harness case 11 at atol<=1e-5).
            if exact_topology:
                # DFS can encounter this cell before the owning root is
                # discovered elsewhere. Retain the obligation and retry
                # against the final source-certified root census.
                terminal_boxes.append(((t0,t1),(u0,u1),(v0,v1)))
            else:
                exhausted = True
                cause = cause or "depth"
                if obligation_diagnostics is not None:
                    obligation_diagnostics['unresolved_obligations_complete'] = False
            continue

        # Subdivide along the axis with the largest span
        spans = [t1 - t0, u1 - u0, v1 - v0]
        axis = int(np.argmax(spans))
        if exact_topology:
            lo, hi = ((t0, t1), (u0, u1), (v0, v1))[axis]
            if not lo < 0.5 * (lo + hi) < hi:
                exhausted = True
                cause = cause or 'resolution'
                if obligation_diagnostics is not None:
                    obligation_diagnostics['unresolved_obligations_complete'] = False
                continue

        G_L, G_R = de_casteljau_split_nd(G_cell, axis=axis, t=0.5)
        if axis == 0:
            t_split = 0.5 * (t0 + t1)
            seg_c_L, seg_c_R = _subdivide_curve(seg_c)
            F_L, F_R = _subdivide_sq_dist_net(F_cell, axis=0)
            pw_L = seg_c_L[:, -1].copy() if rational else np.ones(seg_c_L.shape[0])
            pw_R = seg_c_R[:, -1].copy() if rational else np.ones(seg_c_R.shape[0])
            stack.append((F_L, G_L, seg_c_L, pw_L, sw.copy(), t0, t_split, u0, u1, v0, v1, depth+1))
            stack.append((F_R, G_R, seg_c_R, pw_R, sw.copy(), t_split, t1, u0, u1, v0, v1, depth+1))
        elif axis == 1:
            u_split = 0.5 * (u0 + u1)
            F_L, F_R = _subdivide_sq_dist_net(F_cell, axis=1)
            if rational:
                sw_L, sw_R = _subdivide_surface_weights(sw, axis=0)
            else:
                sw_L = sw.copy()
                sw_R = sw.copy()
            stack.append((F_L, G_L, seg_c.copy(), pw.copy(), sw_L, t0, t1, u0, u_split, v0, v1, depth+1))
            stack.append((F_R, G_R, seg_c.copy(), pw.copy(), sw_R, t0, t1, u_split, u1, v0, v1, depth+1))
        else:
            v_split = 0.5 * (v0 + v1)
            F_L, F_R = _subdivide_sq_dist_net(F_cell, axis=2)
            if rational:
                sw_L, sw_R = _subdivide_surface_weights(sw, axis=1)
            else:
                sw_L = sw.copy()
                sw_R = sw.copy()
            stack.append((F_L, G_L, seg_c.copy(), pw.copy(), sw_L, t0, t1, u0, u1, v0, v_split, depth+1))
            stack.append((F_R, G_R, seg_c.copy(), pw.copy(), sw_R, t0, t1, u0, u1, v_split, v1, depth+1))

    for terminal_box in terminal_boxes:
        ownership = _certify_csx_owned_cell(
            G_original,terminal_box,isolated,source_scale,charge=charge_union,
            coefficient_error=coefficient_error)
        if ownership is not True:
            exhausted = True
            cause = cause or ('cells' if ownership is None else 'depth')
            if obligation_diagnostics is not None:
                obligation_diagnostics['unresolved_obligations_complete'] = False
            if ownership is None:
                break

    if exact_topology and unresolved_root_boxes is not None:
        # DFS may establish the owning root after an earlier failed
        # existence proposal reserved its neighborhood. Retire only the
        # reservations covered by that later source root's union proof.
        pending = []
        reservations = unresolved_root_boxes[unresolved_start:]
        for index, reservation in enumerate(reservations):
            box = tuple(reservation[key] for key in ('t_range', 'u_range', 'v_range'))
            ownership = _certify_csx_owned_cell(
                G_original, box, isolated, source_scale, charge=charge_union,
                coefficient_error=coefficient_error)
            if ownership is not True:
                pending.append(reservation)
            if ownership is None:
                pending.extend(reservations[index+1:])
                exhausted, cause = True, 'cells'
                if obligation_diagnostics is not None:
                    obligation_diagnostics['unresolved_obligations_complete'] = False
                break
        unresolved_root_boxes[unresolved_start:] = pending
        if (reservations and not pending and cause == 'resolution'
                and obligation_diagnostics is not None
                and obligation_diagnostics.get('unresolved_obligations_complete', False)):
            exhausted, cause = False, None

    # Return only NEW results (exclude the pre-loaded known points)
    return isolated, exhausted, cells, cause


# ---------------------------------------------------------------------------
# Main algorithm: two-phase architecture
# ---------------------------------------------------------------------------

from mmcore.numeric.intersection._sq_dist_classify import (
    _check_min_of_net, _weight_max_product,
    _find_precise_boundary_zeros as _find_precise_bz_2d,
)


# L42: separate Phase-2 allowance once a valley-confirmed overlap pair has
# failed the exact affine certificate — an exact continuum cannot be turned
# into a sound public overlap by subdivision, so it must not burn the whole
# caller allowance failing to (CCX's non-affine fallback uses the same
# pattern; CSX cells price ~2x a curve pair's, hence the 2x number).
_NON_AFFINE_OVERLAP_FALLBACK_CELLS = 4_000


def _source_residual_csx(C, S, source_residual, source_exact_root, source_exact_root_work, atol, rational,
                         max_depth, max_cells, max_results):
    """Search the entire source residual cube; C/S supply proposals only.

    No rounded-geometry boundary census or overlap witness can remove any
    part of this product domain. Source coefficient enclosures are carried
    into every exclusion, injectivity, ownership, and existence decision.
    """
    try:
        net, error = source_residual
        # Native Bernstein split functions require writable buffers; the
        # caller may legitimately share an immutable cached source net.
        net = np.array(net, dtype=float, copy=True, order='C')
        error = np.broadcast_to(np.asarray(error, dtype=float), (3,)).copy()
    except (ValueError, TypeError) as exc:
        raise ValueError('source_residual must be (net, scalar or 3-vector error)') from exc
    if (net.ndim != 4 or net.shape[-1] != 3 or any(n < 1 for n in net.shape)
            or not np.all(np.isfinite(net)) or not np.all(np.isfinite(error))
            or np.any(error < 0.)):
        raise ValueError('source_residual requires a finite (nt,nu,nv,3) net and nonnegative error')
    if source_exact_root is not None and not callable(source_exact_root):
        raise ValueError('source_exact_root must be callable')
    if int(source_exact_root_work) != source_exact_root_work or source_exact_root_work < 0:
        raise ValueError('source_exact_root_work must be a nonnegative integer')
    if max_cells <= 0 or max_results <= 0:
        return dict(isolated=[], overlaps=[], parameter_fibers=[],
                    budget_exhausted=True, cells_processed=0,
                    truncation_cause='preflight', boundary_topology_complete=False,
                    unresolved_obligations_complete=False)
    radii = _compute_param_tols_csx(C, S, atol, rational)
    if max_depth is None:
        max_depth = _derived_max_depth(*radii)
    unresolved, diagnostics = [], {}
    # Squared distance is not an exclusion or topology premise on this
    # route. A constant placeholder supports the shared subdivision stack.
    isolated, exhausted, used, cause = _phase2_isolated_search(
        np.zeros((1, 1, 1)), net, C, S, C, S, 0., 1., atol, rational, *radii,
        max_depth=int(max_depth), max_cells=int(max_cells), max_results=int(max_results),
        exact_topology=True, unresolved_root_boxes=unresolved,
        obligation_diagnostics=diagnostics, source_residual=(net, error),
        source_exact_root=source_exact_root, source_exact_root_work=int(source_exact_root_work))
    for entry in isolated:
        entry.pop('_existence_established', None)
        box = entry.pop('_unique_box', None)
        if box is not None:
            entry['parameter_uniqueness_box'] = box
        entry['point_certification'] = 'proposal_geometry'
    result = dict(isolated=isolated, overlaps=[], parameter_fibers=[],
                  budget_exhausted=bool(exhausted), cells_processed=int(used),
                  truncation_cause=cause if exhausted else None,
                  boundary_topology_complete=not (exhausted or unresolved),
                  unresolved_obligations_complete=bool(diagnostics.get(
                      'unresolved_obligations_complete', False)))
    if unresolved:
        result['unresolved_parameter_boxes'] = unresolved
    return result


def bez_csx(
    C,
    S,
    atol=1e-3,
    rational=True,
    # Three independently subdivided parameters can need more than 50
    # levels before every span reaches its computed resolution (an exact
    # corner root in the legacy SSX overlap case needs 53).  Cell and result
    # budgets remain the termination backstops.
    max_depth=None,
    max_cells=100_000,
    max_results=4_096,
    tolerance_tier=True,
    source_residual=None,
    source_exact_root=None,
    source_exact_root_work=0,
) -> dict:
    """Bezier curve-surface intersection via two-phase architecture.

    Phase 1: Find boundary intersections and overlaps on the initial patch.
             These can ONLY exist at the boundaries of the original objects.
    Phase 2: Search for isolated intersections on the remaining curve intervals
             via subdivision + Newton. No boundary analysis or overlap checks.

    Parameters
    ----------
    C : ndarray
        Control polygon of the Bezier curve. Shape ``(p+1, D)``.
    S : ndarray
        Control net of the Bezier surface. Shape ``(m+1, n+1, D)``.
    atol : float
        Geometric tolerance for intersection detection.
    rational : bool
        Whether the control nets are homogeneous (last column = weight).
    max_depth : int
        Maximum subdivision depth for Phase 2.
    max_cells : int
        Maximum total cells shared by boundary analysis, nested CCX calls,
        and Phase 2 (safety limit).
    max_results : int
        Maximum isolated roots materialized before returning a partial
        result. Positive-dimensional sets must be classified separately;
        this cap prevents an unrecognized set from turning dedup quadratic.
    tolerance_tier : bool
        Allow residual-certified curve/surface overlaps within ``atol``.
        This modeling mode may merge isolated events at its parameter
        resolution; complete status is not an exhaustive exact-root census.
        Set False when a caller needs only exact intersection topology;
        exact affine overlaps and isolated-root search remain enabled.
    source_residual : tuple, optional
        Expert source certificate ``(net, coefficient_error)``. The finite
        ``(nt,nu,nv,3)`` Bernstein net is over this call's unit parameter
        cube; a scalar or three-vector nonnegative error bounds every
        coefficient of the original exact source residual. This selects a
        full product search using only these source enclosures for proofs.
        C/S supply numerical proposals; returned XYZ requires validation
        against the original source geometry by the caller.
    source_exact_root : callable, optional
        With source_residual, an externally charged exact-source identity
        predicate on a candidate (t,u,v). Only True establishes a root;
        False/None leave existence to source interval Krawczyk inclusion.
    source_exact_root_work : int
        Work units prepaid from this call's shared allowance before each
        exact-source callback invocation. The callback must not charge
        those units again. Zero supports an externally prepaid callback.

    Returns
    -------
    dict
        In addition to intersections, the result reports
        ``budget_exhausted``, ``cells_processed``, and
        ``boundary_topology_complete``.
    """
    C = np.asarray(C, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)

    if source_residual is not None:
        return _source_residual_csx(
            C, S, source_residual, source_exact_root, source_exact_root_work, atol, rational,
            max_depth, max_cells, max_results)

    if rational:
        c_pts = C[:, :-1] / C[:, -1:]
        s_pts = S.reshape((-1,4))[:,:-1]/  S.reshape((-1,4))[:,-1:]
    else:
        c_pts = C
        s_pts = S.reshape((-1,3))

    if not aabb_intersect(aabb_offset( aabb(np.ascontiguousarray(c_pts)),atol/2), aabb_offset( aabb(np.ascontiguousarray(s_pts)),atol/2)):
        return {"isolated": [], "overlaps": [], "parameter_fibers": [],
                "budget_exhausted": False, "cells_processed": 0,
                "boundary_topology_complete": True}

    if int(max_cells) <= 0:
        return {"isolated": [], "overlaps": [], "parameter_fibers": [],
                "budget_exhausted": True, "cells_processed": 0,
                "truncation_cause": "preflight",
                "boundary_topology_complete": False}

    if (not tolerance_tier and int(max_results) > 0
            and not geometry_collapsed(c_pts)):
        # Bound exact coefficient construction before converting an
        # arbitrary-degree source net to Fraction arithmetic. This is an
        # operation-derived preflight against the existing shared cap.
        from mmcore.numeric.intersection._exact_univariate import coefficient_build_work
        planar_construction_work = coefficient_build_work(len(C))
        if S.shape[:2] == (2, 2) and planar_construction_work > max_cells:
            return {'isolated': [], 'overlaps': [], 'parameter_fibers': [],
                    'budget_exhausted': True, 'boundary_topology_complete': False,
                    'cells_processed': 0, 'truncation_cause': 'preflight',
                    'required_coefficient_work': planar_construction_work}
        from mmcore.numeric.intersection.csx._planar_overlap import (
            exact_planar_bilinear_overlap,
        )
        planar_overlaps = exact_planar_bilinear_overlap(
            C, S, rational=rational)
        if planar_overlaps:
            # The helper proves the full curve is inside a convex planar
            # patch, or exactly clips a line against that convex patch.
            # Its complement is therefore empty of additional roots.
            return {
                'isolated': [], 'overlaps': planar_overlaps,
                'parameter_fibers': [], 'budget_exhausted': False,
                'cells_processed': planar_construction_work, 'boundary_topology_complete': True,
            }
        from mmcore.numeric.intersection.csx._planar_roots import (
            exact_planar_bilinear_roots,
        )
        planar_roots = exact_planar_bilinear_roots(
            C, S, rational=rational, max_cells=int(max_cells),
            max_results=int(max_results), atol=atol)
        if planar_roots is not None:
            return planar_roots

    strict_root_tol = _strict_csx_root_tol(C, S, rational)

    # A rational Bezier curve whose Euclidean control points all coincide
    # is identically one point even when its homogeneous weights vary.  If
    # that point lies on the surface, the CSX solution set is
    # positive-dimensional in ``t``: every curve parameter is a solution.
    # Treating the fiber as isolated roots made Phase 2 cut out one tiny
    # t-neighbourhood at a time (case 14: 16,385 pseudo-roots), then paid an
    # O(n^2) duplicate scan.  Classify the fiber explicitly and solve only
    # the genuine point-on-surface problem.
    if geometry_collapsed(c_pts):
        from mmcore.numeric._bez_closest_point import bez_surface_closest_points

        query = eval_curve(C, 0.5, rational=rational)
        closest_cells = max(1, min(int(max_cells), 2_000))
        closest_stats = {}
        entities = bez_surface_closest_points(
            S, query, atol=atol, rational=rational,
            max_cells=closest_cells, stats=closest_stats,
        )
        closest_used = int(closest_stats.get("cells_processed", 0))
        closest_capped = bool(closest_stats.get("budget_exhausted", False))
        fibers = []
        surface_collapsed = geometry_collapsed(s_pts)
        topology_incomplete = bool(closest_capped)
        result_limit = max(0, int(max_results))

        def _append_fiber(candidate):
            nonlocal topology_incomplete
            if len(fibers) >= result_limit:
                topology_incomplete = True
                return False
            fibers.append(candidate)
            return True

        for entity in ([] if closest_capped else entities):
            if float(entity.get("distance", np.inf)) > atol:
                continue
            kind = entity.get("kind", "min")
            if kind == "degenerate_surface":
                if surface_collapsed:
                    exact_member, u_member, v_member, _ = (
                        _collapsed_point_surface_membership(
                            C, S, 0.5, 0.5, 0.5, atol, rational,
                            strict_root_tol))
                    if not exact_member:
                        continue
                    if not _append_fiber({
                        "t_range": (0.0, 1.0),
                        "u_range": (0.0, 1.0),
                        "v_range": (0.0, 1.0),
                        "point": np.asarray(query, dtype=np.float64),
                        "surface_kind": kind,
                    }):
                        break
                else:
                    # A closest-point band can classify a nearly
                    # equidistant patch without proving the exact
                    # point-on-surface parameter region.  A representative
                    # (u,v) would understate its dimension, so surface an
                    # honest partial result instead.
                    topology_incomplete = True
                continue
            if kind == "degenerate_curve" or "u" not in entity or "v" not in entity:
                topology_incomplete = True
                continue
            u_entity = float(entity.get("u", 0.5))
            v_entity = float(entity.get("v", 0.5))
            exact_member, u_entity, v_entity, _ = (
                _collapsed_point_surface_membership(
                    C, S, 0.5, u_entity, v_entity, atol, rational,
                    strict_root_tol))
            if not exact_member:
                continue
            if not _append_fiber({
                "t_range": (0.0, 1.0),
                "u": u_entity,
                "v": v_entity,
                "point": np.asarray(query, dtype=np.float64),
                "surface_kind": kind,
            }):
                break
        return {"isolated": [], "overlaps": [],
                "parameter_fibers": fibers,
                "budget_exhausted": bool(topology_incomplete),
                "cells_processed": closest_used,
                "boundary_topology_complete": not topology_incomplete}



    F = curve_surface_distance_squared_net_homog(C, S, rational=rational)
    G_full = _residual_vec_net(C, S, rational=rational)

    _, Pw = extract_weights(C, rational=rational)
    _, Sw = extract_weights(S, rational=rational)

    C_orig = C
    S_orig = S

    ptol_t, ptol_u, ptol_v = _compute_param_tols_csx(C, S, atol, rational)
    if max_depth is None:
        # Derived from the tolerances actually in force — see
        # `_derived_max_depth`.  An explicit value still wins, so callers
        # that must bound the work can.
        max_depth = _derived_max_depth(ptol_t, ptol_u, ptol_v)
    max_depth = int(max_depth)

    isolated = []
    overlaps = []
    budget_exhausted = False
    # WHY this call truncated, when it did (schema addition
    # 2026-07-26).  `budget_exhausted` alone cannot distinguish a
    # resource shortfall the caller can fix by raising a knob from an
    # internal structural ceiling it cannot.  SSX escalated the latter
    # to a global hard stop blaming `work_budget`; see `_run_csx`.
    #   'cells' | 'results' | 'depth' | 'boundary' | 'preflight'
    truncation_cause = None
    cells = DownCounter(max_cells)

    # ===================================================================
    # PHASE 1: Boundary analysis + overlap detection (initial patch only)
    # ===================================================================
    boundary_result_cap = min(max(0, int(max_results)), 128)
    boundary_diagnostics = {}
    (csx_boundary_zeros, boundary_exhausted,
     boundary_cells) = _find_csx_boundary_zeros(
        F, C, S, atol, ptol_t, ptol_u, ptol_v, rational,
        max_cells=cells.remaining, max_results=boundary_result_cap,
        diagnostics=boundary_diagnostics,
        tolerance_tier=tolerance_tier,
    )
    cells.spend(boundary_cells)
    if boundary_exhausted:
        # Partial face topology cannot safely drive OVERLAP classification
        # (the valley pairing below needs the complete boundary-zero set,
        # so it is gated on `not boundary_exhausted`), but each zero found
        # so far is polished through the strict per-root certificate and is
        # individually sound — keep them (ledger L51: discarding returned
        # `{isolated: [], budget_exhausted: True}` with certified roots in
        # hand and ~no Phase-2 budget left to re-find them; CCX keeps its
        # validated hits in the same situation). The public flag still
        # records that the topology is incomplete.
        budget_exhausted = True
        truncation_cause = truncation_cause or boundary_diagnostics.get(
            'truncation_cause', 'boundary')
    elif not all(isinstance(bz, BoundaryZero) for bz in csx_boundary_zeros):
        budget_exhausted = True
        boundary_exhausted = True
        truncation_cause = truncation_cause or "boundary"
        csx_boundary_zeros = []

    t_exclude = []  # t-intervals to cut from the curve
    exact_topology = not tolerance_tier
    source_scale = _csx_residual_source_scale(C, S, rational)
    search_unresolved_boxes = list(boundary_diagnostics.get('unresolved_parameter_boxes', []))
    obligations_complete = (not boundary_exhausted or bool(
        boundary_diagnostics.get('unresolved_obligations_complete', False)))

    # Accept boundary zeros. Each candidate is refined via Newton:
    # boundary analysis can return near-miss candidates (e.g., CCX
    # finds where curves are closest, even if the closest distance
    # is above atol) that lie near a real root just inside the
    # boundary. Newton refinement converges to the real root if one
    # exists nearby; otherwise the candidate is rejected.
    for bz in csx_boundary_zeros:
        if len(isolated) >= max_results:
            budget_exhausted = True
            obligations_complete = False
            truncation_cause = truncation_cause or 'results'
            break
        t_bz, u_bz, v_bz = _boundary_zero_to_tuv(bz, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        boundary_proof = getattr(bz, '_root_proof', None) if exact_topology else None
        if boundary_proof is not None:
            t_r, u_r, v_r = boundary_proof['parameters']
            root_ok = True
        else:
            t_r, u_r, v_r, G_r, root_ok = _polish_csx_boundary_root(
                C, S, bz, t_bz, u_bz, v_bz, rational,
                strict_root_tol)

        if root_ok:
            pt_r = (boundary_proof['point'] if boundary_proof is not None else
                    eval_curve(C, t_r, rational=rational))
            if not _is_duplicate(isolated,t_r, u_r, v_r, pt_r, atol,ptol_t,ptol_u,ptol_v,
                                 exact_topology=exact_topology):
                entry = {
                    "t": float(t_r), "u": float(u_r), "v": float(v_r),
                    "point": pt_r,
                }
                if exact_topology:
                    if boundary_proof is not None:
                        entry['_boundary_root_proof'] = boundary_proof
                        entry['parameter_root_box'] = boundary_proof['parameter_root_box']
                        entry['root_existence_certification'] = 'exact_boundary_curve_intersection'
                        entry['_existence_established'] = True
                    box = _unique_csx_root_box(
                        G_full, (t_r, u_r, v_r),
                        (ptol_t, ptol_u, ptol_v), source_scale)
                    if box is not None:
                        proof_inside = (boundary_proof is not None and all(
                            lo <= a <= b <= hi for (a, b), (lo, hi)
                            in zip(boundary_proof['parameter_root_box'], box)))
                        certificate = ('exact_boundary_curve_intersection' if proof_inside
                                       else _root_existence_certificate(
                                           C, S, (t_r, u_r, v_r), box, G_full, rational, source_scale))
                        if certificate is not None:
                            entry['_unique_box'] = box
                            entry['root_existence_certification'] = certificate
                            entry['_existence_established'] = True
                            if 'parameter_root_box' not in entry:
                                entry['parameter_root_box'] = (tuple((t, t) for t in (t_r, u_r, v_r))
                                                               if certificate == 'exact_parameter_identity' else box)
                isolated.append(entry)
                # An isolated root resolves its (t,u,v) neighborhood,
                # not this t slab across the entire surface. Phase 2
                # receives it as a known point and cuts its product box.

    if tolerance_tier:
        # Recognize free target parameters before Phase2 fills its result
        # budget with one geometric hit. Other surface preimages at the
        # same curve parameter remain in the full-product search.
        probe_work = max(1, (2*S.size+127)//128)
        for entry in isolated:
            if cells.remaining < probe_work:
                break
            cells.spend(probe_work)
            fibers = _surface_fibers_at_point(
                S, entry['u'], entry['v'], entry['point'], atol, rational)
            if fibers:
                entry['_surface_fibers'] = fibers

    # Valley check for overlap — only on a COMPLETE boundary-zero set: a
    # truncated set could pair the wrong endpoints into a false overlap
    # claim (L51: the per-root keeps above are sound, this pairing is not).
    non_affine_overlap_span = None
    _valley_pair_seen = False
    if (not boundary_exhausted or exact_topology) and len(csx_boundary_zeros) >= 2:
        overlap_pair = _check_csx_overlap_valley(C, S, csx_boundary_zeros, atol, rational)
        _valley_pair_seen = overlap_pair is not None
        if overlap_pair is not None:
            bz_a, bz_b = overlap_pair
            t_a, u_a, v_a = _boundary_zero_to_tuv(bz_a, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
            t_b, u_b, v_b = _boundary_zero_to_tuv(bz_b, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
            t_a, u_a, v_a, _Ga, root_a = _polish_csx_boundary_root(
                C, S, bz_a, t_a, u_a, v_a, rational,
                strict_root_tol)
            t_b, u_b, v_b, _Gb, root_b = _polish_csx_boundary_root(
                C, S, bz_b, t_b, u_b, v_b, rational,
                strict_root_tol)
            endpoint_a = (t_a, u_a, v_a)
            endpoint_b = (t_b, u_b, v_b)
            affine_overlap = (root_a and root_b and _certify_affine_csx_overlap(
                    C, S, endpoint_a, endpoint_b, rational))
            if affine_overlap and exact_topology:
                proof_cost = len(C)+sum(n-1 for n in S.shape[:-1])
                if cells.remaining < proof_cost:
                    affine_overlap = False
                    budget_exhausted = True
                    truncation_cause = 'cells'
                else:
                    cells.spend(proof_cost)
                    affine_overlap = _affine_residual_is_zero(
                        C, S, endpoint_a, endpoint_b, rational)
            if affine_overlap:
                bz_a_strict = _boundary_zero_from_tuv_like(
                    bz_a, *endpoint_a)
                bz_b_strict = _boundary_zero_from_tuv_like(
                    bz_b, *endpoint_b)
                t_lo_ovl, t_hi_ovl = min(t_a, t_b), max(t_a, t_b)
                overlaps.append({
                    "boundary_zeros": [
                        (bz_a.axis, bz_a.side), (bz_b.axis, bz_b.side)],
                    "overlap_endpoints": [bz_a_strict, bz_b_strict],
                    "t_range": (t_lo_ovl, t_hi_ovl),
                    "u_range": (min(u_a, u_b), max(u_a, u_b)),
                    "v_range": (min(v_a, v_b), max(v_a, v_b)),
                    "certification": "exact",
                })
                t_exclude.append(
                    (t_lo_ovl - (0. if exact_topology else ptol_t),
                     t_hi_ovl + (0. if exact_topology else ptol_t)))
                # Remove isolated points inside the certified overlap.
                padding = 0. if exact_topology else atol
                isolated = [iso for iso in isolated
                            if not (t_lo_ovl - padding
                                    <= iso["t"] <= t_hi_ovl + padding)]
            elif root_a and root_b:
                # Ledger L42: a valley-confirmed pair whose affine identity
                # cannot be certified is either a curved-UV EXACT overlap
                # (the public endpoint-range schema cannot represent it) or
                # a broad near-tangent tolerance valley.  Neither exclusion
                # nor tolerance merging is sound here (sub-atol valleys
                # carry real topology), and an unrestricted Phase-2 walk of
                # an exact continuum floods thousands of ptol-lattice
                # "isolated" roots REPORTED COMPLETE — silent wrong
                # topology (measured: 1,679 roots @33,685 cells on a
                # parabola lying on a bilinear patch).  Port of the CCX
                # non-affine fallback: Phase 2 keeps running so genuine
                # isolated roots in a benign valley are still found, but
                # under a separate bounded allowance, and a continuum
                # signature in the outcome marks the topology incomplete.
                non_affine_overlap_span = (min(t_a, t_b), max(t_a, t_b))

    if exact_topology and overlaps:
        error = _csx_residual_roundoff_bound(G_full, source_scale=source_scale)
        if not _jacobian_is_injective(G_full, (1, 2), error):
            # One affine correspondence does not exhaust the parameter
            # product of a folded surface. Keep it as a geometric witness
            # while refusing to discharge the unknown UV complement.
            isolated, unresolved_roots = _certify_csx_root_existence(
                isolated, C, S, G_full, (ptol_t, ptol_u, ptol_v), rational, source_scale)
            for entry in isolated:
                box = entry.pop('_unique_box', None)
                if box is not None:
                    entry['parameter_uniqueness_box'] = box
            return {
                'isolated': isolated, 'overlaps': overlaps,
                'parameter_fibers': [], 'budget_exhausted': True,
                'cells_processed': int(cells.processed),
                'truncation_cause': 'resolution',
                'boundary_topology_complete': False,
                'unresolved_parameter_boxes': [
                    {'t_range': tuple(o['t_range']),
                     'u_range': (0.0, 1.0), 'v_range': (0.0, 1.0),
                     'reason': 'surface_preimage_uniqueness'}
                    for o in overlaps] + unresolved_roots,
            }

    # L59 (USER DECISION: theorem-first tier): certify tolerance/exact
    # overlap spans whenever cheap endpoint evidence arms — a curve t-end
    # lying ON the surface (an axis-0 boundary zero) or a valley pair —
    # INDEPENDENTLY of the exact-affine identity.  The real-data class has
    # only ONE boundary zero (the other span end pins on the uv-domain
    # edge in the projected sense: measured 6.7e-4 clearance from the edge
    # line — no exact 3-D root exists there for the boundary phase to
    # find).  Certified spans bypass the Phase-2 grind entirely.
    # Arming: a curve-end zero on-surface, a valley pair, OR a pair with
    # NO boundary zeros at all (measured on user data: a coincident
    # stretch that enters AND exits through patch edges with sub-atol
    # clearance produces zero exact boundary roots — the certificate must
    # still get its 17-sample look; transversal SSX-nested calls always
    # carry boundary zeros, so their cost profile is untouched).
    if (tolerance_tier and not boundary_exhausted and not overlaps
            and (_valley_pair_seen
                 or not csx_boundary_zeros
                 or any(bz.axis == 0 for bz in csx_boundary_zeros))):
        # Split pricing (§11.5 drift-gate lesson): the 17-projection
        # arming scan is billed always; the dense pass (65 witnesses +
        # refines) only when the scan HITS — nested cut-face calls
        # routinely arm-and-miss and must not pay the full tier.
        _tier_price = 17
        _dense_price = 65 + 80
        if cells.remaining > _tier_price + _dense_price:
            cells.spend(_tier_price)
            _tol_overlaps = _tolerance_csx_overlap_certificate(
                C, S, atol, rational, ptol_t, strict_root_tol,
                on_dense=lambda: cells.spend(
                    min(_dense_price, max(0, cells.remaining))))
            if _tol_overlaps:
                for _o in _tol_overlaps:
                    overlaps.append(_o)
                    _t_lo, _t_hi = _o["t_range"]
                    t_exclude.append((_t_lo - ptol_t, _t_hi + ptol_t))
                    isolated = [iso for iso in isolated
                                if not (_t_lo - atol
                                        <= iso["t"] <= _t_hi + atol)]
                # the span(s) are certified: the L42 uncertified-span
                # fallback no longer applies to them.
                non_affine_overlap_span = None

    # ===================================================================
    # PHASE 2: Isolated intersection search on remaining curve intervals
    # ===================================================================
    t_intervals = _compute_remaining_intervals(t_exclude, 0.0, 1.0)

    # L42 fallback allowance: with an uncertified valley pair on record,
    # Phase 2 cannot turn the rejected candidate into a sound public
    # overlap, so it must not burn the caller's whole allowance failing to.
    fallback_cells_remaining = (
        cells.tier(_NON_AFFINE_OVERLAP_FALLBACK_CELLS)
        if non_affine_overlap_span is not None else None)

    def _interval_hits_span(a, b):
        if non_affine_overlap_span is None:
            return False
        s_lo, s_hi = non_affine_overlap_span
        return not (b < s_lo - ptol_t or a > s_hi + ptol_t)

    # L28 attribution: retirement of the structural overlap reason at the
    # SSX level is sound only if the truncation touched NOTHING outside the
    # uncertified span — record whether any disjoint interval was truncated
    # or skipped.
    non_span_truncation = False
    t_intervals = list(t_intervals)

    for _ii, (t_lo, t_hi) in enumerate(t_intervals):
        if t_hi <= t_lo:
            continue

        # Restrict nets and curve to sub-interval
        F_sub = _restrict_net_t(F, t_lo, t_hi)
        G_sub = _restrict_net_axis_v(G_full, 0, t_lo, t_hi, 0.0, 1.0)
        C_sub = _restrict_curve(C, t_lo, t_hi)

        # Quick check: all positive → no intersection
        from mmcore.numeric.intersection._sq_dist_classify import (
            _check_min_of_net, _weight_max_product,
        )

        if rational:
            cb_pts = C_sub[:, :-1] / C_sub[:, -1:]

        else:
            cb_pts = C_sub

        if not aabb_intersect(aabb(np.ascontiguousarray(cb_pts)), aabb(np.ascontiguousarray(s_pts))):
            continue

        _, Pw_sub = extract_weights(C_sub, rational=rational)
        w_sc = _weight_max_product(Pw_sub, Sw.ravel())
        if not exact_topology and _check_min_of_net(F_sub, atol, w_sc):
            continue

        # L60: whole-interval geometry-aligned exclusion — a near-band
        # interval whose residual direction rotates slowly clears here in
        # one test instead of thousands of phase-2 cells.
        residual_error = _csx_residual_roundoff_bound(
            G_full, source_scale=source_scale)
        if (_residual_excludes_zero(G_sub, residual_error)
                or _residual_aligned_excludes_zero(G_sub, residual_error)):
            cells.spend(1)
            continue

        # Search for isolated intersections
        _rest_has_disjoint = (
            non_affine_overlap_span is not None
            and any(not _interval_hits_span(a, b)
                    for a, b in t_intervals[_ii:] if (b - a) >= ptol_t))
        if cells.remaining <= 0 or len(isolated) >= max_results:
            budget_exhausted = True
            truncation_cause = truncation_cause or (
                "cells" if cells.remaining <= 0 else "results")
            if _rest_has_disjoint:
                non_span_truncation = True
            break
        if (fallback_cells_remaining is not None
                and fallback_cells_remaining <= 0):
            budget_exhausted = True
            truncation_cause = truncation_cause or "cells"
            if _rest_has_disjoint:
                non_span_truncation = True
            break
        phase2_cell_limit = cells.remaining
        if fallback_cells_remaining is not None:
            phase2_cell_limit = min(
                phase2_cell_limit, fallback_cells_remaining)
        phase2_obligations = {}
        (_phase2_iso, _phase2_exhausted, _cells_used,
         _phase2_cause) = _phase2_isolated_search(
            F_sub, G_sub, C_sub, S, C_orig, S_orig,
            t_lo, t_hi, atol, rational, ptol_t, ptol_u, ptol_v,
            known_points=isolated,
            max_depth=max_depth, max_cells=phase2_cell_limit,
            max_results=max_results - len(isolated),
            exact_topology=exact_topology,
            unresolved_root_boxes=search_unresolved_boxes,
            obligation_diagnostics=phase2_obligations,
        )
        obligations_complete &= bool(phase2_obligations.get('unresolved_obligations_complete', False))
        if _phase2_cause is not None and truncation_cause is None:
            truncation_cause = _phase2_cause
        cells.spend(_cells_used)
        if fallback_cells_remaining is not None:
            fallback_cells_remaining -= _cells_used
        if _phase2_exhausted and not _interval_hits_span(t_lo, t_hi):
            non_span_truncation = True
        budget_exhausted = budget_exhausted or _phase2_exhausted



    # L42 outcome test: with an uncertified valley pair, either the bounded
    # fallback ran dry, or the surviving roots inside the span form a
    # ptol-lattice chain (the continuum signature — a short exact overlap
    # can complete under the fallback cap and would otherwise revive the
    # completeness lie).  Both directions only ADD flags (conservative);
    # a benign near-tangent valley with finitely many transversal roots
    # completes under the cap, forms no chain, and stays complete.
    overlap_topology_incomplete = False
    if non_affine_overlap_span is not None:
        span_lo, span_hi = non_affine_overlap_span
        in_span = sorted(
            iso["t"] for iso in isolated
            if span_lo - ptol_t <= iso["t"] <= span_hi + ptol_t)
        chain = (len(in_span) > 12
                 and all(b - a <= 4.0 * ptol_t
                         for a, b in zip(in_span, in_span[1:])))
        if budget_exhausted or chain:
            budget_exhausted = True
            overlap_topology_incomplete = True
    if (tolerance_tier and not overlap_topology_incomplete
            and len(isolated) >= 3):
        # L52 slice 10a (A2's confirmed lead): a SHORT exact overlap span
        # survives only by DOMAIN CLIPPING (a polynomial curve exactly on
        # the surface over an open sub-interval is on it everywhere, so
        # only the uv-domain edge can end a genuine span), and a clipped
        # span shorter than the >12-root chain bar produced a few lattice
        # roots reported COMPLETE (measured: 4.2·ptol_t corner-clipped
        # continuum → 3 isolated roots, complete=True, no span). Detect
        # the lattice: a run of ≥3 roots with every consecutive gap
        # ≤ 4·ptol_t whose GAP MIDPOINTS all pass the STRICT residual
        # certificate — exact continuums verify at roundoff scale, while
        # sub-atol-valley root pairs FAIL strict (their valley floors sit
        # far above roundoff), so distinct zeros connected by sub-atol
        # sampled valleys are refused; the public modeling tier does not
        # promise separation of every sub-resolution exact event.
        entries = sorted(
            ((float(e["t"]), float(e["u"]), float(e["v"]))
             for e in isolated), key=lambda x: x[0])
        runs, run = [], [entries[0]]
        for prev, cur in zip(entries, entries[1:]):
            if cur[0] - prev[0] <= 4.0 * ptol_t:
                run.append(cur)
            else:
                runs.append(run)
                run = [cur]
        runs.append(run)

        def _run_is_continuum(run_entries):
            for (ta, ua, va), (tb, ub, vb) in zip(run_entries,
                                                  run_entries[1:]):
                tm = 0.5 * (ta + tb)
                pm = eval_curve(C, tm, rational=rational)
                um, vm, _dist = _project_point_on_surface(
                    pm, S, 0.5 * (ua + ub), 0.5 * (va + vb),
                    atol, rational)
                ok, _res = _strict_csx_residual_ok(
                    C, S, tm, um, vm, rational, strict_root_tol)
                if not ok:
                    return False
            return True

        # Many surface parameters can represent one geometric hit on C.
        # A collapsed target isoline is not an overlap interval on C; its
        # representatives must span more than the curve's CAD resolution
        # before they can arm this continuum fallback.
        verified = [r for r in runs if len(r) >= 3
                    and r[-1][0]-r[0][0] > ptol_t
                    and _run_is_continuum(r)]
        if verified:
            largest = max(verified, key=len)
            non_affine_overlap_span = (largest[0][0], largest[-1][0])
            overlap_topology_incomplete = True
            # more than one verified continuum: structure ALSO lives
            # outside the exported span — the caller must not retire the
            # incompleteness after representing the span alone.
            non_span_truncation = len(verified) > 1

    obligations_complete &= (truncation_cause in (None, 'resolution')
                             and not overlap_topology_incomplete)
    unresolved_roots = list(search_unresolved_boxes)
    if exact_topology:
        isolated, final_unresolved_roots = _certify_csx_root_existence(
            isolated, C, S, G_full, (ptol_t, ptol_u, ptol_v), rational, source_scale)
        unresolved_roots.extend(final_unresolved_roots)
        if unresolved_roots:
            budget_exhausted = True
            truncation_cause = 'resolution'
    for entry in isolated:
        entry.pop('_boundary_root_proof', None)
        entry.pop('_surface_fibers', None)
        entry.pop('_existence_established', None)
        box = entry.pop('_unique_box', None)
        if box is not None:
            entry['parameter_uniqueness_box'] = box
    result = {"isolated": isolated, "overlaps": overlaps,
              "parameter_fibers": [],
              "budget_exhausted": bool(budget_exhausted),
              "cells_processed": int(cells.processed),
              # None unless budget_exhausted; see the declaration above.
              "truncation_cause": (truncation_cause if budget_exhausted
                                   else None),
              "boundary_topology_complete": not (
                  boundary_exhausted or overlap_topology_incomplete or unresolved_roots)}
    result['unresolved_obligations_complete'] = bool(obligations_complete)
    if unresolved_roots:
        result['unresolved_parameter_boxes'] = unresolved_roots
    if overlap_topology_incomplete:
        if exact_topology:
            # A valley proposal plus a truncated zero search proves no
            # dimension. Multiple close roots and true overlaps both
            # reach this fallback, so report only the unresolved product.
            result['truncation_cause'] = 'resolution'
            result.setdefault('unresolved_parameter_boxes', []).append({
                't_range': tuple(non_affine_overlap_span),
                'u_range': (0.0, 1.0), 'v_range': (0.0, 1.0),
                'reason': 'uncertified_relation',
            })
            return result
        # Typed L42 outcome for the SSX consumer (ledger L28): the span
        # names WHERE the uncertifiable positive-dimensional structure
        # lives, and `non_span_truncation` says whether anything OUTSIDE
        # it was also truncated (in which case the caller must not retire
        # its incompleteness even after representing the span as a region).
        result["uncertified_overlap_span"] = (
            float(non_affine_overlap_span[0]),
            float(non_affine_overlap_span[1]))
        result["non_span_truncation"] = bool(non_span_truncation)
    return result


def _is_duplicate(isolated, t,u,v, pt, atol, ptol_t,ptol_u,ptol_v,
                  exact_topology=False):
    for entry in isolated:
        if exact_topology:
            if (entry['t'], entry['u'], entry['v']) == (t, u, v):
                return True
            continue
        if _matching_surface_fiber(entry, (t, u, v), pt, atol,
                                   (ptol_t, ptol_u, ptol_v)):
            return True
        # Two independently refined representatives of one root can each
        # sit up to one parametric tolerance from it (boundary vs Phase-2
        # is the common pairing), hence a 2*ptol comparison.  Retain the
        # xyz guard so nearby parameters on distinct sheets are never
        # merged merely because their boxes touch.
        if (abs(entry['t'] - t) <= 2.0 * ptol_t
                and abs(entry['u'] - u) <= 2.0 * ptol_u
                and abs(entry['v'] - v) <= 2.0 * ptol_v
                and np.linalg.norm(pt - entry['point']) < atol):


            return True
    return False
