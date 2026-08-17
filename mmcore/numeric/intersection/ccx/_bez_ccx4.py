"""Bezier curve-curve intersection using squared-distance Bernstein net classification.

This module implements a subdivision-based CCX algorithm that uses the
squared-distance net ``||C1(u) - C2(v)||^2`` in Bernstein form to classify
cells as NO_INTERSECTION, UNIQUE_ISOLATED, OVERLAP, or INDETERMINATE,
avoiding explicit Jacobian-rank analysis.
"""
from __future__ import annotations

import numpy as np

from mmcore.numeric.aabb import aabb_offset
from mmcore.numeric.aabb import aabb_intersect,aabb
from mmcore.numeric._work_budget import DownCounter
from mmcore.numeric.bern import de_casteljau_split_nd
from mmcore.numeric.bern_sq_dist import curve_curve_squared_net_homog
from mmcore.numeric._bezier_common import (
    extract_weights, eval_curve, eval_curve_d1, newton_ccx,
    bernstein_product_1d, subdivide_curve, subdivide_sq_dist_net,
    restrict_net_axis,
)
from mmcore.numeric.intersection._sq_dist_classify import (
    classify_sq_dist_net,
    NO_INTERSECTION,
    UNIQUE_ISOLATED,
    OVERLAP,
    INDETERMINATE,
    BOUNDARY_ZERO,
    BoundaryZero, _boundary_zero_to_param_point,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# L52 slice 7: shared implementations in _bezier_common (verbatim moves).
_subdivide_curve = subdivide_curve
_subdivide_sq_dist_net = subdivide_sq_dist_net

def _subdivide_sq_dist_net_2d(F, u=0.5,v=0.5):
    """Subdivide the scalar sq-dist Bernstein net along *axis*.

    ``de_casteljau_split_nd`` requires a trailing value dimension, so we
    temporarily add one and squeeze it back off.
    """
    Fv = F[..., np.newaxis]

    left_v, right_v = de_casteljau_split_nd(Fv, axis=1, t=u)
    left_u_left_v,right_u_left_v=de_casteljau_split_nd(left_v, axis=0, t=v)
    left_u_right_v,right_u_right_v =de_casteljau_split_nd(right_v, axis=0, t=v)
    return left_u_left_v[..., 0], left_u_right_v[..., 0],right_u_right_v[..., 0],right_u_left_v[..., 0]

def _boundary_zero_to_uv(bz: BoundaryZero, u0: float, u1: float, v0: float, v1: float) -> tuple[float, float]:
    """Convert a BoundaryZero from the classifier into global (u, v) parameters.

    For a 2D CCX net:
    - axis=0, side=0 → u=u0, v = v0 + bz.param * (v1 - v0)
    - axis=0, side=1 → u=u1, v = v0 + bz.param * (v1 - v0)
    - axis=1, side=0 → v=v0, u = u0 + bz.param * (u1 - u0)
    - axis=1, side=1 → v=v1, u = u0 + bz.param * (u1 - u0)
    """
    if bz.axis == 0:
        u = u0 if bz.side == 0 else u1
        v = v0 + bz.param * (v1 - v0)
    else:
        v = v0 if bz.side == 0 else v1
        u = u0 + bz.param * (u1 - u0)
    return u, v


def _compute_param_tols(C1, C2, atol, rational):
    """Compute parametric tolerances for both curves using curve geometry.

    Returns (tol_u, tol_v) — the maximum parameter perturbation in each
    curve that corresponds to geometric deviation <= atol.
    """
    from mmcore.geom._nurbs_param_tol import bez_curve_param_tolerance
    tol_u = bez_curve_param_tolerance(C1, tol=atol, rational=rational)
    tol_v = bez_curve_param_tolerance(C2, tol=atol, rational=rational)

    return float(tol_u), float(tol_v)


# Roundings in the common-origin centering of one coordinate: one division,
# one multiply, one subtract, one final normalization.  Used as the eps
# multiplier for the centering envelope in BOTH this module and csx's twin.
_CENTERING_OPS = 4.0


def _cartesian_curve_controls_for_exactness(C, rational):
    C = np.asarray(C, dtype=np.float64)
    if rational:
        weights = C[:, -1:]
        if (not np.all(np.isfinite(weights))
                or np.any(weights == 0.0)):
            return None
        points = C[:, :-1] / weights
    else:
        points = C
    if not np.all(np.isfinite(points)):
        return None
    return points


def _center_curve_homogeneous_for_exactness(C, rational, origin,
                                            with_source_scale=False):
    """Return a stably normalized homogeneous curve translated by origin.

    With ``with_source_scale`` this also returns, per coordinate entry, the
    magnitude of the TWO OPERANDS the translation subtracts, carried through
    the same final normalization as the values themselves.

    Why any caller needs that: the net is divided by ``scale`` BEFORE it is
    translated, so a coordinate whose exact translated value is zero is
    computed as ``fl(x/scale) - fl(origin*fl(w/scale))`` — pure cancellation.
    The error of that difference is governed by the operands, not by the
    cancelled result, so an envelope built only from the (already cancelled)
    output is arbitrarily far below the noise it must absorb.  Callers whose
    test direction is UNSOUND on an underestimate must add this term; see
    `_vector_residual_hull_excludes_zero`.  This mirrors the
    `source_product_scale` term that `_overlap_mapping_is_identity` and
    csx's `_certify_affine_csx_overlap` already carry for their own
    post-centering restriction step.
    """
    C = np.asarray(C, dtype=np.float64)
    if rational:
        H = C.copy()
    else:
        H = np.concatenate(
            [C, np.ones((len(C), 1), dtype=np.float64)], axis=1)
    scale = float(np.max(np.abs(H)))
    if not np.isfinite(scale) or scale <= 0.0:
        return (None, None) if with_source_scale else None
    H /= scale
    shift = np.asarray(origin, dtype=np.float64) * H[:, -1:]
    # Operand magnitudes BEFORE the cancelling subtract.  A coordinate that
    # is genuinely absent from the sources (all controls zero and no shift
    # on that axis) keeps a zero floor here, so a real one-sided offset can
    # never be hidden by this term.
    operand_mag = np.abs(H[:, :-1]) + np.abs(shift)
    H[:, :-1] -= shift
    local_scale = float(np.max(np.abs(H)))
    if not np.isfinite(local_scale) or local_scale <= 0.0:
        return (None, None) if with_source_scale else None
    out = np.ascontiguousarray(H / local_scale)
    if not with_source_scale:
        return out
    return out, np.ascontiguousarray(operand_mag / local_scale)


def _ccx_exactness_context(C1, C2, rational):
    """Common-origin context for translation-invariant equality tests.

    ``component_scale`` is the per-axis control scale that
    `_eval_curve_scaled_components` DIVIDES by, so an axis whose entire
    post-centering content is the centering's own cancellation noise must
    report scale 0 (= "this coordinate is absent"), exactly as it does when
    the sources are literally zero on that axis.  Reporting the noise itself
    would amplify it to O(1) on division and make the strict predicate
    reject genuine roots — measured on the user's boundary-coincidence pair,
    where a z-planar pair keeps scale 0 in world coordinates but acquired a
    ~1e-17 "scale" once centered, after which every root was refused.

    The absence test is the coordinate's own roundoff envelope, built from
    the operand magnitudes the centering subtraction consumed and expressed
    in the same dehomogenized units as ``component_scale``.  Its factor is
    the ``32 * degree_factor`` family `_strict_residual_ok` already uses, so
    an axis is dropped only when it is unresolvable rather than merely
    small: on this fixture the noise sits ~4 orders below the bar.
    """
    points1 = _cartesian_curve_controls_for_exactness(C1, rational)
    points2 = _cartesian_curve_controls_for_exactness(C2, rational)
    if points1 is None or points2 is None:
        return None
    origin = points1[0].copy()
    H1, src1 = _center_curve_homogeneous_for_exactness(
        C1, rational, origin, with_source_scale=True)
    H2, src2 = _center_curve_homogeneous_for_exactness(
        C2, rational, origin, with_source_scale=True)
    if H1 is None or H2 is None:
        return None
    local1 = H1[:, :-1] / H1[:, -1:]
    local2 = H2[:, :-1] / H2[:, -1:]
    component_scale = np.maximum(
        np.max(np.abs(local1), axis=0),
        np.max(np.abs(local2), axis=0))
    w1 = np.abs(H1[:, -1:])
    w2 = np.abs(H2[:, -1:])
    if np.any(w1 == 0.0) or np.any(w2 == 0.0):
        return None
    # Factor derivation (review 2026-07-26 — the FACTOR has an operand too).
    # This envelope bounds ONE subtraction, not a degree-n accumulation:
    #   a = fl(x/scale)              1 rounding
    #   b = fl(origin * fl(w/scale)) 2 roundings
    #   d = fl(a - b)                1 rounding   (+1 for the /local_scale)
    # so |err| <= ~4*eps*(|a|+|b|) = _CENTERING_OPS*eps*operand_mag.  The
    # first version used the `32*degree_factor` family borrowed from
    # `_strict_residual_ok`, but that factor prices a de Casteljau chain and
    # is ~256x too large here (measured); on an ACCEPT path that surplus is
    # exactly the false-root window, so it is not a free safety margin.
    centering_noise = (_CENTERING_OPS * np.finfo(np.float64).eps
                       * np.maximum(np.max(src1 / w1, axis=0),
                                    np.max(src2 / w2, axis=0)))
    component_scale = np.where(
        component_scale > centering_noise, component_scale, 0.0)
    # The envelope travels WITH the context: a consumer that treats a
    # zero-scale axis as "must be exactly zero" needs it, because after a
    # centering the exact zero is only ever reached when the sources were
    # themselves exactly zero on that axis (`_reported_residual_ok`).
    return H1, H2, component_scale, centering_noise


def _eval_curve_scaled_components(C, t, rational, component_scale):
    """Evaluate Cartesian coordinates after normalizing each quiet axis.

    macOS aliases ``longdouble`` to float64, so multiplying the smallest
    subnormal by a Bernstein basis value can underflow even in the helper
    above.  Dividing each numerator coordinate by its own nonzero control
    scale *before* de Casteljau keeps that exactness information at O(1).
    """
    C = np.asarray(C, dtype=np.float64)
    scales = np.asarray(component_scale, dtype=np.float64)
    if rational:
        common = float(np.max(np.abs(C)))
        if not np.isfinite(common) or common <= 0.0:
            return None
        H = C / common
        weights = H[:, -1]
    else:
        H = C
        weights = None

    def _eval_scalar(values):
        work = np.asarray(values, dtype=np.float64).copy()
        tt = float(t)
        for _ in range(1, len(work)):
            work = (1.0 - tt) * work[:-1] + tt * work[1:]
        return float(work[0])

    denominator = _eval_scalar(weights) if rational else 1.0
    if not np.isfinite(denominator) or denominator == 0.0:
        return None
    result = np.zeros(len(scales), dtype=np.float64)
    for axis, scale in enumerate(scales):
        if scale == 0.0:
            continue
        numerator = _eval_scalar(H[:, axis] / scale)
        result[axis] = numerator / denominator
    return result


def _strict_residual_ok(C1, C2, u, v, rational, component_scale=None):
    """Accept a point equality only inside a floating roundoff envelope.

    ``atol`` is a search/resolution tolerance, not membership in the exact
    intersection set.  The envelope is tied to each coordinate's own control
    scale and curve degrees, so every representable nonzero offset in an
    otherwise constant coordinate remains nonzero for this predicate.
    """
    p1 = eval_curve(C1, float(u), rational=rational)
    p2 = eval_curve(C2, float(v), rational=rational)
    if not (np.all(np.isfinite(p1)) and np.all(np.isfinite(p2))):
        return False, p1, p2
    if component_scale is None:
        component_scale = _ccx_exactness_context(C1, C2, rational)
    if component_scale is None:
        return False, p1, p2
    C1_centered, C2_centered, scales = component_scale[:3]
    scales = np.asarray(scales, dtype=np.float64)
    centering_noise = np.asarray(component_scale[3], dtype=np.float64)
    absent = scales == 0.0
    # ABSENT MUST NOT MEAN IGNORED (review 2026-07-26).
    # `_eval_curve_scaled_components` skips any axis whose scale is 0, so
    # marking a noise-level axis absent used to drop it from this test
    # entirely — and this is the module's ONLY membership gate.  Two
    # segments in parallel planes x=X0 and x=X0+1e-8 were then certified as
    # intersecting at X0=1e6.  Evaluate those axes UNSCALED instead, and
    # bound them by the centering envelope they are absent WITH RESPECT TO.
    eval_scales = np.where(absent, 1.0, scales)
    p1_scaled = _eval_curve_scaled_components(
        C1_centered, u, True, eval_scales)
    p2_scaled = _eval_curve_scaled_components(
        C2_centered, v, True, eval_scales)
    if (p1_scaled is None or p2_scaled is None
            or not np.all(np.isfinite(p1_scaled))
            or not np.all(np.isfinite(p2_scaled))):
        return False, p1, p2
    degree_factor = max(1, len(C1) + len(C2))
    bound = ((32.0 * degree_factor * np.finfo(np.float64).eps)
             * np.maximum(1.0, np.maximum(
                 np.abs(p1_scaled), np.abs(p2_scaled))))
    # Sources genuinely zero on an axis give envelope 0, so the pre-existing
    # exact-zero behaviour is preserved bit-for-bit for a plane through the
    # origin; a plane at z=const keeps a real, nonzero envelope.
    # A PRESENT axis carries the same centering loss, just expressed in its
    # own scaled units.  Without this the bound demands agreement to
    # ~32*deg*eps of the SCALED value while the centered coordinates
    # themselves carry noise proportional to the model's WORLD position, so
    # nothing can be certified far from the origin: measured on the
    # float-built-subcurve fixture, `_strict_polish_ccx` returned None
    # 3020/3020 times at |T|=1e6 (5 of 6 at the model's own origin), which
    # is how a genuine exact overlap degraded to no overlap at all.
    # Absent axes are already bounded by the unscaled envelope above.
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled_noise = np.where(absent, 0.0,
                                centering_noise / np.where(absent, 1.0, scales))
    scaled_noise = np.where(np.isfinite(scaled_noise), scaled_noise, 0.0)
    bound = np.where(absent, centering_noise, bound + scaled_noise)
    return bool(np.all(np.abs(p1_scaled - p2_scaled) <= bound)), p1, p2


def _strict_polish_ccx(C1, C2, u, v, rational, component_scale=None,
                       require_newton=False):
    """Globally polish a candidate, then require exact-set membership.

    The Newton calls are intentionally unbounded by the current subdivision
    cell (they retain only the public [0,1] curve domains).  Cell bounds are
    a search device and must not turn a root just across a cell seam into a
    near-root.  Neither Newton's step size nor ``atol`` can accept the result;
    the component-wise residual certificate above is the sole membership
    gate.
    """
    u = float(np.clip(u, 0.0, 1.0))
    v = float(np.clip(v, 0.0, 1.0))
    ok, p1, p2 = _strict_residual_ok(
        C1, C2, u, v, rational, component_scale)
    if ok and not require_newton:
        return u, v, p1

    if component_scale is None:
        polish_C1, polish_C2, polish_rational = C1, C2, rational
    else:
        polish_C1, polish_C2 = component_scale[0], component_scale[1]
        polish_rational = True

    def _reported_residual_ok(G):
        if component_scale is None:
            return False
        scales = component_scale[2]
        # Per-axis centering-roundoff envelope (see `_ccx_exactness_context`).
        noise = component_scale[3]
        G = np.asarray(G, dtype=np.float64)
        scales = np.asarray(scales, dtype=np.float64)
        noise = np.asarray(noise, dtype=np.float64)
        if G.shape != scales.shape or not np.all(np.isfinite(G)):
            return False
        degree_factor = max(1, len(C1) + len(C2))
        limit = 32.0 * degree_factor * np.finfo(np.float64).eps
        for value, scale, axis_noise in zip(G, scales, noise):
            if scale == 0.0:
                # An absent axis carries no scale to normalize by, so the
                # bar is its own centering envelope.  Sources that really
                # are zero on this axis give envelope 0 and the test
                # degenerates to the exact `value == 0` it always was;
                # a centered planar pair, whose zero is reached only up to
                # the translation's cancellation, is admitted on the same
                # terms instead of being refused for being off by 1e-19.
                if abs(value) > axis_noise:
                    return False
            elif abs(value / scale) > limit:
                return False
        return True

    # A damped pass is robust at tangencies; the almost-undamped pass removes
    # the residual floor on well-conditioned transversal intersections.
    for damp in (1e-12, 1e-18):
        u, v, G, _last_step = newton_ccx(
            polish_C1, polish_C2, u, v, rational=polish_rational,
            tol=0.0, step_tol=8.0 * np.finfo(np.float64).eps,
            max_it=64, lm_damp=damp,
        )
        ok, p1, p2 = _strict_residual_ok(
            C1, C2, u, v, rational, component_scale)
        if ok and _reported_residual_ok(G):
            return float(u), float(v), p1
    return None


def _restrict_curve_interval(C, lo, hi):
    """Return control points reparameterized from [0,1] onto [lo,hi]."""
    C = np.asarray(C)
    reverse = hi < lo
    a, b = (hi, lo) if reverse else (lo, hi)
    if not (0.0 <= a < b <= 1.0):
        return None
    out = C.copy()
    if a > 0.0:
        _, out = _subdivide_curve(out, a)
    if b < 1.0:
        local_b = (b - a) / (1.0 - a) if a > 0.0 else b
        out, _ = _subdivide_curve(out, local_b)
    if reverse:
        out = out[::-1].copy()
    return out


# L52 slice 6a: the shared exact product (broadcast-generalized; same
# accumulation order and longdouble factor arithmetic as the old private
# copy — 1-D operands are the degenerate broadcast case).
_bernstein_product_1d = bernstein_product_1d


def _homogeneous_curve_for_identity(C, rational, origin,
                                    with_source_scale=False):
    if not with_source_scale:
        H = _center_curve_homogeneous_for_exactness(C, rational, origin)
        if H is None:
            return None
        return np.asarray(H, dtype=np.longdouble)
    H, src = _center_curve_homogeneous_for_exactness(
        C, rational, origin, with_source_scale=True)
    if H is None:
        return None, None
    return (np.asarray(H, dtype=np.longdouble),
            np.asarray(src, dtype=np.longdouble))


def _overlap_mapping_is_identity(C1, C2, u_range, v_range, rational):
    """Certify ``C1(u(t)) == C2(v(t))`` as a Bernstein identity.

    This is an all-parameter certificate, not a finite sample test.  For
    rational curves it forms every coefficient of

        P1_k(t) W2(t) - P2_k(t) W1(t)

    after restricting both curves to the proposed affine parameter ranges.
    A coefficient is zero only within the roundoff accumulated by its two
    product terms; a one-sided nonzero coordinate offset therefore cannot be
    hidden by a model-scale tolerance or independent homogeneous scaling.
    """
    points1 = _cartesian_curve_controls_for_exactness(C1, rational)
    points2 = _cartesian_curve_controls_for_exactness(C2, rational)
    if points1 is None or points2 is None:
        return False
    origin = points1[0]
    H1, src1 = _homogeneous_curve_for_identity(
        C1, rational, origin, with_source_scale=True)
    H2, src2 = _homogeneous_curve_for_identity(
        C2, rational, origin, with_source_scale=True)
    if H1 is None or H2 is None or H1.shape[1] != H2.shape[1]:
        return False
    h1_xyz_scale = np.max(np.abs(H1[:, :-1]), axis=0)
    h2_xyz_scale = np.max(np.abs(H2[:, :-1]), axis=0)
    h1_weight_scale = np.max(np.abs(H1[:, -1]))
    h2_weight_scale = np.max(np.abs(H2[:, -1]))
    source_product_scale = (
        h1_xyz_scale * h2_weight_scale
        + h2_xyz_scale * h1_weight_scale)
    # Third term (2026-07-26): the common-origin CENTERING itself.
    # The two scales above are taken from the already-centered nets, so
    # they measure what survived the cancellation, not what was consumed by
    # it -- the same circular measurement removed from the Phase-2 prune in
    # the cluster-4 tier.  A curve translated away from the origin loses
    # precision proportional to its WORLD position while these scales stay
    # at the model's own extent, so the envelope stopped covering the
    # noise: this module's own calibrated float-built-subcurve fixture
    # certifies at its native position but was refused 139/200 at |T|=1 and
    # 200/200 at |T|=1e6, downgrading genuine exact overlaps to
    # 'tolerance' and then to no overlap at all.
    centering_product_scale = (
        np.max(src1, axis=0) * h2_weight_scale
        + np.max(src2, axis=0) * h1_weight_scale)
    H1 = _restrict_curve_interval(H1, float(u_range[0]), float(u_range[1]))
    H2 = _restrict_curve_interval(H2, float(v_range[0]), float(v_range[1]))
    if H1 is None or H2 is None:
        return False
    w1 = H1[:, -1]
    w2 = H2[:, -1]
    if (np.any(w1 == 0.0) or np.any(w2 == 0.0)
            or not np.all(np.isfinite(w1))
            or not np.all(np.isfinite(w2))):
        return False

    eps = np.finfo(np.longdouble).eps
    op_factor = np.longdouble(
        64 * max(1, len(H1)) * max(1, len(H2))) * eps
    # Interval restriction is performed in the source floating precision
    # before the long-double Bernstein products.  Its cancellation error is
    # tied to the un-restricted source products, not to a coefficient that
    # may have cancelled to zero.  The common-origin centering above is
    # essential: a large world translation cannot inflate this floor and
    # hide a real offset.
    source_factor = (np.longdouble(8192 * max(1, len(H1) + len(H2)))
                     * np.longdouble(np.finfo(np.float64).eps))
    # The centering is ONE subtraction (four roundings) — priced with
    # `_CENTERING_OPS`, not with the restriction's accumulation factor.  At
    # the model's own origin its operands are O(the model), so this term is
    # ~eps and changes nothing; it grows only in proportion to the
    # precision a world translation actually destroys.
    centering_factor = (np.longdouble(_CENTERING_OPS)
                        * np.longdouble(np.finfo(np.float64).eps))
    for axis in range(H1.shape[1] - 1):
        lhs = _bernstein_product_1d(H1[:, axis], w2)
        rhs = _bernstein_product_1d(H2[:, axis], w1)
        residual = np.abs(lhs - rhs)
        roundoff = (
            centering_factor * centering_product_scale[axis]
            + op_factor * (np.abs(lhs) + np.abs(rhs))
            + source_factor * source_product_scale[axis])
        if np.any(residual > roundoff):
            return False
    return True


def _invert_point_on_curve(C, pt, v0, rational, max_iter=40):
    """Project *pt* onto curve ``C``: damped 1-D Gauss-Newton on
    ``||C(v) - pt||`` with a monotone-decreasing line search, clamped to
    [0,1]. Returns ``(v, residual)`` for the best point found."""
    pt = np.asarray(pt, dtype=np.float64)
    v = float(min(1.0, max(0.0, float(v0))))
    p, d = eval_curve_d1(C, v, rational=rational)
    r = p - pt
    best_v, best_r = v, float(np.linalg.norm(r))
    for _ in range(max_iter):
        denom = float(np.dot(d, d))
        if denom < 1e-30:
            break
        step = -float(np.dot(r, d)) / denom
        if abs(step) < 1e-16:
            break
        scale = 1.0
        improved = False
        for _ls in range(20):
            cand = min(1.0, max(0.0, v + scale * step))
            p_c, d_c = eval_curve_d1(C, cand, rational=rational)
            r_c = p_c - pt
            n_c = float(np.linalg.norm(r_c))
            if n_c < best_r:
                v, p, d, r = cand, p_c, d_c, r_c
                best_v, best_r = cand, n_c
                improved = True
                break
            scale *= 0.5
        if not improved:
            break
    return best_v, best_r


def _project_point_on_curve(C, pt, rational, seed=None):
    """Coarse-scan (when unseeded) + Newton projection of *pt* onto ``C``."""
    if seed is None:
        ts = np.linspace(0.0, 1.0, 17)
        ds = [float(np.linalg.norm(
            eval_curve(C, float(t), rational=rational)
            - np.asarray(pt, dtype=np.float64))) for t in ts]
        seed = float(ts[int(np.argmin(ds))])
    return _invert_point_on_curve(C, pt, seed, rational)


def _tolerance_overlap_certificate(C1, C2, atol, rational, ptol_u, ptol_v,
                                   n_samples=65):
    """L47 residual-certified overlap tier (user decision 2026-07-12).

    The exact-affine identity above certifies coefficient-identical span
    pairs only. Near-coincident pairs (same path within ``atol`` but not
    exactly) and same-locus non-affine reparameterizations are genuine
    overlaps at modeling tolerance — the semantics CSX claims (L27) and SSX
    regions (L28) already use. This certificate:

      1. gates on the four domain-endpoint inversions (an admissible span
         must be pinned by a domain end of one curve at each end; interior-
         ended near-coincident bands stay unpromoted — typed partial);
      2. densely samples the candidate span, requiring every point of C1 to
         invert onto C2 within ``atol`` with a monotone parameter pairing;
      3. refuses to merge sub-tolerance TOPOLOGY (the CSX invariant, 1-D
         form): a transverse-direction flip between consecutive GAP samples
         — bridging any root-like samples in between, so a crossing landing
         exactly on a sample node is still seen — means the curves CROSS
         inside the band; the certificate rejects and returns the crossing
         brackets so the caller can certify them as isolated roots instead.
         A non-flipping (tangential) touch inside the band is covered by
         the overlap; root-like dips alone are not crossing evidence.

    Returns ``(overlap | None, brackets, band_evidence, span_evidence)``:
    ``brackets`` is a list of ``(u, v)`` seeds for strict root polishing;
    ``band_evidence`` is True when some qualifying domain end continues as a
    within-``atol`` coincidence BAND into the domain interior (inward-probe
    test) — a mere corner CONTACT (curves within atol at one endpoint but
    diverging immediately, e.g. consecutive edges of a loop sharing a
    vertex) is NOT band evidence and must not arm the bounded fallback;
    ``span_evidence`` is the widest endpoint-qualified u-extent (or None).
    """
    ends1 = [np.asarray(eval_curve(C1, t, rational=rational), dtype=np.float64)
             for t in (0.0, 1.0)]
    ends2 = [np.asarray(eval_curve(C2, t, rational=rational), dtype=np.float64)
             for t in (0.0, 1.0)]
    pts1 = _cartesian_curve_controls_for_exactness(C1, rational)
    pts2 = _cartesian_curve_controls_for_exactness(C2, rational)
    if pts1 is None or pts2 is None:
        return None, [], 0, None
    allpts = np.vstack([pts1, pts2])
    diag = float(np.linalg.norm(allpts.max(axis=0) - allpts.min(axis=0)))
    # Roundoff floor separating "exact root at this sample" from "genuine
    # gap": generous vs eval noise, far below any modeling tolerance.
    tiny = max(4096.0 * float(np.finfo(np.float64).eps), 1e-12) * max(1.0, diag)

    # Sound AABB pre-filter: the curve lies inside its control-point box, so
    # an endpoint farther than atol from the box cannot project within atol.
    # This keeps the endpoint gate ~free on generic (non-coincident) pairs —
    # CSX runs up to 4 nested CCX calls per cut face, so a Newton projection
    # per endpoint on every call is real wall-clock on the SSX gates.
    lo1 = pts1.min(axis=0) - atol
    hi1 = pts1.max(axis=0) + atol
    lo2 = pts2.min(axis=0) - atol
    hi2 = pts2.max(axis=0) + atol

    def _outside(pt, lo, hi):
        return bool(np.any(pt < lo) or np.any(pt > hi))

    cands = []
    band_ends = []   # (curve_index, t_end) of qualifying domain ends
    for t_end, pt in ((0.0, ends1[0]), (1.0, ends1[1])):
        if _outside(pt, lo2, hi2):
            continue
        v_end, r_end = _project_point_on_curve(C2, pt, rational)
        if r_end <= atol:
            cands.append((t_end, v_end))
            band_ends.append((0, t_end))
    for t_end, pt in ((0.0, ends2[0]), (1.0, ends2[1])):
        if _outside(pt, lo1, hi1):
            continue
        u_end, r_end = _project_point_on_curve(C1, pt, rational)
        if r_end <= atol:
            cands.append((u_end, t_end))
            band_ends.append((1, t_end))

    # Inward-probe band test: a coincidence BAND continues within atol into
    # the domain interior; a corner CONTACT (shared vertex of consecutive
    # edges) diverges immediately. Only bands justify the bounded fallback.
    band_evidence = False
    for which, t_end in band_ends:
        src, dst = (C1, C2) if which == 0 else (C2, C1)
        for step in (0.02, 0.05):
            t_in = step if t_end == 0.0 else 1.0 - step
            pt_in = np.asarray(eval_curve(src, t_in, rational=rational),
                               dtype=np.float64)
            _t_proj, r_in = _project_point_on_curve(dst, pt_in, rational)
            if r_in <= atol:
                band_evidence = True
                break
        if band_evidence:
            break

    if len(cands) < 2:
        return None, [], band_evidence, None
    uniq = []
    for u, v in cands:
        if not any(abs(u - uu) <= 4.0 * ptol_u and abs(v - vv) <= 4.0 * ptol_v
                   for uu, vv in uniq):
            uniq.append((float(u), float(v)))
    span_evidence = None
    if len(uniq) >= 2:
        u_ext = [u for u, _v in uniq]
        span_evidence = (float(min(u_ext)), float(max(u_ext)))

    pairs = [(uniq[i], uniq[j])
             for i in range(len(uniq)) for j in range(i + 1, len(uniq))]
    pairs.sort(key=lambda pr: -abs(pr[1][0] - pr[0][0]))

    brackets: list[tuple[float, float]] = []
    for (ua, va), (ub, vb) in pairs:
        if (abs(ub - ua) <= max(4.0 * ptol_u, 1e-6)
                or abs(vb - va) <= max(4.0 * ptol_v, 1e-6)):
            continue
        if ub < ua:
            ua, ub, va, vb = ub, ua, vb, va
        direction = 1.0 if vb >= va else -1.0
        us = np.linspace(ua, ub, n_samples)
        res = np.empty(n_samples)
        vs = np.empty(n_samples)
        dvecs = np.empty((n_samples, ends1[0].shape[0]))
        ok = True
        v_prev = None
        for k in range(n_samples):
            u_k = float(us[k])
            pt = np.asarray(eval_curve(C1, u_k, rational=rational),
                            dtype=np.float64)
            v_seed = va + (vb - va) * (k / (n_samples - 1.0))
            if v_prev is not None:
                v_seed = v_prev + (vb - va) / (n_samples - 1.0)
            v_k, r_k = _invert_point_on_curve(C2, pt, v_seed, rational)
            if r_k > atol:
                v_k2, r_k2 = _project_point_on_curve(C2, pt, rational)
                if r_k2 < r_k:
                    v_k, r_k = v_k2, r_k2
            if r_k > atol:
                ok = False
                break
            if (v_prev is not None
                    and direction * (v_k - v_prev) < -4.0 * ptol_v):
                ok = False    # folded pairing — not a functional overlap
                break
            vs[k] = v_k
            res[k] = r_k
            dvecs[k] = pt - np.asarray(
                eval_curve(C2, float(v_k), rational=rational),
                dtype=np.float64)
            v_prev = v_k
        if not ok:
            continue

        root_like = res <= tiny
        if not bool(np.all(root_like)):
            # Sub-tolerance topology guard (CSX invariant, 1-D form): a
            # transverse-direction FLIP between consecutive GAP samples
            # means the curves cross inside the band — never merge; reject
            # the promotion and hand the crossing brackets to strict root
            # polishing. Root-like samples carry no direction information
            # (a residual dip below roundoff scale alone is NOT crossing
            # evidence — a y-offset of a curve is locally tangent to it
            # wherever the tangent is parallel to the offset, dipping to
            # (offset)^2·kappa with no root there), so the flip test
            # BRIDGES root-like runs and compares each gap sample with the
            # NEXT gap sample: a crossing landing exactly ON a sample node
            # (dyadic-64 fractions — audit L54[A2-1]) shows as a root-like
            # node between opposite-side gaps and must be caught, with the
            # bracket at that node (the root is exactly there). A
            # non-flipping exact touch inside the band stays legitimately
            # covered by the tolerance overlap. Residual limit: an EVEN
            # number of crossings inside one sample interval still aliases
            # to no flip (both gap neighbours same side) — inherent to the
            # fixed 65-sample grid, documented in the L54 ledger entry.
            pair_brackets = []
            gap_idx = [k for k in range(n_samples) if not root_like[k]]
            for a_i, b_i in zip(gap_idx, gap_idx[1:]):
                if float(np.dot(dvecs[a_i], dvecs[b_i])) < 0.0:
                    if b_i - a_i > 1:
                        k_root = a_i + 1
                        pair_brackets.append(
                            (float(us[k_root]), float(vs[k_root])))
                    else:
                        pair_brackets.append(
                            (0.5 * float(us[a_i] + us[b_i]),
                             0.5 * float(vs[a_i] + vs[b_i])))
            if pair_brackets:
                brackets.extend(pair_brackets)
                continue
        return ({
            "boundary_zeros": [],
            "overlap_endpoints": [],
            "u_range": (float(ua), float(ub)),
            "v_range": (float(va), float(vb)),
            "certification": "tolerance",
            "residual_max": float(res.max()),
        }, brackets, band_evidence, span_evidence)
    return None, brackets, band_evidence, span_evidence


def _vector_residual_hull_excludes_zero(C1, C2, rational, depth):
    """Certify that one Cartesian residual component cannot be zero.

    For each component this constructs the full tensor-product Bernstein net

        P1_k(u) W2(v) - P2_k(v) W1(u).

    If every coefficient is strictly on the same side of zero, with a
    subdivision- and product-roundoff margin, the two curve pieces cannot
    intersect.  Independent homogeneous scales cancel because each curve is
    normalized as a whole before the cross product.

    Envelope (2026-07-25, cluster-4 burn-down).  The margin is the house
    TWO-TERM derived form, not the operator term alone:

      * the OPERATOR term ``op_factor*(|lhs|+|rhs|)`` covers the cross
        products, which are formed from the values at hand;
      * the SOURCE term covers the common-origin centering that produced
        those values.  Centering divides by the net scale and only then
        subtracts ``origin*w``, so a coordinate that is identically zero
        after translation arrives as cancellation noise rather than as an
        exact zero.  Bounding that noise by the cancelled value itself is
        circular — measured on the user's boundary-coincidence pair, the
        z row carried 5.7e-19 of noise against a 2.6e-31 operator margin
        and this prune deleted the entire root cell of a transversal
        crossing.  The term is therefore built from the operand magnitudes
        the subtraction consumed (`with_source_scale`).

    Direction matters here in a way it does not for the sibling
    certificates: `_overlap_mapping_is_identity` and csx's
    `_certify_affine_csx_overlap` ask "is this residual explainable as
    zero?", where too small a margin merely REFUSES to certify.  This
    predicate asserts a residual is provably NON-zero and deletes the cell,
    with no downstream recourse — an underestimate silently loses
    solutions.  Reusing their (calibrated-from-below) source factor is thus
    conservative in exactly the right direction here.  A coordinate genuinely
    absent from both sources keeps a zero source floor, so the prune retains
    full strength on real separations.
    """
    points1 = _cartesian_curve_controls_for_exactness(C1, rational)
    points2 = _cartesian_curve_controls_for_exactness(C2, rational)
    if points1 is None or points2 is None:
        return False
    origin = points1[0]
    H1, src1 = _homogeneous_curve_for_identity(
        C1, rational, origin, with_source_scale=True)
    H2, src2 = _homogeneous_curve_for_identity(
        C2, rational, origin, with_source_scale=True)
    if H1 is None or H2 is None or H1.shape[1] != H2.shape[1]:
        return False
    w1 = H1[:, -1]
    w2 = H2[:, -1]
    if (not np.all(np.isfinite(w1)) or not np.all(np.isfinite(w2))
            or np.any(w1 == 0.0) or np.any(w2 == 0.0)):
        return False

    eps = np.finfo(np.longdouble).eps
    op_factor = np.longdouble(
        1024 * max(1, depth + 1) * max(len(H1), len(H2))) * eps
    # The centering runs in the source float64 precision ahead of the
    # long-double cross products, so its term scales with eps_float64.
    # FACTOR (review 2026-07-26): `_CENTERING_OPS`, not the siblings'
    # `8192*(n1+n2)`.  That constant prices a Bernstein product/restriction
    # chain; what is being bounded here is the same single subtraction as in
    # `_ccx_exactness_context`.  Borrowing it cost ~3 orders of prune reach
    # at large world positions — measured floor 2.9e-11 at |T|=1 and 2.9e-2
    # at |T|=1e9, the latter ABOVE a default atol=1e-3, i.e. the prune had
    # stopped separating anything at tolerance scale out there.
    source_factor = (np.longdouble(_CENTERING_OPS)
                     * np.longdouble(np.finfo(np.float64).eps))
    abs_w1 = np.abs(w1)
    abs_w2 = np.abs(w2)
    for axis in range(H1.shape[1] - 1):
        lhs = H1[:, axis, None] * w2[None, :]
        rhs = w1[:, None] * H2[None, :, axis]
        residual = lhs - rhs
        source_term = (src1[:, axis, None] * abs_w2[None, :]
                       + abs_w1[:, None] * src2[None, :, axis])
        roundoff = (op_factor * (np.abs(lhs) + np.abs(rhs))
                    + source_factor * source_term)
        if (np.all(residual > roundoff)
                or np.all(residual < -roundoff)):
            return True
    return False



from mmcore.numeric.bern import bernstein_partial_derivative_coeffs

# ---------------------------------------------------------------------------
# Phase 2 helpers
# ---------------------------------------------------------------------------

# L52 slice 7: shared implementation in _bezier_common (verbatim move).
_restrict_net_axis = restrict_net_axis


def _split_intervals(cut, lo, hi, ptol):
    """Split [lo, hi] into up to 3 sub-intervals around cut ± ptol."""
    cut_lo = max(cut - ptol, lo)
    cut_hi = min(cut + ptol, hi)
    intervals = []
    if lo + 1e-15 < cut_lo:
        intervals.append((lo, cut_lo))
    intervals.append((cut_lo, cut_hi))
    if cut_hi < hi - 1e-15:
        intervals.append((cut_hi, hi))
    return intervals


def _cutout_2d(F_cell, seg1, seg2, pw, qw, u0, u1, v0, v1, depth,
               u_cut, v_cut, ptol_u, ptol_v, rational):
    """Cut out a ptol-neighborhood around (u_cut, v_cut) from a 2D cell.

    Splits along both axes at cut ± ptol, producing 3×3 = 9 boxes.
    The center box is discarded. Remaining 8 are returned with restricted nets.
    """
    u_intervals = _split_intervals(u_cut, u0, u1, ptol_u)
    v_intervals = _split_intervals(v_cut, v0, v1, ptol_v)

    u_center = len(u_intervals) // 2 if len(u_intervals) == 3 else (0 if len(u_intervals) == 1 else -1)
    v_center = len(v_intervals) // 2 if len(v_intervals) == 3 else (0 if len(v_intervals) == 1 else -1)

    sub_cells = []
    for ui, (u_lo, u_hi) in enumerate(u_intervals):
        for vi, (v_lo, v_hi) in enumerate(v_intervals):
            if ui == u_center and vi == v_center:
                continue
            if u_hi - u_lo < 1e-15 or v_hi - v_lo < 1e-15:
                continue

            F_sub = _restrict_net_axis(F_cell, 0, u_lo, u_hi, u0, u1)
            F_sub = _restrict_net_axis(F_sub, 1, v_lo, v_hi, v0, v1)

            # Restrict curves
            u_frac_lo = (u_lo - u0) / max(u1 - u0, 1e-30)
            u_frac_hi = (u_hi - u0) / max(u1 - u0, 1e-30)
            C1_sub = seg1
            if u_frac_lo > 1e-12:
                _, C1_sub = _subdivide_curve(C1_sub, u_frac_lo)
            if u_frac_hi < 1.0 - 1e-12:
                uf_rescaled = (u_frac_hi - u_frac_lo) / (1.0 - u_frac_lo) if u_frac_lo > 1e-12 else u_frac_hi
                C1_sub, _ = _subdivide_curve(C1_sub, uf_rescaled)
            pw_sub = C1_sub[:, -1].copy() if rational else np.ones(C1_sub.shape[0])

            v_frac_lo = (v_lo - v0) / max(v1 - v0, 1e-30)
            v_frac_hi = (v_hi - v0) / max(v1 - v0, 1e-30)
            C2_sub = seg2
            if v_frac_lo > 1e-12:
                _, C2_sub = _subdivide_curve(C2_sub, v_frac_lo)
            if v_frac_hi < 1.0 - 1e-12:
                vf_rescaled = (v_frac_hi - v_frac_lo) / (1.0 - v_frac_lo) if v_frac_lo > 1e-12 else v_frac_hi
                C2_sub, _ = _subdivide_curve(C2_sub, vf_rescaled)
            qw_sub = C2_sub[:, -1].copy() if rational else np.ones(C2_sub.shape[0])

            sub_cells.append((C1_sub, C2_sub, F_sub, pw_sub, qw_sub,
                              u_lo, u_hi, v_lo, v_hi, depth + 1))
    return sub_cells


def _phase2_ccx(F, C1, C2, C1_orig, C2_orig,
                u_lo, u_hi, v_lo, v_hi,
                atol, rational, ptol_u, ptol_v,
                known_points=None,
                max_depth=50, max_cells=50_000,
                max_results=4_096,
                initial_stack=None):
    """Phase 2: find isolated intersections via subdivision + Newton + cutout.

    No boundary analysis, no overlap checks, no classifier.
    Just: min-of-net → derivative sign → Newton → cutout.
    """
    from mmcore.numeric.intersection._sq_dist_classify import (
        _check_min_of_net, _check_lipschitz, _weight_max_product,
    )

    isolated = list(known_points) if known_points else []
    n_known = len(isolated)
    cells = 0
    exhausted = False
    component_scale = _ccx_exactness_context(
        C1_orig, C2_orig, rational)

    if initial_stack is not None:
        stack = list(initial_stack)
    else:
        _, Pw = extract_weights(C1, rational=rational)
        _, Qw = extract_weights(C2, rational=rational)
        stack = [(C1, C2, F, Pw.copy(), Qw.copy(),
                  u_lo, u_hi, v_lo, v_hi, 0)]

    while stack:
        if cells >= max_cells or len(isolated) - n_known >= max_results:
            exhausted = True
            break
        cells += 1

        seg1, seg2, F_cell, pw, qw, u0, u1, v0, v1, depth = stack.pop()
        #print(f"CCX: {cells} cells: {( ( u0, u1), (v0, v1), depth)}")
        # AABB prune: cheapest possible check — control point bounding boxes
        from mmcore.numeric._aabb import aabb, aabb_intersect
        if rational:
            pts1 = seg1[:, :-1] / seg1[:, -1:]
            pts2 = seg2[:, :-1] / seg2[:, -1:]
        else:
            pts1 = seg1
            pts2 = seg2
        if _vector_residual_hull_excludes_zero(
                seg1, seg2, rational, depth):
            continue
        bb1 = np.array(aabb(pts1)); bb1[0] -= atol; bb1[1] += atol
        bb2 = np.array(aabb(pts2)); bb2[0] -= atol; bb2[1] += atol
        if not aabb_intersect(bb1, bb2):
            continue

        w_sc = _weight_max_product(pw, qw)

        # min-of-net prune
        if _check_min_of_net(F_cell, atol, w_sc):
            continue

        # Lipschitz prune
        if _check_lipschitz(F_cell, atol, w_sc):
            continue

        # Derivative sign pruning
        Fv = F_cell[..., np.newaxis]
        can_have_stationary = True
        for ax in range(2):
            dF = bernstein_partial_derivative_coeffs(Fv, axis=ax)
            coeffs = dF[..., 0]
            if np.min(coeffs) > 0 or np.max(coeffs) <= 0:
                can_have_stationary = False
                break
        if not can_have_stationary:
            continue

        # ptol-based early termination
        if (u1 - u0) <= ptol_u and (v1 - v0) <= ptol_v:
            u_mid = 0.5 * (u0 + u1)
            v_mid = 0.5 * (v0 + v1)
            polished = _strict_polish_ccx(
                C1_orig, C2_orig, u_mid, v_mid, rational,
                component_scale=component_scale, require_newton=True)
            if polished is not None:
                u_sol, v_sol, pt = polished
                if (u0 - 0.25 * ptol_u <= u_sol <= u1 + 0.25 * ptol_u
                        and v0 - 0.25 * ptol_v <= v_sol
                        <= v1 + 0.25 * ptol_v
                        and not _is_duplicate(isolated, pt, atol)):
                    isolated.append({
                        "u": float(u_sol), "v": float(v_sol),
                        "point": pt, "_micro": True,
                    })
            continue

        # Newton from cell center
        u_mid = 0.5 * (u0 + u1)
        v_mid = 0.5 * (v0 + v1)
        uv_candidates = [
            (u0, v0), (u0, v1), (u1, v1), (u1, v0),
            (u_mid, v_mid),
        ]
        root_found = False
        for u_mid, v_mid in uv_candidates:
            if root_found:
                break
            polished = _strict_polish_ccx(
                C1_orig, C2_orig, u_mid, v_mid, rational,
                component_scale=component_scale, require_newton=True)
            if polished is not None:
                u_sol, v_sol, pt = polished
            else:
                continue
            if u0 < u_sol < u1 and v0 < v_sol < v1:
                is_new = not _is_duplicate(isolated, pt, atol)
                #print(f"CCX: is_new: {is_new}")
                if is_new:
                    isolated.append({"u": float(u_sol), "v": float(v_sol), "point": pt})
                    sub_cells = _cutout_2d(
                        F_cell, seg1, seg2, pw, qw, u0, u1, v0, v1, depth,
                        float(u_sol), float(v_sol), ptol_u, ptol_v, rational,
                    )
                    stack.extend(sub_cells)
                    root_found=True
                    break

        if root_found:continue


        if depth >= max_depth:
            # None of the certificates above proved this cell root-free and
            # Newton did not resolve it.  Dropping it at the depth guard is a
            # partial search just like dropping queued cells at ``max_cells``.
            exhausted = True
            continue

        # Subdivide




        u_mid_split = 0.5 * (u0 + u1)
        v_mid_split = 0.5 * (v0 + v1)
        seg1_L, seg1_R = _subdivide_curve(seg1)
        seg2_L, seg2_R = _subdivide_curve(seg2)
        F_LL, F_LR, F_RR,F_RL, =_subdivide_sq_dist_net_2d(F_cell,0.5 ,0.5)


        pw_L = seg1_L[:, -1].copy() if rational else np.ones(seg1_L.shape[0])
        pw_R = seg1_R[:, -1].copy() if rational else np.ones(seg1_R.shape[0])
        qw_L = seg2_L[:, -1].copy() if rational else np.ones(seg2_L.shape[0])
        qw_R = seg2_R[:, -1].copy() if rational else np.ones(seg2_R.shape[0])
        stack.append((seg1_L, seg2_L, F_LL, pw_L, qw_L, u0, u_mid_split, v0, v_mid_split, depth+1))
        stack.append((seg1_L, seg2_R, F_LR, pw_L, qw_R,u0, u_mid_split, v_mid_split, v1, depth+1))
        stack.append((seg1_R, seg2_R, F_RR, pw_R,  qw_R,u_mid_split, u1, v_mid_split, v1, depth+1))
        stack.append((seg1_R, seg2_L, F_RL, pw_R, qw_L,u_mid_split, u1,  v0, v_mid_split, depth+1))

    return isolated[n_known:], exhausted, cells


# ---------------------------------------------------------------------------
# Main algorithm: two-phase architecture
# ---------------------------------------------------------------------------
from mmcore.numeric._bezier_common import _compute_remaining_intervals


def bez_ccx(
    C1,
    C2,
    atol=1e-3,
    rational=False,
    max_depth=50,
    max_cells=100_000,
    max_results=4_096,
) -> dict:
    """Bezier curve-curve intersection via two-phase architecture.

    Phase 1: Find boundary intersections and overlaps on the initial patch.
             These can ONLY exist at the boundaries of the original objects.
    Phase 2: Search for isolated intersections on the remaining parameter
             intervals via subdivision + Newton + cutout.

    ``max_cells`` is shared by Phase 1 and every Phase-2 interval.
    ``max_results`` also caps Phase-1 boundary roots before pairwise topology
    checks.  A capped return sets ``budget_exhausted``; callers must treat
    ``boundary_topology_complete=False`` as diagnostic partial output.

    Overlap entries carry ``certification``: ``'exact'`` (Bernstein affine
    identity) or ``'tolerance'`` (L47 residual tier: dense-sample inversion
    pairing within ``atol`` + ``residual_max``, e.g. near-coincident twins
    and non-affine same-locus reparameterizations). An overlap-class
    structure that NEITHER certificate can promote and the bounded fallback
    cannot discretize returns ``uncertified_overlap_span=(u_lo, u_hi)`` with
    ``boundary_topology_complete=False`` — typed, not a bare budget flag.
    """
    C1 = np.asarray(C1, dtype=np.float64)
    C2 = np.asarray(C2, dtype=np.float64)

    cells = DownCounter(max_cells)
    budget_exhausted = False

    def _result(isolated, overlaps, *, topology_complete=True):
        return {
            "isolated": isolated,
            "overlaps": overlaps,
            "budget_exhausted": bool(budget_exhausted),
            "cells_processed": int(cells.processed),
            "boundary_topology_complete": bool(topology_complete),
        }

    if cells.remaining <= 0:
        budget_exhausted = True
        return _result([], [], topology_complete=False)

    F = curve_curve_squared_net_homog(C1, C2, rational=rational)

    _, Pw = extract_weights(C1, rational=rational)
    _, Qw = extract_weights(C2, rational=rational)

    C1_orig = C1
    C2_orig = C2

    ptol_u, ptol_v = _compute_param_tols(C1, C2, atol, rational)
    component_scale = _ccx_exactness_context(
        C1_orig, C2_orig, rational)

    isolated = []
    overlaps = []

    # ===================================================================
    # PHASE 1: Boundary analysis + overlap (initial patch only)
    # ===================================================================
    # Charge the classifier itself, then share the remaining allowance with
    # every recursive 1-D boundary solve it invokes.  The root cap is kept
    # deliberately below the point where the classifier's pairwise valley
    # check becomes a material O(B^2) operation.
    cells.spend(1)
    from mmcore.numeric._bern_zero_1d import bernstein_zero_budget
    boundary_root_cap = min(max(0, int(max_results)), 128)

    # ``_check_overlap`` pairs every near-zero coefficient on opposite
    # boundaries.  Bound that candidate materialization before entering the
    # legacy classifier; limiting only the later precise-root list would be
    # too late to prevent this separate quadratic path.
    w_scale = float(np.max(np.abs(Pw))) * float(np.max(np.abs(Qw)))
    boundary_threshold = (atol * w_scale) ** 2
    boundary_candidate_counts = (
        int(np.count_nonzero(np.abs(F[0, :]) < boundary_threshold)),
        int(np.count_nonzero(np.abs(F[-1, :]) < boundary_threshold)),
        int(np.count_nonzero(np.abs(F[:, 0]) < boundary_threshold)),
        int(np.count_nonzero(np.abs(F[:, -1]) < boundary_threshold)),
    )
    if any(count > boundary_root_cap for count in boundary_candidate_counts):
        budget_exhausted = True
        return _result([], [], topology_complete=False)

    with bernstein_zero_budget(cells.remaining, boundary_root_cap) as zero_budget:
        cls = classify_sq_dist_net(F, atol, Pw, Qw)
    cells.spend(zero_budget.nodes)

    # A capped boundary solve is not a smaller-but-valid topology.  In
    # particular, using its endpoints to declare an overlap or cut Phase 2
    # could hide real roots.  Discard it and mark the result explicitly
    # partial so callers such as CSX cannot silently consume it.
    if zero_budget.exhausted:
        budget_exhausted = True
        return _result([], [], topology_complete=False)
    if any(not isinstance(bz, BoundaryZero) for bz in cls.precise_zeros):
        budget_exhausted = True
        return _result([], [], topology_complete=False)
    #print(f"CCX: {cls} (phase 1)")
    if cls.kind == NO_INTERSECTION:
        return _result([], [])

    # 1a. Collect validated boundary zeros (don't add to isolated yet)
    boundary_hits = []  # list of strictly validated (u, v, point)
    if cls.precise_zeros:
        for bz in cls.precise_zeros:
            u_bz, v_bz = _boundary_zero_to_uv(bz, 0.0, 1.0, 0.0, 1.0)
            polished = _strict_polish_ccx(
                C1_orig, C2_orig, u_bz, v_bz, rational,
                component_scale=component_scale)
            if polished is not None:
                u_sol, v_sol, point = polished
                boundary_hits.append((float(u_sol), float(v_sol), point))

    # 1b. Check for overlap
    overlap_found = False
    if cls.kind == OVERLAP:
        # The legacy classifier's OVERLAP kind is intentionally tolerant and
        # its ``overlap_endpoints`` payload can be only a valley witness.  It
        # is therefore a candidate generator, never a public topology claim.
        # Build paired strict endpoint roots, then promote only an affine map
        # whose full Bernstein cross-residual is the zero identity.
        uv_roots = []
        uv_eps = 64.0 * np.finfo(np.float64).eps
        for u_hit, v_hit, _point in boundary_hits:
            if not any(abs(u_hit - u0) <= uv_eps
                       and abs(v_hit - v0) <= uv_eps
                       for u0, v0 in uv_roots):
                uv_roots.append((u_hit, v_hit))

        endpoint_pairs = []
        for i in range(len(uv_roots)):
            for j in range(i + 1, len(uv_roots)):
                endpoint_pairs.append((uv_roots[i], uv_roots[j]))

        # A classifier may summarize a full overlap without retaining typed
        # endpoint roots.  These two domain-spanning affine maps are still
        # candidates, but their endpoints and coefficient identity must pass
        # the same strict gates as every inferred sub-range.
        for pair in (
                ((0.0, 0.0), (1.0, 1.0)),
                ((0.0, 1.0), (1.0, 0.0))):
            p0 = _strict_polish_ccx(
                C1_orig, C2_orig, *pair[0], rational,
                component_scale=component_scale)
            p1 = _strict_polish_ccx(
                C1_orig, C2_orig, *pair[1], rational,
                component_scale=component_scale)
            if p0 is not None and p1 is not None:
                endpoint_pairs.append(
                    ((p0[0], p0[1]), (p1[0], p1[1])))

        endpoint_pairs.sort(
            key=lambda pair: -(
                abs(pair[1][0] - pair[0][0])
                + abs(pair[1][1] - pair[0][1])))
        promoted = None
        seen_ranges = set()
        for (ua, va), (ub, vb) in endpoint_pairs:
            if ub < ua:
                ua, ub, va, vb = ub, ua, vb, va
            if abs(ub - ua) <= uv_eps or abs(vb - va) <= uv_eps:
                continue
            key = tuple(round(x, 15) for x in (ua, ub, va, vb))
            if key in seen_ranges:
                continue
            seen_ranges.add(key)
            if _overlap_mapping_is_identity(
                    C1_orig, C2_orig, (ua, ub), (va, vb), rational):
                promoted = (ua, ub, va, vb)
                break

        if promoted is not None:
            ua, ub, va, vb = promoted
            overlaps.append({
                "boundary_zeros": cls.boundary_zeros,
                "overlap_endpoints": cls.overlap_endpoints,
                "u_range": (float(ua), float(ub)),
                "v_range": (float(va), float(vb)),
                "certification": "exact",
            })
            overlap_found = True

    # Ledger L47 (user decision 2026-07-12): residual-certified overlap tier.
    # Coincidence at modeling tolerance is a real overlap even when no exact
    # affine identity exists — near-coincident pairs (imported geometry) and
    # same-locus non-affine reparameterizations. The certificate inverts
    # dense samples across a domain-end-pinned span (residual <= atol,
    # monotone pairing) and REFUSES to merge crossing structure inside the
    # band (transverse-direction flips between gap-scale samples),
    # returning those brackets for strict isolated-root certification.
    residual_band_evidence = False
    uncertified_span_evidence = None
    interior_bracket_hits = []
    if not overlap_found and cells.remaining > 0:
        cells.spend(1)
        tol_overlap, tol_brackets, residual_band_evidence, \
            uncertified_span_evidence = _tolerance_overlap_certificate(
                C1_orig, C2_orig, atol, rational, ptol_u, ptol_v)
        if tol_overlap is not None:
            overlaps.append(tol_overlap)
            overlap_found = True
        for u_seed, v_seed in tol_brackets:
            polished = _strict_polish_ccx(
                C1_orig, C2_orig, u_seed, v_seed, rational,
                component_scale=component_scale)
            if polished is None:
                continue
            u_sol, v_sol, point = polished
            if overlap_found:
                lo, hi = overlaps[-1]["u_range"]
                if min(lo, hi) - ptol_u <= u_sol <= max(lo, hi) + ptol_u:
                    continue
            interior_bracket_hits.append(
                (float(u_sol), float(v_sol), point))

    # A tolerant OVERLAP classification (or residual-gate coincidence
    # evidence) whose certificates all failed is often a broad, near-zero
    # distance valley.  An unrestricted Phase-2 subdivision of that valley
    # is both expensive and incapable of turning the rejected candidate
    # into a sound public overlap.  Give the isolated-root fallback a
    # separate, bounded allowance and report a partial result if that
    # allowance cannot certify the remaining cells.
    non_affine_overlap_fallback = (
        (cls.kind == OVERLAP or residual_band_evidence)
        and not overlap_found)
    non_affine_overlap_cells_remaining = (
        cells.tier(2_000)
        if non_affine_overlap_fallback else None
    )

    def _finalize(topology_complete=True):
        # Typed L47 outcome, mirroring CSX's L42 export: when the overlap-
        # class structure could not be certified AND the bounded fallback
        # could not discretize it, name the span instead of billing the
        # failure to the budget with topology claimed complete.
        structural = (non_affine_overlap_fallback and budget_exhausted
                      and not overlap_found)
        res = _result(isolated, overlaps,
                      topology_complete=topology_complete and not structural)
        if structural:
            span = uncertified_span_evidence or (0.0, 1.0)
            res["uncertified_overlap_span"] = (
                float(span[0]), float(span[1]))
        return res

    # 1c. Classify boundary hits: overlap endpoints go into the overlap,
    #     remaining boundary hits become isolated intersections.
    if overlap_found and overlaps:
        ovl = overlaps[-1]
        u_start_ovl, u_end_ovl = ovl["u_range"]
        v_start_ovl, v_end_ovl = ovl["v_range"]
        u_lo_ovl = min(u_start_ovl, u_end_ovl)
        u_hi_ovl = max(u_start_ovl, u_end_ovl)
        for u_bz, v_bz, pt in boundary_hits:
            on_overlap = False
            if u_lo_ovl - ptol_u <= u_bz <= u_hi_ovl + ptol_u:
                lam = ((u_bz - u_start_ovl)
                       / (u_end_ovl - u_start_ovl))
                v_expected = ((1.0 - lam) * v_start_ovl
                              + lam * v_end_ovl)
                if ovl.get("certification") == "tolerance":
                    # The pairing of a residual-certified overlap need not
                    # be affine — locate the expected partner parameter by
                    # inversion (seeded by the affine guess).
                    v_expected, _r_inv = _invert_point_on_curve(
                        C2_orig,
                        eval_curve(C1_orig, float(u_bz), rational=rational),
                        v_expected, rational)
                on_overlap = abs(v_bz - v_expected) <= 2.0 * ptol_v
            if not on_overlap:
                if not _is_duplicate(isolated, pt, atol):
                    if len(isolated) >= max_results:
                        budget_exhausted = True
                        break
                    isolated.append({"u": u_bz, "v": v_bz, "point": pt})
    else:
        for u_bz, v_bz, pt in boundary_hits:
            if not _is_duplicate(isolated, pt, atol):
                if len(isolated) >= max_results:
                    budget_exhausted = True
                    break
                isolated.append({"u": u_bz, "v": v_bz, "point": pt})

    # Interior crossings certified from the residual tier's rejected
    # brackets (crossing structure inside a tolerance band is topology,
    # never merged — CSX invariant, 1-D form).
    for u_hit, v_hit, pt in interior_bracket_hits:
        if not _is_duplicate(isolated, pt, atol):
            if len(isolated) >= max_results:
                budget_exhausted = True
                break
            isolated.append({"u": u_hit, "v": v_hit, "point": pt})

    if budget_exhausted:
        return _finalize()

    # ===================================================================
    # PHASE 2: Isolated intersection search
    # ===================================================================
    # Cut from the FIRST curve's parameter (u) only — same principle as
    # CSX cutting along t. For each remaining u-interval, search against
    # the FULL second curve [0,1].

    # Collect u-intervals to exclude
    u_exclude = []
    if overlap_found and overlaps:
        ovl = overlaps[-1]
        u_lo_ovl = min(ovl["u_range"])
        u_hi_ovl = max(ovl["u_range"])
        u_exclude.append((u_lo_ovl - ptol_u, u_hi_ovl + ptol_u))
    for iso in isolated:
        u_exclude.append((iso["u"] - ptol_u, iso["u"] + ptol_u))


    u_intervals = _compute_remaining_intervals(u_exclude, 0.0, 1.0)

    for u_lo, u_hi in u_intervals:
        if u_hi - u_lo < ptol_u * 0.1:
            continue

        # Restrict net and first curve to this u sub-interval
        F_sub = _restrict_net_axis(F, 0, u_lo, u_hi, 0.0, 1.0)
        C1_sub = C1
        if u_lo > 1e-12:
            _, C1_sub = _subdivide_curve(C1_sub, u_lo)
        if u_hi < 1.0 - 1e-12:
            u_hi_rescaled = (u_hi - u_lo) / (1.0 - u_lo) if u_lo > 1e-12 else u_hi
            C1_sub, _ = _subdivide_curve(C1_sub, u_hi_rescaled)
        pw_sub = C1_sub[:, -1].copy() if rational else np.ones(C1_sub.shape[0])

        # Quick min-of-net check
        from mmcore.numeric.intersection._sq_dist_classify import (
            _check_min_of_net, _weight_max_product,
        )
        w_sc = _weight_max_product(pw_sub, Qw)
        if _check_min_of_net(F_sub, atol, w_sc):
            continue

        # Run Phase 2 on this sub-interval × full v
        if (cells.remaining <= 0 or len(isolated) >= max_results
                or (non_affine_overlap_fallback
                    and non_affine_overlap_cells_remaining <= 0)):
            budget_exhausted = True
            break
        phase2_cell_limit = cells.remaining
        if non_affine_overlap_fallback:
            phase2_cell_limit = min(
                phase2_cell_limit, non_affine_overlap_cells_remaining)
        phase2_iso, phase2_exhausted, cells_used = _phase2_ccx(
            F_sub, C1_sub, C2, C1_orig, C2_orig,
            u_lo, u_hi, 0.0, 1.0,
            atol, rational, ptol_u, ptol_v,
            known_points=isolated,
            max_depth=max_depth, max_cells=phase2_cell_limit,
            max_results=max_results - len(isolated),
        )
        cells.spend(cells_used)
        if non_affine_overlap_fallback:
            non_affine_overlap_cells_remaining -= cells_used
        budget_exhausted = budget_exhausted or phase2_exhausted

        for iso in phase2_iso:
            iso.pop('_micro', None)
            if not _is_duplicate(isolated, iso["point"], atol):
                isolated.append(iso)

        if non_affine_overlap_fallback and phase2_exhausted:
            break

    return _finalize()


def _is_duplicate(isolated, pt, atol):
    """Check if *pt* is within *atol* of any existing isolated point."""
    for entry in isolated:
        existing = np.asarray(entry["point"])
        if np.linalg.norm(existing - pt) < atol:
            return True
    return False
