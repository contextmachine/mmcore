"""Tests for mmcore.numeric.intersection._bezier_common shared utilities."""

import numpy as np
import pytest

from mmcore.numeric.intersection._bezier_common import (
    extract_weights,
    dehomogenize_ctrl,
    is_homogeneous,
    newton_ccx,
    newton_csx,
    eval_curve,
    eval_curve_d1,
    eval_surface,
    eval_surface_d1,
)


# ---------------------------------------------------------------------------
# extract_weights
# ---------------------------------------------------------------------------

def test_extract_weights_rational():
    ctrl = np.array([[1.0, 0.0, 0.0, 1.0],
                     [0.707, 0.707, 0.0, 0.707],
                     [0.0, 1.0, 0.0, 1.0]])
    xyz, w = extract_weights(ctrl, rational=True)
    assert xyz.shape == (3, 3)
    assert w.shape == (3,)
    np.testing.assert_allclose(w, [1.0, 0.707, 1.0])


def test_extract_weights_polynomial():
    ctrl = np.array([[0.0, 0.0, 0.0],
                     [1.0, 1.0, 0.0],
                     [2.0, 0.0, 0.0]])
    xyz, w = extract_weights(ctrl, rational=False)
    assert xyz.shape == (3, 3)
    np.testing.assert_allclose(w, [1.0, 1.0, 1.0])


def test_extract_weights_surface_rational():
    ctrl = np.random.rand(3, 4, 4)
    xyz, w = extract_weights(ctrl, rational=True)
    assert xyz.shape == (3, 4, 3)
    assert w.shape == (3, 4)


# ---------------------------------------------------------------------------
# dehomogenize_ctrl
# ---------------------------------------------------------------------------

def test_dehomogenize():
    ctrl_h = np.array([[2.0, 0.0, 0.0, 2.0],
                       [0.0, 1.0, 0.0, 1.0]])
    result = dehomogenize_ctrl(ctrl_h)
    np.testing.assert_allclose(result, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


# ---------------------------------------------------------------------------
# is_homogeneous
# ---------------------------------------------------------------------------

def test_is_homogeneous_forced_true():
    ctrl = np.array([[0.0, 0.0, 0.0]])
    assert is_homogeneous(ctrl, rational=True) is True


def test_is_homogeneous_forced_false():
    ctrl = np.array([[0.0, 0.0, 0.0, 1.0]])
    assert is_homogeneous(ctrl, rational=False) is False


def test_is_homogeneous_auto_4d():
    ctrl = np.array([[1.0, 0.0, 0.0, 1.0],
                     [0.0, 1.0, 0.0, 1.0]])
    assert is_homogeneous(ctrl) is True


def test_is_homogeneous_auto_3d_polynomial():
    ctrl = np.array([[0.0, 0.0, 0.0],
                     [1.0, 1.0, 0.0]])
    assert is_homogeneous(ctrl) is False


# ---------------------------------------------------------------------------
# eval_curve  (polynomial and rational)
# ---------------------------------------------------------------------------

def test_eval_curve_polynomial_endpoints():
    C = np.array([[0.0, 0.0, 0.0],
                  [1.0, 1.0, 0.0],
                  [2.0, 0.0, 0.0]])
    np.testing.assert_allclose(eval_curve(C, 0.0, rational=False), [0.0, 0.0, 0.0], atol=1e-14)
    np.testing.assert_allclose(eval_curve(C, 1.0, rational=False), [2.0, 0.0, 0.0], atol=1e-14)


def test_eval_curve_rational():
    # Quarter-circle arc control points (weighted)
    C = np.array([[1.0, 0.0, 0.0, 1.0],
                  [0.707106781, 0.707106781, 0.0, 0.707106781],
                  [0.0, 1.0, 0.0, 1.0]])
    pt = eval_curve(C, 0.5, rational=True)
    assert pt.shape == (3,)
    # Mid-parameter on rational quarter circle should be near (cos(45), sin(45), 0)
    np.testing.assert_allclose(np.linalg.norm(pt[:2]), 1.0, atol=1e-3)


# ---------------------------------------------------------------------------
# eval_curve_d1
# ---------------------------------------------------------------------------

def test_eval_curve_d1_polynomial():
    C = np.array([[0.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0]])  # linear curve
    pt, d1 = eval_curve_d1(C, 0.5, rational=False)
    np.testing.assert_allclose(pt, [0.5, 0.0, 0.0], atol=1e-14)
    np.testing.assert_allclose(d1, [1.0, 0.0, 0.0], atol=1e-12)


# ---------------------------------------------------------------------------
# eval_surface / eval_surface_d1
# ---------------------------------------------------------------------------

def test_eval_surface_bilinear():
    # Bilinear patch: (2,2,3) control net
    S = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                  [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]], dtype=np.float64)
    pt = eval_surface(S, 0.5, 0.5, rational=False)
    np.testing.assert_allclose(pt, [0.5, 0.5, 0.0], atol=1e-14)


def test_eval_surface_d1_bilinear():
    S = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                  [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]], dtype=np.float64)
    pt, du, dv = eval_surface_d1(S, 0.5, 0.5, rational=False)
    np.testing.assert_allclose(pt, [0.5, 0.5, 0.0], atol=1e-14)
    # du should be [0, 1, 0] and dv should be [1, 0, 0] for this bilinear patch
    np.testing.assert_allclose(du, [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(dv, [1.0, 0.0, 0.0], atol=1e-12)


# ---------------------------------------------------------------------------
# newton_ccx
# ---------------------------------------------------------------------------

def test_newton_ccx_transversal():
    C1 = np.array([[0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [2.0, 0.0, 0.0]])
    C2 = np.array([[0.0, 0.5, 0.0],
                    [1.0, -0.5, 0.0],
                    [2.0, 0.5, 0.0]])
    u, v, G, last_step = newton_ccx(C1, C2, 0.3, 0.3, rational=False)
    # Check residual — Newton converges well for this transversal case
    assert np.linalg.norm(G) < 1e-7
    # Last step should be tiny (converged)
    assert abs(last_step[0]) < 1e-10 and abs(last_step[1]) < 1e-10
    pt1 = eval_curve(C1, u, rational=False)
    pt2 = eval_curve(C2, v, rational=False)
    np.testing.assert_allclose(pt1, pt2, atol=1e-7)


def test_newton_ccx_no_intersection():
    C1 = np.array([[0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0]])
    C2 = np.array([[0.0, 10.0, 0.0],
                    [1.0, 10.0, 0.0]])
    u, v, G, last_step = newton_ccx(C1, C2, 0.5, 0.5, rational=False)
    # Curves are far apart — residual should be large
    assert np.linalg.norm(G) > 1.0


# ---------------------------------------------------------------------------
# newton_csx
# ---------------------------------------------------------------------------

def test_newton_csx_transversal():
    # Line curve along x-axis at y=0.5
    C = np.array([[0.0, 0.5, 0.0],
                  [1.0, 0.5, 0.0]])
    # Bilinear patch on the xy-plane
    S = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                  [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]], dtype=np.float64)
    t, u, v, G, last_step = newton_csx(C, S, 0.5, 0.5, 0.5, rational=False)
    assert np.linalg.norm(G) < 1e-7
    pt_c = eval_curve(C, t, rational=False)
    pt_s = eval_surface(S, u, v, rational=False)
    np.testing.assert_allclose(pt_c, pt_s, atol=1e-7)


def test_newton_csx_no_intersection():
    # Line curve far from the surface
    C = np.array([[0.0, 0.0, 10.0],
                  [1.0, 0.0, 10.0]])
    S = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                  [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]], dtype=np.float64)
    t, u, v, G, last_step = newton_csx(C, S, 0.5, 0.5, 0.5, rational=False)
    assert np.linalg.norm(G) > 1.0


# --- bernstein_product_1d (ledger L52 slice 6a: the shared exact product) ---

def _reference_product(a, b):
    """The ccx-arithmetic reference: all factor terms in longdouble."""
    import math as _math
    a = np.asarray(a, dtype=np.longdouble)
    b = np.asarray(b, dtype=np.longdouble)
    p = a.shape[0] - 1
    q = b.shape[0] - 1
    out = np.zeros((p + q + 1,) + np.broadcast_shapes(
        a.shape[1:], b.shape[1:]), dtype=np.longdouble)
    for i in range(p + 1):
        for j in range(q + 1):
            k = i + j
            factor = (np.longdouble(_math.comb(p, i))
                      * np.longdouble(_math.comb(q, j))
                      / np.longdouble(_math.comb(p + q, k)))
            out[k] += factor * a[i] * b[j]
    return out


def test_bernstein_product_1d_scalar_operands_bit_identical_to_reference():
    from mmcore.numeric.intersection._bezier_common import bernstein_product_1d

    rng = np.random.default_rng(7)
    for p, q in ((1, 1), (2, 3), (3, 5), (7, 2)):
        a = rng.standard_normal(p + 1)
        b = rng.standard_normal(q + 1)
        got = bernstein_product_1d(a, b)
        ref = _reference_product(a, b)
        # exact equality: same accumulation order, same longdouble factor
        # arithmetic — certificate envelopes compare against eps-scale
        # bounds, so a last-ulp drift here is a behavior change.
        assert got.dtype == np.longdouble
        assert np.array_equal(got, ref)


def test_bernstein_product_1d_broadcasts_trailing_value_axes():
    from mmcore.numeric.intersection._bezier_common import bernstein_product_1d

    rng = np.random.default_rng(11)
    A = rng.standard_normal((4, 3))     # degree-3 curve, 3 components
    B = rng.standard_normal((3, 1))     # degree-2 weights column
    got = bernstein_product_1d(A, B)
    assert got.shape == (6, 3)
    assert np.array_equal(got, _reference_product(A, B))


def test_bernstein_product_1d_delegations_are_the_same_object():
    from mmcore.numeric.intersection._bezier_common import bernstein_product_1d
    from mmcore.numeric.intersection.ccx import _bez_ccx4
    from mmcore.numeric.intersection.csx import _bez_csx4

    assert _bez_ccx4._bernstein_product_1d is bernstein_product_1d
    assert _bez_csx4._bernstein_product_1d is bernstein_product_1d
