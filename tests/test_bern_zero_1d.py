"""Tests for the 1D Bernstein zero-finder."""
import numpy as np
import pytest

from mmcore.numeric._bern_zero_1d import (
    find_bernstein_zeros_1d,
    _count_sign_changes,
    _bernstein_deriv_coeffs_1d,
    _de_casteljau_eval_1d,
    _longest_positive_run_center,
)


def test_count_sign_changes_basic():
    assert _count_sign_changes(np.array([1.0, -0.3, 0.7])) == 2
    assert _count_sign_changes(np.array([1.0, 2.0, 3.0])) == 0
    assert _count_sign_changes(np.array([-1.0, 1.0])) == 1
    assert _count_sign_changes(np.array([1.0, 0.0, -1.0])) == 1  # zero skipped


def test_count_sign_changes_all_positive():
    assert _count_sign_changes(np.array([0.1, 0.5, 0.3, 0.8])) == 0


def test_deriv_coeffs():
    # Linear: f(t) = (1-t)*1 + t*3 = 1 + 2t → f'(t) = 2
    d = _bernstein_deriv_coeffs_1d(np.array([1.0, 3.0]))
    np.testing.assert_allclose(d, [2.0])

    # Quadratic: f(t) = sum B_{i,2}*c_i → f'(t) = 2*sum B_{i,1}*diff(c)
    d = _bernstein_deriv_coeffs_1d(np.array([0.0, 1.0, 0.0]))
    np.testing.assert_allclose(d, [2.0, -2.0])


def test_eval_1d():
    # Linear: f(t) = (1-t)*0 + t*1 = t
    assert abs(_de_casteljau_eval_1d(np.array([0.0, 1.0]), 0.5) - 0.5) < 1e-14
    assert abs(_de_casteljau_eval_1d(np.array([0.0, 1.0]), 0.0) - 0.0) < 1e-14
    assert abs(_de_casteljau_eval_1d(np.array([0.0, 1.0]), 1.0) - 1.0) < 1e-14


def test_longest_positive_run():
    coeffs = np.array([0.1, -0.02, 0.3, 0.8, 0.9, -0.01, 0.2])
    t = _longest_positive_run_center(coeffs)
    # Longest positive run is indices 2,3,4 → center at (2+4)/2 = 3.5 → 3.5/6 ≈ 0.583
    assert abs(t - 3.5 / 6) < 0.01


def test_longest_positive_run_all_negative():
    coeffs = np.array([-1.0, -2.0, -3.0])
    t = _longest_positive_run_center(coeffs)
    assert t == 0.5  # fallback


def test_no_zeros_all_positive():
    """All coefficients well above zero → no zeros."""
    coeffs = np.array([5.0, 3.0, 4.0, 6.0])
    zeros = find_bernstein_zeros_1d(coeffs, atol=1e-3)
    assert len(zeros) == 0


def test_zero_at_endpoint_t0():
    """Coefficient at t=0 is zero."""
    coeffs = np.array([0.0, 1.0, 2.0])
    zeros = find_bernstein_zeros_1d(coeffs, atol=1e-3)
    assert any(abs(z) < 1e-6 for z in zeros)


def test_zero_at_endpoint_t1():
    """Coefficient at t=1 is zero."""
    coeffs = np.array([2.0, 1.0, 0.0])
    zeros = find_bernstein_zeros_1d(coeffs, atol=1e-3)
    assert any(abs(z - 1.0) < 1e-6 for z in zeros)


def test_interior_minimum_touching_zero():
    """Quadratic with minimum at t=0.5 touching zero: f(t) = 4(t-0.5)^2.

    In Bernstein form for degree 2: coeffs = [1, 0, 1]
    f(0.5) = 0.25*1 + 0.5*0 + 0.25*1 = 0.5 ... that's not zero.

    Use f(t) = (2t-1)^2 = 4t^2 - 4t + 1.
    Bernstein degree 2: c0 = f(0) = 1, c2 = f(1) = 1
    c1 = (f(0) + f(1))/2 - f''/(2*2!) ... let's just compute directly.
    f(t) = sum B_{i,2}(t) * c_i where c_i = f(i/2)
    c0 = f(0) = 1, c1 = f(0.5) = 0, c2 = f(1) = 1.
    """
    # f(t) = (2t-1)^2 in Bernstein form degree 2: [1, -1, 1]
    # (coeffs[1] is negative even though f >= 0 — normal for Bernstein)
    coeffs = np.array([1.0, -1.0, 1.0])
    zeros = find_bernstein_zeros_1d(coeffs, atol=0.1)
    # f(0.5) = 0, which is < atol^2 = 0.01
    assert len(zeros) >= 1
    assert any(abs(z - 0.5) < 0.05 for z in zeros)


def test_overlap_case_boundary_u0():
    """The actual overlap case from the benchmark: F|_{u=0} has a root at v~0.191."""
    from mmcore.numeric.bern_sq_dist import curve_curve_squared_net_homog
    from mmcore.numeric.bern import bernstein_boundary_nd

    C1 = np.array([[-19.77608536, 23.10065701, 0.], [-14.86834768, 28.69713066, 0.],
                   [-5.8568525, 25.12677787, 0.], [-12.62581769, 15.26478654, 0.]])
    C2 = np.array([[-22.0315362, 18.75969713, 0.], [-19.42270945, 28.2502867, 0.],
                   [-8.46791623, 27.56878356, 0.], [-10.43007782, 19.78973126, 0.]])

    F = curve_curve_squared_net_homog(C1, C2, rational=False)
    F_dim = F[..., np.newaxis]

    # Boundary at u=0
    bnd_u0 = bernstein_boundary_nd(F_dim, axis=0, side=0)
    bnd_u0_1d = np.squeeze(bnd_u0)

    zeros = find_bernstein_zeros_1d(bnd_u0_1d, atol=1e-3)
    assert len(zeros) >= 1
    # Should find root near v=0.191
    assert any(abs(z - 0.19069) < 0.01 for z in zeros), f"Expected root near 0.191, got {zeros}"


def test_overlap_case_boundary_v1():
    """The actual overlap case: F|_{v=1} has a root at u~0.828."""
    from mmcore.numeric.bern_sq_dist import curve_curve_squared_net_homog
    from mmcore.numeric.bern import bernstein_boundary_nd

    C1 = np.array([[-19.77608536, 23.10065701, 0.], [-14.86834768, 28.69713066, 0.],
                   [-5.8568525, 25.12677787, 0.], [-12.62581769, 15.26478654, 0.]])
    C2 = np.array([[-22.0315362, 18.75969713, 0.], [-19.42270945, 28.2502867, 0.],
                   [-8.46791623, 27.56878356, 0.], [-10.43007782, 19.78973126, 0.]])

    F = curve_curve_squared_net_homog(C1, C2, rational=False)
    F_dim = F[..., np.newaxis]

    # Boundary at v=1
    bnd_v1 = bernstein_boundary_nd(F_dim, axis=1, side=1)
    bnd_v1_1d = np.squeeze(bnd_v1)

    zeros = find_bernstein_zeros_1d(bnd_v1_1d, atol=1e-3)
    assert len(zeros) >= 1
    # Should find root near u=0.828
    assert any(abs(z - 0.82760) < 0.01 for z in zeros), f"Expected root near 0.828, got {zeros}"
