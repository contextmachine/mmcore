"""Known factored polynomials check the numerical isolation independently."""
from math import comb

import numpy as np
import pytest

from mmcore.numeric._bern_roots import bernstein_roots


def _from_roots(roots):
    power = np.polynomial.polynomial.polyfromroots(roots)
    degree = len(power)-1
    return np.array([sum(power[j]*comb(i, j)/comb(degree, j)
                         for j in range(i+1)) for i in range(degree+1)])


@pytest.mark.parametrize('degree', [2, 4, 8, 10, 12])
def test_flat_multiple_root_is_one_location(degree):
    net = np.array([(-1.)**(degree-i)*2.**-degree for i in range(degree+1)])
    result = bernstein_roots(net)
    assert result.complete and not result.constant_zero
    np.testing.assert_allclose(result.roots, [.5], atol=1e-12, rtol=0)
    assert result.work < 1000


@pytest.mark.parametrize('roots', [[.49, .4905], [0., .31, .79, 1.], [.2, .2, .7, .7]])
def test_distinct_components_and_endpoint_roots_are_retained(roots):
    result = bernstein_roots(_from_roots(roots))
    assert result.complete
    np.testing.assert_allclose(result.roots, sorted(set(roots)), atol=1e-9, rtol=0)


def test_nonbinary_double_root_and_scaled_coefficients():
    for factor in (1e-20, 1., 1e20):
        result = bernstein_roots(factor*np.array([1., -2., 4.]))
        np.testing.assert_allclose(result.roots, [1/3], atol=1e-12, rtol=0)


def test_positive_gap_is_not_a_cloud_of_small_residual_roots():
    result = bernstein_roots(np.array([.25, -.25, .25])+1e-8)
    assert result.complete and result.roots == () and not result.constant_zero


def test_shallow_extremum_cannot_replace_two_sign_changing_roots():
    gap = 2.**-50
    result = bernstein_roots(np.array([.25, -.25, .25])-gap,
                             parameter_tol=1e-14)
    roots = np.array([.5-2.**-25, .5+2.**-25])
    # A stretched CAD chart maps this small parameter gap to 0.06 units.
    assert len(result.roots) == 2
    np.testing.assert_allclose(1e6*np.array(result.roots), 1e6*roots,
                               atol=1e-3, rtol=0)


def test_constant_and_work_limit_are_distinguished():
    assert bernstein_roots([0., 0.]).constant_zero
    assert bernstein_roots([1e-30, 1e-30]).roots == ()
    accepted = []
    def charge(unit):
        if len(accepted) == 3:
            return False
        accepted.append(unit)
        return True
    result = bernstein_roots(_from_roots([.2, .7]), charge=charge)
    assert not result.complete and result.work == len(accepted) == 3
