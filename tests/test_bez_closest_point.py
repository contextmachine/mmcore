# tests/test_bez_closest_point.py
import numpy as np
from math import comb
from mmcore.numeric._bez_closest_point import _bernstein_product_nd


def _bern_eval_1d(coeffs, t):
    n = len(coeffs) - 1
    B = np.array([comb(n, i) * t**i * (1 - t) ** (n - i) for i in range(n + 1)])
    return float(B @ coeffs)


def _bern_eval_2d(net, u, v):
    m, n = net.shape[0] - 1, net.shape[1] - 1
    Bu = np.array([comb(m, i) * u**i * (1 - u) ** (m - i) for i in range(m + 1)])
    Bv = np.array([comb(n, j) * v**j * (1 - v) ** (n - j) for j in range(n + 1)])
    return float(Bu @ net @ Bv)


def test_bernstein_product_1d_matches_pointwise():
    a = np.array([1.0, -2.0, 3.0])      # degree 2
    b = np.array([0.5, 4.0])            # degree 1
    c = _bernstein_product_nd(a, b)     # degree 3
    assert c.shape == (4,)
    for t in np.linspace(0, 1, 11):
        assert abs(_bern_eval_1d(c, t) - _bern_eval_1d(a, t) * _bern_eval_1d(b, t)) < 1e-12


def test_bernstein_product_2d_matches_pointwise():
    a = np.array([[1.0, 2.0], [3.0, -1.0], [0.0, 2.0]])   # bidegree (2,1)
    b = np.array([[1.0, 0.5], [-1.0, 2.0]])               # bidegree (1,1)
    c = _bernstein_product_nd(a, b)                        # bidegree (3,2)
    assert c.shape == (4, 3)
    for u in (0.0, 0.25, 0.5, 0.9, 1.0):
        for v in (0.0, 0.3, 1.0):
            assert abs(_bern_eval_2d(c, u, v) - _bern_eval_2d(a, u, v) * _bern_eval_2d(b, u, v)) < 1e-12
