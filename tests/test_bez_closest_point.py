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


def _eval_bern_1d(coeffs, t):
    n = len(coeffs) - 1
    B = np.array([comb(n, i) * t**i * (1 - t) ** (n - i) for i in range(n + 1)])
    return float(B @ coeffs)


def _eval_bern_2d(net, u, v):
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


# tests/test_bez_closest_point.py  (append)
from mmcore.numeric._bez_closest_point import (
    point_curve_stationarity_net,
    point_surface_stationarity_nets,
)
from mmcore.numeric import bern_sq_dist


def _g_curve(F, Qw, t):
    return bern_sq_dist.eval_point_curve_distance_sq(F, Qw, t)


def test_curve_stationarity_net_nonrational_is_Fprime():
    # Quadratic non-rational curve
    C = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 0.0, 0.0]])
    P = np.array([1.0, -1.0, 0.0])
    N, F, Qw = point_curve_stationarity_net(P, C, rational=False)
    # N must change sign at the same t where d/dt ||P-C(t)||^2 = 0.
    # Find a sign change of N by sampling and confirm g has a stationary point there.
    ts = np.linspace(0, 1, 401)
    Nvals = np.array([_eval_bern_1d(N, t) for t in ts])
    sign_changes = np.where(np.sign(Nvals[:-1]) != np.sign(Nvals[1:]))[0]
    assert len(sign_changes) >= 1
    # At each sign change, g'(t) computed by finite difference is ~0
    for k in sign_changes:
        t0 = ts[k]
        h = 1e-5
        gp = (_g_curve(F, Qw, min(1, t0 + h)) - _g_curve(F, Qw, max(0, t0 - h))) / (2 * h)
        assert abs(gp) < 1e-1  # near-zero at the bracketed root


def test_curve_stationarity_net_rational_tracks_true_derivative():
    # Rational quadratic quarter circle in xy-plane
    s = np.sqrt(2) / 2
    C = np.array([[1.0, 0.0, 0.0, 1.0],
                  [s, s, 0.0, s],     # homogeneous: (x*w, y*w, z*w, w)
                  [0.0, 1.0, 0.0, 1.0]])
    P = np.array([0.6, 0.6, 0.0])
    N, F, Qw = point_curve_stationarity_net(P, C, rational=True)
    # Sign change of N must bracket a stationary point of the TRUE distance.
    ts = np.linspace(0, 1, 801)
    Nvals = np.array([_eval_bern_1d(N, t) for t in ts])
    sc = np.where(np.sign(Nvals[:-1]) != np.sign(Nvals[1:]))[0]
    assert len(sc) >= 1
    for k in sc:
        t0 = ts[k]
        h = 1e-5
        gp = (_g_curve(F, Qw, min(1, t0 + h)) - _g_curve(F, Qw, max(0, t0 - h))) / (2 * h)
        assert abs(gp) < 1e-2


def test_surface_stationarity_nets_nonrational_are_partials():
    # Bilinear non-rational patch (unit square, z=0)
    S = np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                  [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]])
    P = np.array([0.3, 0.4, 5.0])
    Nu, Nv, F, Sw = point_surface_stationarity_nets(P, S, rational=False)
    # Gradient of g vanishes at (u,v)=(0.3,0.4): both nets bracket zero there.
    assert _eval_bern_2d(Nu, 0.3, 0.4) == 0.0 or abs(_eval_bern_2d(Nu, 0.3, 0.4)) < 1e-9
    assert abs(_eval_bern_2d(Nv, 0.3, 0.4)) < 1e-9
