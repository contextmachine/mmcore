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


# tests/test_bez_closest_point.py  (append)
from mmcore.numeric._bez_closest_point import eval_curve_d2, eval_surface_d2
from mmcore.numeric.intersection._bezier_common import eval_curve, eval_surface


def test_eval_curve_d2_matches_finite_difference():
    C = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 0.0, 0.0], [3.0, 1.0, 0.0]])
    t = 0.37
    pt, d1, d2 = eval_curve_d2(C, t, rational=False)
    h = 1e-5
    fd1 = (eval_curve(C, t + h, rational=False) - eval_curve(C, t - h, rational=False)) / (2 * h)
    fd2 = (eval_curve(C, t + h, rational=False) - 2 * eval_curve(C, t, rational=False)
           + eval_curve(C, t - h, rational=False)) / h**2
    assert np.allclose(d1, fd1, atol=1e-5)
    assert np.allclose(d2, fd2, atol=1e-3)


def test_eval_surface_d2_matches_finite_difference():
    S = np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.5]],
                  [[1.0, 0.0, 0.5], [1.0, 1.0, 0.0]]])
    u, v = 0.4, 0.6
    pt, Su, Sv, Suu, Suv, Svv = eval_surface_d2(S, u, v, rational=False)
    h = 1e-4
    fSuu = (eval_surface(S, u + h, v, rational=False) - 2 * eval_surface(S, u, v, rational=False)
            + eval_surface(S, u - h, v, rational=False)) / h**2
    fSuv = (eval_surface(S, u + h, v + h, rational=False) - eval_surface(S, u + h, v - h, rational=False)
            - eval_surface(S, u - h, v + h, rational=False) + eval_surface(S, u - h, v - h, rational=False)) / (4 * h**2)
    assert np.allclose(Suu, fSuu, atol=1e-2)
    assert np.allclose(Suv, fSuv, atol=1e-2)


# tests/test_bez_closest_point.py  (append)
from mmcore.numeric._bez_closest_point import (
    newton_curve_closest_point, newton_surface_closest_point,
)


def test_newton_curve_closest_point_segment():
    C = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])   # segment along x
    P = np.array([0.5, 1.0, 0.0])                       # foot at x=0.5 -> t=0.25
    u, R, sq, _ = newton_curve_closest_point(C, P, 0.6, rational=False)
    assert abs(u - 0.25) < 1e-9
    assert abs(sq - 1.0) < 1e-9


def test_newton_surface_closest_point_plane_bounded():
    S = np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                  [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]])  # unit square z=0
    P = np.array([0.3, 0.4, 7.0])
    u, v, R, step = newton_surface_closest_point(
        S, P, 0.5, 0.5, rational=False, bounds=(0.0, 1.0, 0.0, 1.0))
    assert abs(u - 0.3) < 1e-9 and abs(v - 0.4) < 1e-9
    # residual r = (<S-P,Su>, <S-P,Sv>) ~ 0
    assert np.linalg.norm(R) < 1e-7


def test_newton_surface_respects_cell_bounds():
    S = np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                  [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]])
    P = np.array([0.3, 0.4, 7.0])
    # True foot (0.3,0.4) lies OUTSIDE this cell; solver must stay inside.
    u, v, R, step = newton_surface_closest_point(
        S, P, 0.65, 0.65, rational=False, bounds=(0.6, 0.9, 0.6, 0.9))
    assert 0.6 - 1e-9 <= u <= 0.9 + 1e-9
    assert 0.6 - 1e-9 <= v <= 0.9 + 1e-9


# tests/test_bez_closest_point.py  (append)
from mmcore.numeric._bez_closest_point import bez_curve_closest_points


def _dense_min_curve(C, P, rational, nsamp=4000):
    ts = np.linspace(0, 1, nsamp)
    d = np.array([np.linalg.norm(eval_curve(C, t, rational=rational) - P) for t in ts])
    k = int(np.argmin(d))
    return ts[k], d[k]


def test_curve_closest_interior_min():
    C = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 0.0, 0.0]])
    P = np.array([1.0, -1.0, 0.0])
    res = bez_curve_closest_points(C, P, atol=1e-6, rational=False)
    assert len(res) >= 1
    assert res == sorted(res, key=lambda e: e["distance"])
    t_ref, d_ref = _dense_min_curve(C, P, rational=False)
    assert abs(res[0]["distance"] - d_ref) < 1e-3
    assert abs(res[0]["t"] - t_ref) < 1e-2


def test_curve_closest_boundary_min():
    C = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])  # straight along x
    P = np.array([-1.0, 1.0, 0.0])                                     # nearest is t=0 endpoint
    res = bez_curve_closest_points(C, P, atol=1e-6, rational=False)
    assert res[0]["kind"] == "boundary_min"
    assert abs(res[0]["t"]) < 1e-6
    assert abs(res[0]["distance"] - np.sqrt(2.0)) < 1e-6


def test_curve_closest_multiple_minima_U_shape():
    # Cubic "U": two arms -> a point inside has two local minima
    C = np.array([[-2.0, 2.0, 0.0], [-2.0, -3.0, 0.0], [2.0, -3.0, 0.0], [2.0, 2.0, 0.0]])
    P = np.array([0.0, 1.0, 0.0])
    res = bez_curve_closest_points(C, P, atol=1e-6, rational=False)
    minima = [e for e in res if e["kind"] == "min"]
    assert len(minima) >= 2  # both arms
    t_ref, d_ref = _dense_min_curve(C, P, rational=False)
    assert abs(res[0]["distance"] - d_ref) < 1e-3


# tests/test_bez_closest_point.py  (append)
from mmcore.numeric._bez_closest_point import bez_surface_closest_points


def _dense_min_surface(S, P, rational, n=200):
    us = np.linspace(0, 1, n)
    vs = np.linspace(0, 1, n)
    best = (None, None, np.inf)
    for u in us:
        for v in vs:
            d = np.linalg.norm(eval_surface(S, u, v, rational=rational) - P)
            if d < best[2]:
                best = (u, v, d)
    return best


def test_surface_closest_plane_interior():
    S = np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                  [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]])   # unit square z=0
    P = np.array([0.3, 0.4, 5.0])
    res = bez_surface_closest_points(S, P, atol=1e-6, rational=False)
    assert len(res) >= 1
    assert res == sorted(res, key=lambda e: e["distance"])
    assert abs(res[0]["u"] - 0.3) < 1e-4 and abs(res[0]["v"] - 0.4) < 1e-4
    assert abs(res[0]["distance"] - 5.0) < 1e-4
    assert res[0]["kind"] == "min"


def test_surface_closest_curved_patch_matches_dense_grid():
    # Non-planar biquadratic-ish patch (bilinear with a bump via z)
    S = np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
                  [[1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 2.0, 0.0]],
                  [[2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 2.0, 0.0]]])
    P = np.array([1.0, 1.0, 3.0])
    res = bez_surface_closest_points(S, P, atol=1e-6, rational=False)
    u_ref, v_ref, d_ref = _dense_min_surface(S, P, rational=False)
    assert abs(res[0]["distance"] - d_ref) < 5e-3
