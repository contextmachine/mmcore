import numpy as np
import pytest

from mmcore.numeric.intersection.ccx._bez_ccx3 import bez_ccx


# Geometry fixtures replicated from the demo cases in _bez_ccx3.py
curve1 = np.array(
    [
        [-19.77608536, 23.10065701, 0.0],
        [-14.86834768, 28.69713066, 0.0],
        [-5.8568525, 25.12677787, 0.0],
        [-12.62581769, 15.26478654, 0.0],
    ]
)
curve2 = np.array(
    [
        [-22.0315362, 18.75969713, 0.0],
        [-19.42270945, 28.2502867, 0.0],
        [-8.46791623, 27.56878356, 0.0],
        [-10.43007782, 19.78973126, 0.0],
    ]
)
curve3 = np.array(
    [
        [-28.46565557, -11.09883504, 0.0],
        [-31.79098016, 13.62423043, 0.0],
        [-12.99566723, 16.66039636, 0.0],
        [8.11291498, -6.32771715, 0.0],
    ]
)
curve4 = np.array(
    [
        [-45.36434109, -7.12015504, 0.0],
        [-25.49612403, 13.94186047, 0.0],
        [-2.13178295, -17.35271318, 0.0],
        [12.02325581, 20.42248062, 0.0],
    ]
)
curve6 = np.array([[-13.12449258, 9.10030377, 0.0], [-27.74989311, 10.37986052, 0.0], [-29.02944985, -4.24554001, 0.0]])

curve7 = np.array([[0, 0, 0.0], [1, 0, 0.0], [2, 0, 0.0]], float)
curve8 = np.array([[0, 0, 0.0], [1, 1, 0.0], [2, 0, 0.0]], float)

# Rational cases
w_mid = np.sqrt(0.5)
curve_arc_h = np.array([[1.0, 0.0, 1.0], [w_mid, w_mid, w_mid], [0.0, 1.0, 1.0]])
curve_line_h = np.array([[0.0, 0.0, 1.0], [0.5, 0.5, 1.0], [1.0, 1.0, 1.0]])

elliptical_arc_pts_h = np.array(
    [
        [35.633, 61.37, 0.0, 1.0],
        [68.303977, 7.3528, 0.0, 0.707],
        [129.681, 49.963, 0.0, 1.0],
    ]
)
arc_pts2_h = np.array(
    [
        [88.731, 34.508, 0.0, 1.0],
        [52.914001, 42.130837, 0.0, 0.707],
        [49.76, 45.703, 0.0, 1.0],
    ]
)


def _uv_pairs(result):
    return [(float(p['u']), float(p['v'])) for p in sorted(result, key=lambda t: float(t['u']))]


def test_case1_overlap_endpoints():
    res = bez_ccx(curve1, curve2)
    assert res["isolated"] == []
    assert len(res["overlaps"]) == 1
    uv0, uv1 = res["overlaps"][0]["uv_path"][0], res["overlaps"][0]["uv_path"][-1]
    assert np.allclose(uv0, [0.0, 0.19069075484144143])
    assert np.allclose(uv1, [0.8275977622022961, 1.0])


def test_case2_two_isolated():
    res = bez_ccx(curve3, curve4)
    expected = [(0.19649579172632328, 0.2818845674995799), (0.84621222743306, 0.726646442488876)]
    assert len(res["isolated"]) == 2
    got = _uv_pairs(res["isolated"])
    for g, e in zip(got, expected):
        assert np.allclose(g, e)
    assert res["overlaps"] == []


def test_case3_1_two_isolated():
    res = bez_ccx(curve3, curve6)
    expected = [(0.1900373921622664, 0.8152057754324462), (0.577972765192217, 0.09892057701076915)]
    got = _uv_pairs(res["isolated"])
    assert len(got) == 2
    for g, e in zip(got, expected):
        assert np.allclose(g, e)
    assert res["overlaps"] == []


def test_case3_2_two_isolated():
    res = bez_ccx(curve6, curve3)
    expected = [(0.09892057701076917, 0.577972765192217), (0.8152057754324554, 0.19003739216226206)]
    got = _uv_pairs(res["isolated"])
    assert len(got) == 2
    for g, e in zip(got, expected):
        assert np.allclose(g, e)
    assert res["overlaps"] == []


def test_case4_boundary_hits():
    res = bez_ccx(curve7, curve8)
    expected = [(0.0, 0.0), (1.0, 1.0)]
    got = _uv_pairs(res["isolated"])
    assert len(got) == 2
    for g, e in zip(got, expected):
        assert np.allclose(g, e, atol=1e-12)
    assert res["overlaps"] == []


def test_case5_rational_quarter_circle():
    res = bez_ccx(curve_arc_h, curve_line_h, rational=True)
    assert len(res["isolated"]) == 1
    u, v = res["isolated"][0]["u"], res["isolated"][0]["v"]
    assert np.allclose((u, v), (0.5, np.sqrt(0.5)))
    assert res["overlaps"] == []


def test_case6_rational_elliptic_pair():
    res = bez_ccx(elliptical_arc_pts_h, arc_pts2_h, rational=True)
    expected = [(0.20202602329720454, 0.8588967831366813), (0.4880530373099784, 0.07360787579646323)]
    got = _uv_pairs(res["isolated"])
    assert len(got) == 2
    for g, e in zip(got, expected):
        assert np.allclose(g, e)
    assert res["overlaps"] == []
