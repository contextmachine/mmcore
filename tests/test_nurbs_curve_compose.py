"""Tests for compose_nurbs_curve_curve: exact reparameterization of a NURBS
curve C(s) by a 1-D NURBS parameter curve s = sigma(t), giving C(sigma(t)).

The oracle is direct evaluation: for many t, compose_nurbs_curve_curve(C, sigma)
evaluated at t must equal C evaluated at sigma(t).  Composition is exact, so the
agreement is to machine precision.
"""
import numpy as np
import pytest

from mmcore.geom._nurbs_eval import NURBSCurveTuple, evaluate_nurbs_curve
from mmcore.numeric.sbern import compose_nurbs_curve_curve, compose_curve_curve


def _max_compose_error(C, sigma, n=21):
    composed = compose_nurbs_curve_curve(C, sigma)
    t0, t1 = sigma.knot[sigma.order - 1], sigma.knot[-sigma.order]
    err = 0.0
    for t in np.linspace(t0, t1, n):
        got = np.asarray(evaluate_nurbs_curve(composed, float(t))["C"])
        s = float(evaluate_nurbs_curve(sigma, float(t))["C"][0])
        ref = np.asarray(evaluate_nurbs_curve(C, s)["C"])
        err = max(err, np.linalg.norm(got[:len(ref)] - ref))
    return composed, err


# --- spatial curves -------------------------------------------------------
C3_KNOT = NURBSCurveTuple(                       # 3D rational quadratic, 1 interior knot
    order=3,
    knot=np.array([0, 0, 0, 0.5, 1, 1, 1.0]),
    control_points=np.array([[0, 0, 0], [1, 2, 1], [2, 2, -1], [3, 0, 0.0]]),
    weights=np.array([1.0, 2.0, 0.5, 1.0]),
)
C2_BEZ = NURBSCurveTuple(                         # 2D rational cubic Bézier
    order=4,
    knot=np.array([0, 0, 0, 0, 1, 1, 1, 1.0]),
    control_points=np.array([[0.0, 0.0], [1.0, 3.0], [3.0, 3.0], [4.0, 0.0]]),
    weights=np.array([1.0, 2.0, 0.7, 1.0]),
)
C3_2KNOT = NURBSCurveTuple(                       # 3D non-rational, 2 interior knots
    order=3,
    knot=np.array([0, 0, 0, 0.33, 0.66, 1, 1, 1.0]),
    control_points=np.array([[0, 0, 0], [1, 1, 0], [2, -1, 1], [3, 1, 1], [4, 0, 0.0]]),
    weights=np.ones(5),
)

# --- parameter curves -----------------------------------------------------
SIG_RAT_KNOT = NURBSCurveTuple(                   # rational quadratic, own interior knot
    order=3,
    knot=np.array([0, 0, 0, 0.4, 1, 1, 1.0]),
    control_points=np.array([[0.0], [0.3], [0.6], [1.0]]),
    weights=np.array([1.0, 2.0, 0.5, 1.0]),
)
SIG_BEZ = NURBSCurveTuple(                        # single rational Bézier sigma
    order=3,
    knot=np.array([0, 0, 0, 1, 1, 1.0]),
    control_points=np.array([[0.0], [0.55], [1.0]]),
    weights=np.array([1.0, 1.5, 1.0]),
)
SIG_CUBIC_KNOT = NURBSCurveTuple(                 # cubic sigma with own interior knot
    order=4,
    knot=np.array([0, 0, 0, 0, 0.5, 1, 1, 1, 1.0]),
    control_points=np.array([[0.0], [0.1], [0.5], [0.85], [1.0]]),
    weights=np.ones(5),
)
SIG_SUBINTERVAL = NURBSCurveTuple(               # stays inside (0.2, 0.7), crosses 0.33 & 0.66
    order=3,
    knot=np.array([0, 0, 0, 1, 1, 1.0]),
    control_points=np.array([[0.2], [0.45], [0.7]]),
    weights=np.array([1.0, 2.0, 1.0]),
)
SIG_DECREASING = NURBSCurveTuple(                # reversed direction 1 -> 0
    order=3,
    knot=np.array([0, 0, 0, 0.5, 1, 1, 1.0]),
    control_points=np.array([[1.0], [0.6], [0.3], [0.0]]),
    weights=np.ones(4),
)


@pytest.mark.parametrize("name, C, sigma, expected_out_deg", [
    ("3D-rat-knot o sigma-rat-knot",       C3_KNOT,  SIG_RAT_KNOT,   4),
    ("2D-rat-bezier o sigma-bezier",       C2_BEZ,   SIG_BEZ,        6),
    ("3D-2knot o sigma-cubic-knot",        C3_2KNOT, SIG_CUBIC_KNOT, 6),
    ("3D-2knot o sigma-subinterval",       C3_2KNOT, SIG_SUBINTERVAL, 4),
    ("3D-rat-knot o sigma-decreasing",     C3_KNOT,  SIG_DECREASING, 4),
])
def test_composition_matches_direct_evaluation(name, C, sigma, expected_out_deg):
    composed, err = _max_compose_error(C, sigma)
    assert err < 1e-9, f"{name}: composed curve disagrees with C(sigma(t)), max_err={err:.2e}"
    # per-Bézier-piece degree is deg(C) * deg(sigma); merged degree matches that
    assert composed.order - 1 == expected_out_deg
    # output keeps the spatial dimension of C
    assert composed.control_points.shape[1] == C.control_points.shape[1]


def test_output_spans_sigma_domain():
    """The composed curve is parameterized over sigma's t-domain."""
    composed = compose_nurbs_curve_curve(C3_2KNOT, SIG_CUBIC_KNOT)
    assert composed.knot[composed.order - 1] == pytest.approx(0.0)
    assert composed.knot[-composed.order] == pytest.approx(1.0)


def test_kernel_is_dimension_general():
    """compose_curve_curve must accept a 2D (D=3 cols) homogeneous net as well as 3D (4)."""
    # 2D rational quadratic, homogeneous (x*w, y*w, w)
    spatial2d = np.array([[0.0, 0.0, 1.0], [1.0, 2.0, 2.0], [2.0, 0.0, 1.0]])
    param = np.array([[0.0, 1.0], [0.5, 1.0], [1.0, 1.0]])  # (s, w)
    out = compose_curve_curve(spatial2d, param)
    assert out.shape[1] == 3            # stays 2D homogeneous
    assert out.shape[0] - 1 == (3 - 1) * (3 - 1)  # n*p


# --- patch composers: dimension generality + corners ----------------------
from mmcore.numeric.sbern import compose_patch_curve, compose_patch_patch


def test_patch_curve_is_dimension_general():
    """compose_patch_curve accepts a 2D (D=2) homogeneous patch, not only 3D."""
    rs = np.random.RandomState(3)
    patch2d = rs.rand(3, 3, 3)                       # (m+1,n+1, x*w,y*w,w)
    patch2d[..., 2] = rs.rand(3, 3) + 0.5
    uvw = np.column_stack([rs.rand(4), rs.rand(4), rs.rand(4) + 0.5])
    out = compose_patch_curve(patch2d, uvw, curve_ctrl_homogeneous=False)
    assert out.shape[1] == 3                         # stays 2D homogeneous
    assert np.isfinite(out).all()


def test_patch_patch_corners_match_outer():
    """An identity bilinear parameter patch reproduces the outer patch corners."""
    rs = np.random.RandomState(4)
    outer = rs.rand(3, 3, 4)
    outer[..., 3] = rs.rand(3, 3) + 0.5              # positive weights
    # bilinear identity (s,t) -> (s,t) over [0,1]^2, plain (u,v,w)
    param = np.zeros((2, 2, 3))
    param[..., 0] = [[0.0, 0.0], [1.0, 1.0]]         # u = s (axis 0)
    param[..., 1] = [[0.0, 1.0], [0.0, 1.0]]         # v = t (axis 1)
    param[..., 2] = 1.0
    comp = compose_patch_patch(outer, param, return_cartesian=True)
    outer_cart = outer[..., :3] / outer[..., 3:4]
    for (ci, cj), (oi, oj) in [((0, 0), (0, 0)), ((-1, 0), (-1, 0)),
                               ((0, -1), (0, -1)), ((-1, -1), (-1, -1))]:
        assert np.allclose(comp[ci, cj], outer_cart[oi, oj], atol=1e-12)


def test_high_degree_no_overflow():
    """Regression: degree > 30 used to read an out-of-bounds binomial table → inf."""
    rs = np.random.RandomState(9)
    patch = rs.rand(4, 4, 4)            # bicubic
    patch[..., 3] = 1.0
    curve = np.column_stack([np.linspace(0, 1, 7), np.linspace(0, 1, 7), np.ones(7)])
    out = compose_patch_curve(patch, curve, curve_ctrl_homogeneous=False)  # degree (3+3)*6 = 36
    assert out.shape[0] - 1 == 36
    assert np.isfinite(out).all()


@pytest.mark.parametrize("call", [
    lambda: compose_curve_curve(np.zeros((3, 1)), np.ones((2, 2))),                       # 1-col spatial
    lambda: compose_curve_curve(np.zeros((3, 4)), np.ones((2, 3))),                       # 3-col param
    lambda: compose_patch_curve(np.zeros((2, 2, 4)), np.zeros((3, 2)),
                                curve_ctrl_homogeneous=False),                            # 2-col curve
    lambda: compose_patch_patch(np.zeros((2, 2, 4)), np.zeros((2, 2, 2))),               # 2-col param
])
def test_malformed_input_raises_valueerror(call):
    with pytest.raises(ValueError):
        call()
