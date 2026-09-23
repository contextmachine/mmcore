"""Assembly preserves a geometric path under regular nonlinear charts.

These are geometry and topology checks at the public CAD tolerance. The
Bernstein coefficients define analytic intersections independently of the
SSX implementation; no equality of Newton iterates is required.
"""
import numpy as np
import pytest

from mmcore.numeric.intersection.ssx import _bez_ssx5 as ssx


@pytest.mark.parametrize('quadratic', [.2, .8])
@pytest.mark.parametrize('sample_count', [2, 3, 5])
@pytest.mark.parametrize('swap', [False, True])
def test_straight_intersection_survives_nonlinear_parameter_correspondence(
        quadratic, sample_count, swap):
    # f(s)=(1-a)*s+a*s^2 is strictly increasing on [0,1]. The
    # surfaces (f(s),t,0) and (u,0,v) intersect along the entire x-axis.
    # At a parameter midpoint f((s0+s1)/2) != (f(s0)+f(s1))/2, even
    # though every world-space chord lies on both surfaces exactly.
    x_controls = [0., (1.-quadratic)/2., 1.]
    first = np.array([[[x, t, 0.] for t in [0., 1.]]
                      for x in x_controls])
    second = np.array([[[u, 0., v] for v in [0., 1.]]
                       for u in [0., 1.]])
    s = np.linspace(0., 1., sample_count)
    x = (1.-quadratic)*s+quadratic*s*s
    stuv = np.column_stack((s, np.zeros_like(s), x, np.zeros_like(s)))
    xyz = np.column_stack((x, np.zeros_like(x), np.zeros_like(x)))
    if swap:
        first, second = second, first
        stuv = stuv[:, [2, 3, 0, 1]]
    fragment = ssx._Fragment(None, None, stuv, xyz)
    atol = 1e-3
    branches = ssx._assemble_fragments(
        [fragment], S1_full=first, S2_full=second,
        rational_full=False, atol_full=atol)
    assert len(branches) == 1
    assert not branches[0].closed
    actual = np.asarray(branches[0].curve[1])
    assert np.linalg.norm(actual[0]-[0., 0., 0.]) <= atol
    assert np.linalg.norm(actual[-1]-[1., 0., 0.]) <= atol
    assert abs(np.linalg.norm(np.diff(actual, axis=0), axis=1).sum()-1.) <= atol
    for parameters, point in zip(branches[0].curve[0], actual):
        on_first = ssx.eval_surface(first, *parameters[:2], rational=False)
        on_second = ssx.eval_surface(second, *parameters[2:], rational=False)
        assert np.linalg.norm(on_first-point) <= atol
        assert np.linalg.norm(on_second-point) <= atol
