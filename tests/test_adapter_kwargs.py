"""L52 slice 8: NURBS adapters reject unknown kwargs.

`tol=` is the adapter-level spelling and `atol=` the bez-level one; the
former silent `**kwargs` swallow meant `nurbs_ccx(..., atol=1e-6)` ran at
the DEFAULT tolerance without a word (review §10 kwarg-hygiene finding).
"""
import numpy as np
import pytest

from mmcore.geom._nurbs_eval import NURBSCurveTuple, NURBSSurfaceTuple
from mmcore.numeric.intersection.ccx._nccx4 import nurbs_ccx
from mmcore.numeric.intersection.csx._ncsx4 import nurbs_csx


def _line(p0, p1):
    return NURBSCurveTuple(
        control_points=np.array([p0, p1], dtype=float),
        weights=np.ones(2), knot=np.array([0.0, 0.0, 1.0, 1.0]), order=2)


def _plane():
    return NURBSSurfaceTuple(
        control_points=np.array(
            [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
             [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]]),
        weights=np.ones((2, 2)),
        knot_u=np.array([0.0, 0.0, 1.0, 1.0]),
        knot_v=np.array([0.0, 0.0, 1.0, 1.0]),
        order_u=2, order_v=2)


def test_nurbs_ccx_rejects_unknown_kwargs():
    c1 = _line([0.0, -0.5, 0.0], [0.0, 0.5, 0.0])
    c2 = _line([-0.5, 0.0, 0.0], [0.5, 0.0, 0.0])
    with pytest.raises(TypeError, match="atol"):
        nurbs_ccx(c1, c2, atol=1e-6)
    # the accepted knobs still pass through
    isolated, overlaps, status = nurbs_ccx(c1, c2, max_cells=10_000)
    assert status["complete"] is not None


def test_nurbs_csx_rejects_unknown_kwargs_and_uses_tol():
    # L52 slice 8 unification: nurbs_csx's tolerance parameter was `atol=`
    # while its sibling nurbs_ccx used `tol=` — passing either spelling to
    # the other adapter was silently swallowed. Both adapters now spell it
    # `tol=` (the adapter-level convention; `atol` stays the bez level's).
    c = _line([0.5, 0.5, -1.0], [0.5, 0.5, 1.0])
    s = _plane()
    with pytest.raises(TypeError, match="atol"):
        nurbs_csx(c, s, atol=1e-6)
    isolated, overlaps, status = nurbs_csx(c, s, tol=1e-3, max_depth=32)
    assert status["complete"] is not None
    assert len(isolated) == 1
