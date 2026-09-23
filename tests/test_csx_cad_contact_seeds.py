"""A CAD proximity contact proposes, but does not establish, an SSI crossing."""

from types import SimpleNamespace

import numpy as np
import pytest

from mmcore.numeric._bezier_common import eval_surface
from mmcore.numeric.intersection.ssx._bez_ssx5 import _register_cut_contacts


@pytest.mark.parametrize("has_crossing", [False, True])
def test_cut_contact_requires_a_continuation_root(has_crossing):
    plane = np.array([[[0., 0., 0.], [0., 1., 0.]],
                      [[1., 0., 0.], [1., 1., 0.]]])
    first = plane.copy()
    if has_crossing:
        # Fixed s=.5 has a root at t=.0005; the CAD proposal at t=0
        # must move onto that intersection before it can seed tracing.
        first[..., 2] = first[..., 0] + first[..., 1] - .5005
    else:
        # The same sub-atol gap on parallel planes has no crossing
        # direction anywhere on this face.
        first[..., 2] = .0005
    first_h = np.concatenate([first, np.ones((2, 2, 1))], axis=2)
    second_h = np.concatenate([plane, np.ones((2, 2, 1))], axis=2)
    cell = SimpleNamespace(box=((0., 1.),) * 4,
                           g1=SimpleNamespace(surface=first_h),
                           g2=SimpleNamespace(surface=second_h))
    point = dict(t=0., u=.5, v=0.,
                 point=eval_surface(first_h, .5, 0., rational=True),
                 certification='tolerance', d_min=.0005)
    grid = [[[] for _ in range(2)] for _ in range(2)]
    _register_cut_contacts(cell, [point], 0, .5, 0, 2, [.5], grid)
    roots = [root for row in grid for entries in row for root in entries]
    if not has_crossing:
        assert roots == []
        return
    assert roots
    for root in roots:
        assert np.all((0. <= root.stuv) & (root.stuv <= 1.))
        # This tests the numerical refinement step, independently of the
        # public CAD tolerance used to admit the original proposal.
        assert root.stuv[1] > 0.
        first_point = eval_surface(first_h, *root.stuv[:2], rational=True)
        second_point = eval_surface(second_h, *root.stuv[2:], rational=True)
        assert np.linalg.norm(first_point - second_point) <= 1e-8
        assert np.linalg.norm(root.xyz - first_point) <= 1e-3
