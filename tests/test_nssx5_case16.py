"""Real CAD sections: complete endpoints and paired surface geometry at 1e-3."""

import numpy as np
import pytest

from examples.ssx import _case16_data as data
from mmcore.numeric.intersection.ssx import nurbs_ssx
from mmcore.nurbs._nurbs_eval import evaluate_nurbs_surface


@pytest.mark.parametrize('index,name', enumerate(
    ('surf11', 'surf12', 'surf21', 'surf22', 'surf31')))
def test_case16_section_keeps_both_domain_endpoints(index, name):
    atol = 1e-3
    first, second = data.surf0, getattr(data, name)
    result = nurbs_ssx(first, second, atol=atol)

    assert len(result['branches']) == 1
    branch = result['branches'][0]
    assert not branch.closed
    parameters, xyz = map(np.asarray, branch.curve)
    expected = np.asarray(data.expected_endpoints[index])
    forward = np.linalg.norm(xyz[[0, -1]] - expected, axis=1).max()
    reverse = np.linalg.norm(xyz[[-1, 0]] - expected, axis=1).max()
    assert min(forward, reverse) <= atol
    assert np.isfinite(parameters).all() and np.isfinite(xyz).all()
    # Endpoint coverage alone could accept a straight shortcut or unrelated
    # path. Every returned vertex must evaluate on both original NURBS
    # surfaces at the requested physical tolerance, in their native domains.
    for point, q in zip(xyz, parameters):
        for surface, uv in ((first, q[:2]), (second, q[2:])):
            bounds = np.asarray(surface.interval())
            assert np.all(uv >= bounds[:, 0]) and np.all(uv <= bounds[:, 1])
            source = evaluate_nurbs_surface(surface, *uv, d_order=0)['S']
            assert np.linalg.norm(source - point) <= atol
