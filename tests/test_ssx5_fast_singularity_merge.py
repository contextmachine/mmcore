"""Fast geometry and the general singular pass share published records."""
import numpy as np
import pytest

from mmcore.numeric.intersection.ssx._bez_ssx5 import bez_ssx


@pytest.mark.parametrize('swap, transpose', [(False, False), (True, False),
                                            (False, True)])
def test_full_cusp_ruling_is_retained_without_spending_another_output_item(swap, transpose):
    cusp = np.array([[[x, y, float(t)] for t in (0, 1)]
                     for x, y in zip((3., -1., -1., 3.), (-1., 1., -1., 1.))])
    plane = np.array([[[0., y, z] for z in (-.5, 1.5)] for y in (-1.5, 1.5)])
    if transpose:
        cusp = cusp.swapaxes(0, 1).copy()
    pair = (plane, cusp) if swap else (cusp, plane)
    result = bez_ssx(*pair, atol=1e-3, rational=False, max_output_items=2)
    curves = [g for g in result['singularities'] if g.kind == 'cusp_curve']
    assert len(curves) == 1
    assert curves[0].surface == (2 if swap else 1)
    owner = 2 if swap else 0
    free = owner if transpose else owner+1
    samples = np.asarray(curves[0].samples)
    assert samples[:, free].min() <= 1e-3
    assert samples[:, free].max() >= 1.-1e-3
    assert len(samples) > 2  # Both the full ruling and later C1 observations survive.
    assert 'output_cap' not in result['status']['reasons']
    assert result['status']['work']['output_items'] <= 2
