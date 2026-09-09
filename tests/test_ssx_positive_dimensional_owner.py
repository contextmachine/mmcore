from fractions import Fraction

import numpy as np
import pytest

from mmcore.numeric._work_budget import SoftWorkBudget
from mmcore.numeric.intersection.ssx._ssx_positive_dimensional import (
    exact_convex_bilinear_region_owner,
)


def pair():
    return (
        np.array([[[0., 0., 0.], [.25, 2., 0.]],
                  [[2., 0., 0.], [2.5, 2.5, 0.]]]),
        np.array([[[.5, .5, 0.], [.5, 1.5, 0.]],
                  [[1.5, .5, 0.], [1.5, 1.5, 0.]]]),
    )


def budget(**kwargs):
    return SoftWorkBudget(max_cells=10_000, max_csx_calls=100, **kwargs)


@pytest.mark.parametrize('rational', [False, True])
@pytest.mark.parametrize('transpose', [False, True])
def test_exact_region_owner_keeps_full_polygon_and_parameter_obligation(rational, transpose):
    a, b = pair()
    if transpose:
        a, b = b.transpose(1, 0, 2), a[::-1]
    if rational:
        weights = np.array([[1., .5], [2., 4.]])
        a = np.concatenate([a*weights[..., None], weights[..., None]], axis=-1)
        b = np.concatenate([b*weights[::-1, :, None], weights[::-1, :, None]], axis=-1)
    work = budget()
    result = exact_convex_bilinear_region_owner(a, b, rational, work)
    assert result is not None
    assert not result['branches'] and not result['points']
    assert work.incomplete and not work.exhausted
    assert work.csx_calls == 0
    assert work.reasons == ['overlap_region_unsupported']
    owner, = result['unresolved_regions']
    assert owner['stuv_min'] == (0.,)*4 and owner['stuv_max'] == (1.,)*4
    assert owner['exact_dimension'] == 2
    polygon = [tuple(Fraction(x) for x in point) for point in owner['exact_image_polygon']]
    assert set(polygon) == {(Fraction(x), Fraction(y), Fraction(0))
                            for x in (.5, 1.5) for y in (.5, 1.5)}
    assert owner['proof'] == 'exact_convex_rational_bilinear_region'


def test_nonzero_plane_gap_never_becomes_positive_dimensional_identity():
    a, b = pair()
    b[..., 2] = np.nextafter(0., 1.)
    assert exact_convex_bilinear_region_owner(a, b, False, budget()) is None


def test_common_plane_edge_contact_remains_with_the_curve_solver():
    a = np.array([[[s, t, 0.] for t in (0., 1.)] for s in (0., 1.)])
    b = a.copy()
    b[..., 0] += 1.
    assert exact_convex_bilinear_region_owner(a, b, False, budget()) is None


def test_folded_or_higher_degree_charts_are_not_retired():
    a, b = pair()
    a[1, 1] = a[0, 0]
    assert exact_convex_bilinear_region_owner(a, b, False, budget()) is None
    a, b = pair()
    a = np.stack([a[0], (a[0]+a[1])/2., a[1]])
    assert exact_convex_bilinear_region_owner(a, b, False, budget()) is None


def test_work_denial_precedes_exact_conversion(monkeypatch):
    a, b = pair()
    import mmcore.numeric.intersection.ssx._ssx_positive_dimensional as module
    monkeypatch.setattr(module, '_exact_points', lambda *a: pytest.fail('unpaid conversion'))
    work = SoftWorkBudget(max_cells=0, max_csx_calls=100)
    assert module.exact_convex_bilinear_region_owner(a, b, False, work) is None
    assert work.exhausted


def test_output_denial_keeps_the_structural_reason():
    a, b = pair()
    work = budget(max_output_items=0)
    result = exact_convex_bilinear_region_owner(a, b, False, work)
    assert result is not None and result['unresolved_regions'] == []
    assert work.incomplete and 'overlap_region_unsupported' in work.reasons
    assert 'output_cap' in work.reasons


def test_full_ssx_retains_the_exact_owner_without_product_search():
    from mmcore.numeric.intersection.ssx._bez_ssx5 import bez_ssx
    a, b = pair()
    result = bez_ssx(a, b, rational=False)
    assert not result['complete']
    assert result['status']['reasons'] == ['overlap_region_unsupported']
    assert len(result['unresolved_regions']) == 1
    assert not result['branches'] and not result['points']
    assert result['status']['work']['csx_calls'] == 0
    assert result['status']['work']['cell_counts'].get('ssx', 0) == 0
