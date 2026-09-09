"""Necessary interval contraction must preserve every source zero."""
import numpy as np

from mmcore.numeric.intersection.ssx._ssx_cofactor_bounds import SourceCofactorBounds
from mmcore.numeric.intersection.ssx._ssx_residual_contract import SourceResidualContractor


def _plane():
    return np.array([[[u, v, 0., 1.] for v in (0., 1.)] for u in (0., 1.)])


def test_correlated_parameter_box_contracts_without_claiming_existence():
    first = _plane()
    second = np.array([[[u, .5, v-.5, 1.] for v in (0., 1.)] for u in (0., 1.)])
    contractor = SourceResidualContractor(SourceCofactorBounds(first, second))
    result = contractor.contract(((.25, .3), (0., 1.), (0., 1.), (0., 1.)))
    assert result is not None
    assert result[2][0] > .2 and result[2][1] < .4
    assert result[1][0] <= .5 <= result[1][1]
    assert result[3][0] <= .5 <= result[3][1]
    for t in np.linspace(.25, .3, 7):
        assert all(lo <= q <= hi for q, (lo, hi) in zip((t, .5, t, .5), result))


def test_singular_free_parameter_zero_set_is_never_removed():
    plane = _plane()
    contractor = SourceResidualContractor(SourceCofactorBounds(plane, plane))
    box = ((.25, .75),)*4
    result = contractor.contract(box)
    assert result is not None
    for s in (.25, .5, .75):
        for t in (.25, .5, .75):
            assert all(lo <= q <= hi for q, (lo, hi) in zip((s, t, s, t), result))


def test_denied_work_preserves_original_domain():
    plane = _plane()
    source = SourceCofactorBounds(plane, plane)
    source.charge = lambda amount: False
    contractor = SourceResidualContractor(source)
    box = ((.2, .7),)*4
    assert contractor.contract(box) == box
    assert source.exhausted


def test_subnormal_geometry_cannot_create_a_false_exclusion():
    first = _plane()
    first[..., :3] *= np.nextafter(0., 1.)
    contractor = SourceResidualContractor(SourceCofactorBounds(first, first))
    box = ((.25, .75),)*4
    result = contractor.contract(box)
    assert result is not None
    assert all(lo <= .5 <= hi for lo, hi in result)


def test_subnormal_parameter_singleton_keeps_its_exact_source_root():
    tiny = np.nextafter(0., 1.)
    first = _plane()
    second = np.array([[[u, .5, v-.5, 1.] for v in (0., 1.)] for u in (0., 1.)])
    contractor = SourceResidualContractor(SourceCofactorBounds(first, second))
    box = ((tiny, tiny), (.5, .5), (tiny, tiny), (.5, .5))
    assert contractor.contract(box) == box


def test_disjoint_affine_planes_have_proved_empty_parameter_box():
    first = _plane()
    second = np.array([[[u, .5, v+1., 1.] for v in (0., 1.)] for u in (0., 1.)])
    contractor = SourceResidualContractor(SourceCofactorBounds(first, second))
    assert contractor.contract(((0., 1.),)*4) is None
