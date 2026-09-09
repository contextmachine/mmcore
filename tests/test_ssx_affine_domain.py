"""Constraint propagation preserves source geometry and boundary ownership."""
from types import SimpleNamespace

import numpy as np
import pytest

from mmcore.numeric._work_budget import SoftWorkBudget
from mmcore.numeric.intersection._root_box_certificate import exact_bernstein_value
from mmcore.numeric.intersection.ssx import _bez_ssx5 as ssx
from mmcore.numeric.intersection.ssx._ssx_affine_constraints import AffineParameterConstraints
from mmcore.numeric.intersection.ssx._ssx_cofactor_bounds import SourceZeroSetCofactorBounds


def _fixture():
    surface = np.array([[[u, v, u*v, 1.] for v in (0., 1.)] for u in (0., 1.)])
    sources = tuple(ssx.GaussMapBern.from_surf(surface, rational=True) for _ in range(2))
    minors = tuple(np.arange(16.).reshape((2,)*4)+k for k in range(4))
    point = ssx.BoundaryPoint(np.array([.25, .5, .25, .5]), np.array([.25, .5, .125]),
                             (0, -1), tangent_raw=np.ones(4),
                             root_box=np.array([[.25, .25], [.49, .51], [.25, .25], [.49, .51]]))
    cell = ssx._Cell(*sources, [point], ((.25, .75), (0., 1.), (0., 1.), (0., 1.)),
                    T1=minors[0], T2=minors[1], T3=minors[2], T4=minors[3],
                    boundary_complete=True, work_budget=SoftWorkBudget(1000, 100))
    cell.partitions = ssx._build_cell_partitions(cell)
    ssx._classify_boundary_point(point, cell)
    constraints = AffineParameterConstraints((
        {(0, 0): (0, 1), (1, 1): (0, 1)},
        {(0, 0): (0, 1), (1, 1): (0, 1)}))
    return cell, sources, minors, constraints


def test_domain_contraction_preserves_geometry_minors_and_owned_root():
    cell, sources, minors, constraints = _fixture()
    old_partitions = cell.partitions
    point, enclosure = cell.crossings[0], cell.crossings[0].root_box
    unrelated = SimpleNamespace(owner=object())
    point.registrations.append(unrelated)
    box = constraints.contract(cell.box)
    ssx._contract_source_domain(cell, box, sources, minors)
    assert cell.box == ((.25, .75), (0., 1.), (.25, .75), (0., 1.))
    assert cell.boundary_complete and cell.crossings[0] is point and point.root_box is enclosure
    assert unrelated in point.registrations
    assert all(not partition.registrations for partition in old_partitions)
    assert {r.partition.axis for r in point.registrations if r.owner is cell} == {0, 2}
    local = (.25, .375, .75, .625)
    global_ = tuple(lo+t*(hi-lo) for t, (lo, hi) in zip(local, box))
    for i, (restricted, source) in enumerate(zip((cell.g1, cell.g2), sources)):
        np.testing.assert_allclose(
            list(map(float, exact_bernstein_value(restricted.surface, local[2*i:2*i+2]))),
            list(map(float, exact_bernstein_value(source.surface, global_[2*i:2*i+2]))),
            rtol=0., atol=1e-14)
    for restricted, source in zip((cell.T1, cell.T2, cell.T3, cell.T4), minors):
        assert float(exact_bernstein_value(restricted[..., None], local)[0]) == pytest.approx(
            float(exact_bernstein_value(source[..., None], global_)[0]), abs=1e-14)


@pytest.mark.parametrize('refusal', ['zero_width', 'outside_representative', 'budget'])
def test_contraction_refusal_preserves_the_entire_old_cell(refusal):
    cell, sources, minors, constraints = _fixture()
    box = list(constraints.contract(cell.box))
    if refusal == 'zero_width':
        box[2] = (.25, .25)
    elif refusal == 'outside_representative':
        box[2] = (.375, .75)
    else:
        cell.work_budget = SoftWorkBudget(0, 100)
    original = cell.box, cell.g1, cell.g2, cell.T1, cell.partitions, cell.crossings
    ssx._contract_source_domain(cell, tuple(box), sources, minors)
    current = cell.box, cell.g1, cell.g2, cell.T1, cell.partitions, cell.crossings
    assert all(a is b for a, b in zip(original, current))


def test_zero_set_cofactor_proxy_uses_only_a_necessary_source_box():
    cell, _, _, constraints = _fixture()
    calls = []
    answer = np.ones(4), 2*np.ones(4)
    def bounds(box):
        calls.append(box)
        return answer
    proxy = SourceZeroSetCofactorBounds(SimpleNamespace(bounds=bounds), constraints)
    assert proxy.bounds(cell.box) is answer
    assert calls == [constraints.contract(cell.box)]
    assert proxy.bounds(((0., .25), (0., 1.), (.5, 1.), (0., 1.))) is None
    assert len(calls) == 1
