"""Loop absence is not an exhaustive boundary census or an arc pairing."""
from types import SimpleNamespace

import numpy as np
import pytest

from mmcore.numeric._work_budget import SoftWorkBudget
from mmcore.numeric.intersection.ssx import _bez_ssx5 as ssx


def _sources():
    first = np.array([[[s, t, 0.] for t in (0., 1.)] for s in (0., 1.)])
    second = np.array([[[s, t, t-.5] for t in (0., 1.)] for s in (0., 1.)])
    return first, second


@pytest.mark.parametrize('number, boundary_complete', [(2, False), (4, True)])
def test_regular_tracer_does_not_discharge_unproved_arc_ownership(monkeypatch, number,
                                                               boundary_complete):
    sources = tuple(ssx.GaussMapBern.from_surf(surface) for surface in _sources())
    roots = [ssx.BoundaryPoint(np.array([k/(number-1), .5, k/(number-1), .5]),
                              np.array([k/(number-1), .5, 0.]), (0, -1))
             for k in range(number)]
    cell = ssx._Cell(*sources, roots, ((0., 1.),)*4,
                    boundary_complete=boundary_complete,
                    source_cofactors=SimpleNamespace(bounds=lambda box: (np.ones(4), np.ones(4))),
                    work_budget=SoftWorkBudget(1000, 100))
    def forbidden(*args, **kwargs):
        raise AssertionError('ambiguous regular cell reached the numerical marcher')
    monkeypatch.setattr(ssx, '_march_to_boundary', forbidden)
    assert ssx._trace_cell_by_registrations(cell, .001) == ([], [])
    assert cell.trace_incomplete


def test_loop_free_cell_with_an_incomplete_empty_census_stays_unresolved(monkeypatch):
    def unknown_boundary(*args, **kwargs):
        kwargs['census_sink']['complete'] = False
        return [], []
    monkeypatch.setattr(ssx, '_find_ssx_boundary_zeros', unknown_boundary)
    result = ssx.bez_ssx(*_sources(), rational=False, max_depth=0, max_xyz_step=.1)
    assert not result['complete']
    assert 'depth_limit' in result['status']['reasons']
    assert result['unresolved_regions']


@pytest.mark.parametrize('center, gap', [(.5, 2.**-28), (2.**-1000, 2.**-1003)])
def test_certified_parameter_gap_survives_modeling_tolerance(center, gap):
    roots = []
    for parameter in (center-gap, center+gap):
        q = np.array([parameter, .5, parameter, .5])
        point = ssx.BoundaryPoint(q, np.zeros(3), (1, -1),
                                 root_box=np.column_stack((q, q)))
        point._source_root_box = True
        roots.append(point)
    cuts = ssx._root_separating_cuts(roots, 0, ((0., 1.),)*4, np.full(4, .001))
    assert center in cuts


def test_exact_root_interval_can_be_strictly_inside_a_float_touching_face():
    from fractions import Fraction
    from mmcore.numeric.intersection.ssx._ssx_cofactor_bounds import SourceCofactorBounds
    from mmcore.numeric.intersection.ssx._ssx_planar_cut import exact_source_planar_cut
    from mmcore.numeric.intersection.ssx._ssx_root_identity import BoundaryRootIdentity

    first = np.array([[[s, t, 3*t-2, 1.] for t in (0., 1.)] for s in (0., 1.)])
    second = np.array([[[s, t, 0., 1.] for t in (0., 1.)] for s in (0., 1.)])
    matcher = BoundaryRootIdentity(first, second)
    points = []
    for pin in (0., 1.):
        result = exact_source_planar_cut(first, second, 0, pin, ((0., 1.),)*4)
        assert result['boundary_topology_complete'] and len(result['isolated']) == 1
        root = result['isolated'][0]
        point = ssx.BoundaryPoint(root['stuv'], root['point'], (0, -1),
                                 root_box=np.asarray(root['parameter_root_box']))
        point._source_root_box = True
        matcher.register_source_root(point, root['source_cut_certificate'])
        points.append(point)
    low = float(Fraction(2, 3))
    assert Fraction(low) < Fraction(2, 3)
    box = ((0., 1.), (low, 1.), (0., 1.), (low, 1.))
    cell = ssx._Cell(ssx.GaussMapBern.from_surf(first, rational=True),
                    ssx.GaussMapBern.from_surf(second, rational=True), points, box,
                    boundary_complete=True, root_matcher=matcher,
                    source_cofactors=SourceCofactorBounds(first, second),
                    T1=np.ones((1,)*4), T2=np.ones((1,)*4),
                    T3=np.ones((1,)*4), T4=np.ones((1,)*4))
    # Outward float enclosures touch the t/v lower faces, where the true
    # arc derivative is zero. Exact source t=v=2/3 lies strictly inside.
    assert any(point.root_box[1, 0] == low for point in points)
    assert ssx._regular_cell_arc_enclosure(cell) is not None
    matcher.source_certificates.clear()
    assert ssx._regular_cell_arc_enclosure(cell) is None
