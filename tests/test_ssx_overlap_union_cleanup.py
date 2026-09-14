"""Paired overlap unions remove retraces without joining gaps or sheets."""
import numpy as np

from mmcore.numeric.intersection.ssx import _bez_ssx5 as ssx


def _branch(xy, *, overlap=False, shift=0.):
    xy = np.asarray(xy, dtype=float)
    parameters = np.column_stack([xy, xy])
    parameters[:, 0] += shift
    return ssx.SSXBranch(
        curve=(parameters, np.column_stack([xy, np.zeros(len(xy))])),
        overlap=overlap, kind='overlap' if overlap else 'transversal')


def test_trace_can_be_covered_by_two_meeting_overlap_edges():
    left = _branch([[0., 0.], [1., 0.]], overlap=True)
    right = _branch([[1., 0.], [1., 1.]], overlap=True)
    trace = _branch([[0., 0.], [.5, 0.], [1., 0.], [1., .5], [1., 1.]])
    result = ssx._drop_traces_covered_by_overlap_union(
        [trace, left, right], [left, right], .001, np.full(4, .001))
    assert [id(branch) for branch in result] == [id(left), id(right)]


def test_separate_overlap_arrays_do_not_create_a_gap_covering_chord():
    left = _branch([[0., 0.], [.4, 0.]], overlap=True)
    right = _branch([[.6, 0.], [1., 0.]], overlap=True)
    bridge = _branch([[0., 0.], [1., 0.]])
    result = ssx._drop_traces_covered_by_overlap_union(
        [bridge, left, right], [left, right], .001, np.full(4, .001))
    assert any(branch is bridge for branch in result)


def test_geometrically_coincident_distinct_parameter_sheet_survives():
    overlap = _branch([[0., 0.], [1., 0.]], overlap=True)
    other_sheet = _branch([[0., 0.], [.5, 0.], [1., 0.]], shift=.1)
    result = ssx._drop_traces_covered_by_overlap_union(
        [other_sheet, overlap], [overlap], .001, np.full(4, .001))
    assert any(branch is other_sheet for branch in result)


def test_incident_arch_leaving_the_overlap_union_survives():
    left = _branch([[0., 0.], [.5, 0.]], overlap=True)
    right = _branch([[.5, 0.], [1., 0.]], overlap=True)
    arch = _branch([[0., 0.], [.5, .25], [1., 0.]])
    result = ssx._drop_traces_covered_by_overlap_union(
        [arch, left, right], [left, right], .001, np.full(4, .001))
    assert any(branch is arch for branch in result)


def test_cleanup_denial_preserves_discovered_trace():
    overlap = _branch([[0., 0.], [1., 0.]], overlap=True)
    trace = _branch([[0., 0.], [1., 0.]])
    budget = ssx._SSXSoftBudget(max_cells=100, max_csx_calls=10,
                              max_postprocess_work=0)
    result = ssx._drop_traces_covered_by_overlap_union(
        [trace, overlap], [overlap], .001, np.full(4, .001), budget)
    assert any(branch is trace for branch in result)
    assert 'postprocess_cap' in budget.reasons


def _curved_inverse_example(folded_partner=False):
    # S1(s,t)=(s,(1+s)*t,0): a straight horizontal XYZ line has the
    # nonlinear inverse t=.5/(1+s). The second chart is either affine or
    # folded, in which case two different u values represent each point.
    first = np.array([[[0., 0., 0., 1.], [0., 1., 0., 1.]],
                      [[1., 0., 0., 1.], [1., 2., 0., 1.]]])
    if folded_partner:
        second = np.array([[[x, y, 0., 1.] for y in (0., 1.)]
                           for x in (1., -1., 1.)])
    else:
        second = np.array([[[x, y, 0., 1.] for y in (0., 1.)]
                           for x in (0., 1.)])

    def branch(count, overlap, other_sheet=False):
        x = np.linspace(.1, .9, count)
        partner_u = (.5 + (.5 if other_sheet else -.5)*np.sqrt(x)
                     if folded_partner else x)
        q = np.column_stack([x, .5/(1.+x), partner_u, np.full(count, .5)])
        xyz = np.column_stack([x, np.full(count, .5), np.zeros(count)])
        return ssx.SSXBranch(curve=(q, xyz), overlap=overlap,
                             kind='overlap' if overlap else 'transversal')

    return first, second, branch


def test_inverse_resampling_removes_a_coarse_parameter_chord_retrace():
    first, second, make = _curved_inverse_example()
    overlap, trace = make(65, True), make(2, False)
    # The unrefined parameter chord leaves the shared straight XYZ edge.
    original = ssx._drop_traces_covered_by_overlap_union(
        [trace, overlap], [overlap], .001, np.full(4, .001))
    assert any(branch is trace for branch in original)
    result = ssx._drop_traces_covered_by_overlap_union(
        [trace, overlap], [overlap], .001, np.full(4, .001),
        surfaces=(first, second))
    assert [id(branch) for branch in result] == [id(overlap)]


def test_inverse_resampling_keeps_a_folded_partners_other_preimage():
    first, second, make = _curved_inverse_example(folded_partner=True)
    overlap, other_sheet = make(65, True), make(2, False, other_sheet=True)
    result = ssx._drop_traces_covered_by_overlap_union(
        [other_sheet, overlap], [overlap], .001, np.full(4, .001),
        surfaces=(first, second))
    assert any(branch is other_sheet for branch in result)
