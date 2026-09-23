"""CAD polygon overlap: shared corners and both parameter-path images."""
import numpy as np
import pytest

from mmcore.numeric._bezier_common import eval_surface
from mmcore.numeric.intersection.ssx._ssx5_overlap import assemble_overlap_regions
from mmcore.numeric.intersection.ssx._ssx_substrate import SSXBranch


def _homogeneous(points, weights):
    return np.concatenate((points*weights[..., None], weights[..., None]), axis=2)


def _assert_paired_chords(result, first, second, atol):
    for branch in result['rim_branches']:
        q, x = map(np.asarray, branch.curve)
        assert len(q) >= 17
        for i in range(len(q)-1):
            for t in np.linspace(0., 1., 17):
                stuv = (1.-t)*q[i]+t*q[i+1]
                xyz = (1.-t)*x[i]+t*x[i+1]
                for surface, uv in ((first, stuv[:2]), (second, stuv[2:])):
                    assert np.linalg.norm(eval_surface(surface, *uv, rational=True)-xyz) <= atol


@pytest.mark.parametrize('weighted,swap,reverse', [
    (False, False, False), (False, True, True),
    (True, False, False), (True, True, True),
])
def test_oblique_polygon_clipping_keeps_eight_shared_corners(weighted, swap, reverse):
    a = np.array([[[-1., -1., 0.], [-1., 1., 0.]],
                  [[1., -1., 0.], [1., 1., 0.]]])
    b = np.array([[[-1.5, 0., 0.], [0., 1.5, 0.]],
                  [[0., -1.5, 0.], [1.5, 0., 0.]]])
    w1 = np.array([[.4, 2.], [3., .7]]) if weighted else np.ones((2, 2))
    w2 = np.array([[2., .6], [.3, 4.]]) if weighted else np.ones((2, 2))
    pair = [_homogeneous(a, w1), _homogeneous(b, w2)]
    if reverse:
        pair[0] = pair[0][::-1].copy()
    if swap:
        pair.reverse()
    atol = 1e-4
    result = assemble_overlap_regions(*pair, atol=atol, ptol4=np.full(4, atol/4))
    assert len(result['regions']) == 1
    assert len(result['rim_branches']) == 8
    region = result['regions'][0]
    assert region.certification['orientation_consistent']
    assert region.normal_agreement == (-1 if reverse else 1)
    expected = np.array([[-1., -.5, 0.], [-.5, -1., 0.], [.5, -1., 0.], [1., -.5, 0.],
                         [1., .5, 0.], [.5, 1., 0.], [-.5, 1., 0.], [-1., .5, 0.]])
    actual = np.array([b.curve[1][0] for b in result['rim_branches']])
    assert np.max(np.min(np.linalg.norm(expected[:, None]-actual[None], axis=2), axis=1)) < atol
    ordered = []
    for index, reversed_ in region.boundary[0]:
        q, x = result['rim_branches'][index].curve
        if reversed_:
            q, x = q[::-1], x[::-1]
        if ordered:
            np.testing.assert_array_equal(ordered[-1][0][-1], q[0])
            np.testing.assert_array_equal(ordered[-1][1][-1], x[0])
        ordered.append((q, x))
    np.testing.assert_array_equal(ordered[-1][0][-1], ordered[0][0][0])
    _assert_paired_chords(result, *pair, atol)


def test_rational_containment_rims_resolve_curved_parameter_inverse():
    outer = np.array([[[0., 0., 0.], [.3, 2.2, 0.]],
                      [[2.1, -.2, 0.], [2.9, 2.6, 0.]]])
    inner = np.array([[[.6, .5, 0.], [.7, 1.4, 0.]],
                      [[1.5, .55, 0.], [1.45, 1.5, 0.]]])
    pair = [_homogeneous(outer, np.array([[.1, 2.], [3., .3]])),
            _homogeneous(inner, np.array([[2., .3], [.2, 4.]]))]
    atol = 1e-5
    result = assemble_overlap_regions(*pair, atol=atol, ptol4=np.full(4, atol/4))
    assert len(result['regions']) == 1
    assert len(result['rim_branches']) == 4
    # The original 17 samples are insufficient for these nonlinear inverse
    # parameterizations. Adaptive refinement must validate their chords.
    assert max(len(b.curve[0]) for b in result['rim_branches']) > 17
    _assert_paired_chords(result, *pair, atol)


def test_overlap_clipping_budget_denial_preserves_existing_geometry():
    points = np.array([[[0., 0., 0.], [0., 1., 0.]],
                       [[1., 0., 0.], [1., 1., 0.]]])
    surface = _homogeneous(points, np.ones((2, 2)))
    branch = SSXBranch(curve=(np.array([[0., 0., 0., 0.], [1., 0., 1., 0.]]),
                             np.array([[0., 0., 0.], [1., 0., 0.]])),
                       kind='overlap', overlap=True)
    for allowance in (0, 25, 100):
        used = 0

        def charge(n):
            nonlocal used
            if used+n > allowance:
                return False
            used += n
            return True

        result = assemble_overlap_regions(surface, surface, atol=1e-3,
                                          ptol4=np.full(4, 1e-3),
                                          existing_overlap_branches=[branch], charge=charge)
        assert result['regions'] == [] and result['rim_branches'] == []
        assert result['unmatched_branches'] == [branch]
        assert not result['covered'] and used <= allowance


def test_planar_region_owns_paired_interior_samples_but_keeps_foreign_paths():
    points = np.array([[[0., 0., 0.], [.3, 2.2, 0.]],
                       [[2.1, -.2, 0.], [2.9, 2.6, 0.]]])
    surface = _homogeneous(points, np.ones((2, 2)))
    q = np.array([[0., 0., 0., 0.], [1., 1., 1., 1.]])
    x = points[[0, 1], [0, 1]].copy()
    interior = SSXBranch(curve=(q, x), kind='tangential')
    # The paired vertices are valid region samples, although their coarse
    # straight XYZ interpolation does not follow this nonaffine chart.
    result = assemble_overlap_regions(surface, surface, atol=1e-3,
                                      ptol4=np.full(4, 1e-3),
                                      existing_intersection_branches=[interior])
    assert len(result['regions']) == 1 and result['planar_pair_covered']
    assert result['unmatched_intersection_branches'] == []
    foreign = SSXBranch(curve=(q, x+np.array([0., 0., .1])), kind='tangential')
    result = assemble_overlap_regions(surface, surface, atol=1e-3,
                                      ptol4=np.full(4, 1e-3),
                                      existing_intersection_branches=[foreign])
    assert len(result['regions']) == 1
    assert result['unmatched_intersection_branches'] == [foreign]
    assert not result['planar_pair_covered']


def test_planar_offset_is_judged_at_the_requested_cad_tolerance():
    points = np.array([[[0., 0., 0.], [0., 1., 0.]],
                       [[1., 0., 0.], [1., 1., 0.]]])
    first = _homogeneous(points, np.ones((2, 2)))
    atol = 1e-3
    second = _homogeneous(points+np.array([0., 0., .2*atol]), np.ones((2, 2)))
    result = assemble_overlap_regions(first, second, atol=atol, ptol4=np.full(4, atol))
    assert len(result['regions']) == 1 and result['planar_pair_covered']
    _assert_paired_chords(result, first, second, atol)


@pytest.mark.parametrize('output_limit', [0, 2, 4, 5])
def test_early_planar_region_never_references_unreturned_rims(output_limit):
    from examples.ssx.bez_ssx5_case12 import S1, S2
    from mmcore.numeric.intersection.ssx._bez_ssx5 import bez_ssx

    result = bez_ssx(S1, S2, atol=1e-3, rational=False,
                     max_output_items=output_limit, max_postprocess_work=0)
    count = sum(len(result[key]) for key in
                ('branches', 'points', 'singularities', 'overlap_regions'))
    assert count <= output_limit
    for region in result['overlap_regions']:
        for loop in region.boundary:
            for index, _reverse in loop:
                assert 0 <= index < len(result['branches'])
                assert result['branches'][index].kind == 'overlap'
    if output_limit < 5:
        assert result['overlap_regions'] == []
        assert result['complete'] is False
    else:
        assert len(result['overlap_regions']) == 1
        assert len(result['branches']) == 4
        assert result['complete'] is True
    assert result['status']['work']['postprocess_work'] == 0


@pytest.mark.parametrize('allowance', [0, 25, 100])
def test_early_planar_region_respects_shared_work_denial(allowance):
    from examples.ssx.bez_ssx5_case12 import S1, S2
    from mmcore.numeric.intersection.ssx._bez_ssx5 import bez_ssx

    result = bez_ssx(S1, S2, atol=1e-3, rational=False, max_cells=allowance)
    assert result['status']['work']['cells_processed'] <= allowance
    assert result['complete'] is False
    for region in result['overlap_regions']:
        assert all(0 <= index < len(result['branches'])
                   for loop in region.boundary for index, _reverse in loop)


@pytest.mark.parametrize('kind', ['folded', 'nonplanar', 'higher_degree'])
def test_early_planar_entry_leaves_unsupported_charts_to_general_search(kind):
    from mmcore.numeric.intersection.ssx._ssx5_overlap import try_planar_intersection
    from mmcore.numeric.intersection.ssx._bez_ssx5 import _SSXSoftBudget

    points = np.array([[[0., 0., 0.], [0., 1., 0.]],
                       [[1., 0., 0.], [1., 1., 0.]]])
    first = _homogeneous(points, np.ones((2, 2)))
    second = first.copy()
    if kind == 'folded':
        first[1] = first[1, ::-1]
    elif kind == 'nonplanar':
        first[1, 1, 2] = .1
    else:
        first = np.stack((first[0], .5*(first[0]+first[1]), first[1]))
    budget = _SSXSoftBudget(max_cells=1000, max_csx_calls=10)
    assert try_planar_intersection(first, second, 1e-3, budget) is None
    assert not budget.exhausted


def test_early_planar_witness_denial_retains_completed_rim_geometry(monkeypatch):
    from mmcore.numeric.intersection.ssx import _ssx5_overlap as overlap
    from mmcore.numeric.intersection.ssx._bez_ssx5 import bez_ssx

    first = np.array([[[0., 0., 0.], [0., 1., 0.]],
                      [[2., 0., 0.], [2., 1., 0.]]])
    second = first + [1., 0., 0.]
    surfaces = [_homogeneous(points, np.ones((2, 2))) for points in (first, second)]
    used = 0
    def measure(amount):
        nonlocal used
        used += amount
        return True
    context = {}
    rims = overlap._planar_convex_rims(
        *surfaces, 1e-3, measure, context, include_boundary_contacts=True)
    assert context['dimension'] == 2 and len(rims) == 4

    def forbidden(*args, **kwargs):
        raise AssertionError('interior validation ran after work denial')
    monkeypatch.setattr(overlap, '_interior_witness', forbidden)
    result = bez_ssx(first, second, atol=1e-3, rational=False, max_cells=used)
    assert result['complete'] is False
    assert result['status']['work']['cells_processed'] <= used
    assert 'work_budget' in result['status']['reasons']
    assert result['overlap_regions'] == []
    assert len(result['branches']) == 4
    assert all(branch.kind == 'overlap' for branch in result['branches'])
    _assert_paired_chords({'rim_branches': result['branches']}, *surfaces, 1e-3)
