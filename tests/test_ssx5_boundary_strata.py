"""Exact boundary-stratum topology and independent lifted-path checks."""
import numpy as np
import pytest
from scipy.special import comb

from mmcore.numeric._work_budget import SoftWorkBudget
from mmcore.numeric.intersection.ssx._ssx_boundary_strata import exact_boundary_strata_ssx


def _plane():
    return np.array([[[s, t, 0.] for t in (0., 1.)] for s in (0., 1.)])


def _corner_graph(degree=1):
    return np.array([[[s, t, (1-s)*(1-t)]
                      for t in np.linspace(0., 1., degree+1)]
                     for s in np.linspace(0., 1., degree+1)])


def _solve(a, b, atol=1e-6, rational=False, max_cells=100000):
    budget = SoftWorkBudget(max_cells, 1000)
    result = exact_boundary_strata_ssx(a, b, atol, rational, budget)
    if result is not None:
        result.update(budget.result_fields())
    return result


def _eval(surface, uv):
    basis = []
    for axis, coordinate in enumerate(uv):
        degree = surface.shape[axis]-1
        indices = np.arange(degree+1)
        basis.append(comb(degree, indices)*coordinate**indices*(1-coordinate)**(degree-indices))
    return np.einsum('i,j,ijc->c', *basis, surface)


@pytest.mark.parametrize('swap', [False, True])
@pytest.mark.parametrize('degree', [1, 4])
def test_nonnegative_height_exhausts_two_boundary_edges(degree, swap):
    a, b = _corner_graph(degree), _plane()
    result = _solve(*((b, a) if swap else (a, b)))
    assert result['complete'] and result['points'] == []
    assert len(result['branches']) == 2
    edges = []
    for branch in result['branches']:
        assert branch.overlap and branch.kind == 'overlap'
        stuv, xyz = branch.curve
        edges.append(tuple(map(tuple, xyz)))
        own = stuv[:, 2:] if swap else stuv[:, :2]
        np.testing.assert_array_equal(own, xyz[:, :2])
    assert set(edges) == {((1., 0., 0.), (1., 1., 0.)),
                          ((0., 1., 0.), (1., 1., 0.))}


def test_uniform_positive_homogeneous_weights_preserve_boundary_loci():
    nets = [np.concatenate((a*w, np.full(a.shape[:2]+(1,), w)), axis=2)
            for a, w in zip((_corner_graph(), _plane()), (2., .5))]
    result = _solve(*nets, rational=True)
    assert result['complete'] and len(result['branches']) == 2


@pytest.mark.parametrize('swap', [False, True])
@pytest.mark.parametrize('rational', [False, True])
def test_shared_source_corner_has_tangent_junction_on_nonaffine_convex_target(swap, rational):
    source = _corner_graph()
    source[..., :2] = .25+.25*source[..., :2]
    target = np.array([[[u+.25*u*v,v+.125*u*v,0.]
                        for v in (0.,1.)] for u in (0.,1.)])
    nets = [source,target]
    if rational:
        nets = [np.concatenate((net*w,np.full(net.shape[:2]+(1,),w)),axis=-1)
                for net,w in zip(nets,(2.,.5))]
    if swap:
        nets.reverse()
    result = _solve(*nets,atol=1e-5,rational=rational)
    assert result['complete'] and len(result['branches']) == 2
    junction, = result['singularities']
    assert junction.kind == 'tangent_point'
    np.testing.assert_array_equal(junction.xyz,[.5,.5,0.])
    np.testing.assert_array_equal(junction.stuv[2:] if swap else junction.stuv[:2],[1.,1.])
    assert {index for index,_ in junction.branch_links} == {0,1}


@pytest.mark.parametrize('swap', [False, True])
@pytest.mark.parametrize('flat', [False, True])
def test_opposed_halfspace_boundary_strata_have_exact_junction(flat, swap):
    # Both charts have z=0 only on their last row/column. Their open
    # images lie on opposite sides, which forces every common preimage
    # onto those exact boundary strata.
    a = _corner_graph()
    b = _corner_graph()
    b[..., 2] *= 0. if flat else -2.
    result = _solve(*((b, a) if swap else (a, b)))
    assert result is not None and result['complete']
    assert len(result['branches']) == 2 and not result['points']
    tangencies = [g for g in result['singularities'] if g.kind == 'tangent_point']
    assert len(tangencies) == 1
    np.testing.assert_array_equal(tangencies[0].stuv, np.ones(4))
    np.testing.assert_array_equal(tangencies[0].xyz, [1., 1., 0.])


def test_opposed_halfspaces_do_not_require_matching_control_corners():
    a = _corner_graph()
    b = _corner_graph()
    b[..., :2] = 2.*b[..., :2]-.5
    b[..., 2] *= -1.
    # The two L-shaped zero strata are disjoint although their planar
    # bounding rectangles overlap. A projected rectangle is insufficient.
    result = _solve(a, b)
    assert result is not None and result['complete']
    assert result['branches'] == [] and result['points'] == []


def test_rounded_endpoint_alias_cannot_establish_exact_junction():
    from fractions import Fraction
    from mmcore.numeric.intersection.ssx._ssx_opposed_strata import append_exact_junctions
    from mmcore.numeric.intersection.ssx._ssx_substrate import SSXBranch
    exact = [[[(Fraction(float(x)), Fraction(float(y)), Fraction(float(z)))
               for x, y, z in row] for row in source] for source in (_corner_graph(), _plane())]
    q = np.ones((2, 4))
    xyz = np.array([[1., 1., 0.], [1., 1., 0.]])
    first, second = SSXBranch((q.copy(), xyz.copy())), SSXBranch((q.copy(), xyz.copy()))
    root = (Fraction(1),)*4
    other = (Fraction(1)-Fraction(1, 2**55),)+root[1:]
    assert tuple(map(float, other)) == tuple(map(float, root))
    first._exact_endpoint_keys = (root, root)
    second._exact_endpoint_keys = (other, other)
    result = dict(branches=[first, second], points=[], singularities=[], unresolved_regions=[])
    budget = SoftWorkBudget(10000, 100)
    append_exact_junctions(result, exact, budget)
    assert result['singularities'] == []
    assert 'parameter_representation' in budget.result_fields()['status']['reasons']


def test_exact_convex_clip_retains_maximal_source_intervals():
    a = _corner_graph()
    a[:, :, :2] = a[:, :, :2]*2.-.5
    # Only the source t=1 edge intersects this target rectangle.
    b = _plane()
    b[:, :, 1] *= 2.
    result = _solve(a, b)
    assert result['complete'] and len(result['branches']) == 1
    stuv, xyz = result['branches'][0].curve
    np.testing.assert_array_equal(stuv[:, :2], [[.25, 1.], [.75, 1.]])
    np.testing.assert_array_equal(xyz, [[0., 1.5, 0.], [1., 1.5, 0.]])


def test_nonaffine_convex_target_inverse_is_resolved_along_whole_lifted_path():
    a = np.array([[[.25+.5*s, .5+.5*t, t] for t in (0., 1.)] for s in (0., 1.)])
    b = np.array([[[u, (1+u)*v, 0.] for v in (0., 1.)] for u in (0., 1.)])
    atol = 1e-5
    result = _solve(a, b, atol=atol)
    assert result['complete'] and len(result['branches']) == 1
    stuv, xyz = result['branches'][0].curve
    assert len(stuv) > 2  # The target's inverse of this straight line curves.
    np.testing.assert_allclose(stuv[:, 3], .5/(1+stuv[:, 2]), rtol=0., atol=1e-14)
    for index in range(len(stuv)-1):
        for fraction in np.linspace(0., 1., 9):
            q = (1-fraction)*stuv[index]+fraction*stuv[index+1]
            x = (1-fraction)*xyz[index]+fraction*xyz[index+1]
            assert np.linalg.norm(_eval(a, q[:2])-x) <= atol/2
            assert np.linalg.norm(_eval(b, q[2:])-x) <= atol/2


def test_isolated_corner_roots_are_not_promoted_to_edges():
    a = _plane()
    a[:, :, 2] = [[0., 1.], [1., 0.]]
    result = _solve(a, _plane())
    assert result['complete'] and result['branches'] == []
    assert len(result['points']) == 2
    assert {tuple(point.stuv) for point in result['points']} == {(0.,)*4, (1.,)*4}


def test_two_incident_edges_clipped_to_same_corner_publish_one_point():
    a = _corner_graph()
    a[:, :, :2] -= 1.
    result = _solve(a, _plane())
    assert result['complete'] and result['branches'] == []
    assert len(result['points']) == 1
    np.testing.assert_array_equal(result['points'][0].stuv, [1., 1., 0., 0.])


def test_distinct_source_boundary_preimages_of_same_line_remain_distinct():
    a = np.array([[[0., t, height] for t in (0., 1.)] for height in (0., .5, 0.)])
    result = _solve(a, _plane())
    assert result['complete'] and len(result['branches']) == 2
    first, second = result['branches']
    np.testing.assert_array_equal(first.curve[1], second.curve[1])
    assert {first.curve[0][0, 0], second.curve[0][0, 0]} == {0., 1.}


def test_strictly_positive_small_height_is_empty_independently_of_atol():
    a = _corner_graph()
    a[:, :, 2] += 2.**-40
    result = _solve(a, _plane(), atol=.001)
    assert result['complete'] and result['branches'] == [] and result['points'] == []


@pytest.mark.parametrize('mode', ['sign_changes', 'coincident', 'curved_edge', 'nonuniform'])
def test_unsupported_topology_stays_with_general_search(mode):
    a, b, rational = _corner_graph(), _plane(), False
    if mode == 'sign_changes':
        a[:, :, 2] = [[1., -1.], [-1., 1.]]
    elif mode == 'coincident':
        a[:, :, 2] = 0.
    elif mode == 'curved_edge':
        a = _corner_graph(2)
        a[1, -1, 1] += .25
    else:
        a = np.concatenate((a, np.ones((2, 2, 1))), axis=2)
        b = np.concatenate((b, np.ones((2, 2, 1))), axis=2)
        a[0, 0, 3] = 2.
        rational = True
    assert _solve(a, b, rational=rational) is None


def test_unrepresentable_clipped_interval_stays_explicitly_unresolved():
    a = np.array([[[x, t, t] for t in (0., 1.)] for x in (-2.**54, 2.**54)])
    result = _solve(a, _plane(), atol=.001)
    assert not result['complete'] and result['branches'] == []
    assert 'parameter_representation' in result['status']['reasons']


def test_denied_representation_budget_returns_a_typed_pending_stratum():
    result = _solve(_corner_graph(), _plane(), max_cells=25)
    assert not result['complete']
    assert result['branches'] == []
    assert result['unresolved_regions']
    assert 'work_budget' in result['status']['reasons']


def test_finite_homogeneous_input_with_unrepresentable_xyz_is_typed_partial():
    weight = np.nextafter(0., 1.)
    a, b = [np.concatenate((net, np.full((2, 2, 1), weight)), axis=2)
            for net in (_corner_graph(), _plane())]
    result = _solve(a, b, rational=True)
    assert not result['complete'] and result['branches'] == []
    assert 'parameter_representation' in result['status']['reasons']


def test_budget_interruption_preserves_already_validated_edge_prefix():
    a = np.array([[[.25+.5*s, .5+.5*t, t] for t in (0., 1.)] for s in (0., 1.)])
    b = np.array([[[u, (1+u)*v, 0.] for v in (0., 1.)] for u in (0., 1.)])
    result = _solve(a, b, atol=1e-5, max_cells=200)
    assert not result['complete'] and result['unresolved_regions']
    assert len(result['branches']) == 1
    stuv, xyz = result['branches'][0].curve
    assert stuv[0, 0] == 0. and 0. < stuv[-1, 0] < 1.
    for q, x in zip((stuv[:-1]+stuv[1:])/2, (xyz[:-1]+xyz[1:])/2):
        assert np.linalg.norm(_eval(a, q[:2])-x) <= 5e-6
        assert np.linalg.norm(_eval(b, q[2:])-x) <= 5e-6
