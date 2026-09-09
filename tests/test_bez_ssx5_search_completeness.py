"""Analytic regressions for branch ownership on closed parameter cells."""
import numpy as np

from mmcore.numeric.intersection.ssx import _bez_ssx5 as ssx


def _parallel_pair():
    a, b = .49, .4905
    z = 32 * np.array([a*b, a*b-(a+b)/2, (1-a)*(1-b)])
    s1 = np.array([[[16*s, t, z[i]] for t in [0., 1.]]
                   for i, s in enumerate([0., .5, 1.])])
    s2 = np.array([[[16*s, t, 0.] for t in [0., 1.]]
                   for s in [0., 1.]])
    return s1, s2


def test_boundary_root_dedup_preserves_metric_distinct_parallel_branches():
    # Two exact lines are only .0005 apart in each surface's parameter,
    # but .008 apart in xyz: eight times the requested geometric tolerance.
    s1, s2 = _parallel_pair()
    crossings, overlaps = ssx._find_ssx_boundary_zeros(
        s1, s2, atol=.001, rational=False)
    assert not overlaps
    # Independently solved adjacent faces may retain two numerical
    # representatives until root identity is certified. Every analytic
    # boundary event must remain available.
    for x in (7.84, 7.848):
        for y in (0., 1.):
            assert any(np.linalg.norm(c.xyz-np.array([x, y, 0.])) < 1e-8
                       for c in crossings)


def test_subdivision_preserves_cut_roots_near_isoline_endpoints(monkeypatch):
    # Exercise the closed-face ownership stage directly: classification is
    # intentionally inconclusive so one split occurs before the depth limit.
    eps = 2e-7
    s1 = np.array([[[s, t, 0.] for t in [0., 1.]] for s in [0., 1.]])
    s2 = np.array([[[s, t, t-eps] for t in [0., 1.]] for s in [0., 1.]])
    monkeypatch.setattr(ssx, '_check_loop_free', lambda *a, **k: False)
    monkeypatch.setattr(ssx, '_check_tangency', lambda *a, **k: None)
    monkeypatch.setattr(ssx, '_compute_split_plan',
                        lambda *a, **k: (None, None, None, None))
    result = ssx.bez_ssx(s1, s2, atol=1e-9, rational=False, max_depth=1,
                         max_xyz_step=.1)
    target = np.array([.5, eps, .5, eps])
    assert any(np.max(np.abs(p.stuv-target)) < 1e-9
               for p in result['points'])
    assert not result['complete']
    assert 'depth_limit' in result['status']['reasons']


def test_closed_branch_metadata_matches_exact_lifted_chain_closure():
    q = np.array([[.2, .2, .2, .2], [.8, .2, .8, .2],
                  [.5, .8, .5, .8], [.2, .2, .2, .2]])
    xyz = np.column_stack([q[:, :2], np.zeros(len(q))])
    fragment = ssx._Fragment(None, None, q, xyz)
    result = ssx._assemble_fragments([fragment])
    assert len(result) == 1
    assert result[0].closed


def test_same_xyz_with_different_parameter_endpoints_is_not_closed():
    q = np.array([[.2, .2, .2, .2], [.8, .2, .8, .2],
                  [.5, .8, .5, .8], [.3, .2, .2, .2]])
    xyz = np.array([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,0.]])
    result = ssx._assemble_fragments([ssx._Fragment(None, None, q, xyz)])
    assert len(result) == 1
    assert not result[0].closed


def test_failed_regular_march_keeps_cell_in_subdivision_frontier(monkeypatch):
    s1 = np.array([[[s, t, 0.] for t in [0., 1.]] for s in [0., 1.]])
    s2 = np.array([[[s, t, t-.5] for t in [0., 1.]] for s in [0., 1.]])

    def fail_march(a, b, x, **kwargs):
        return np.array([x]), np.array([ssx.eval_surface(a, *x[:2])]), None

    monkeypatch.setattr(ssx, '_march_to_boundary', fail_march)
    result = ssx.bez_ssx(s1, s2, rational=False, max_depth=1, max_xyz_step=.1)
    assert not result['complete']
    assert 'depth_limit' in result['status']['reasons']
    assert result['unresolved_regions']
    assert result['status']['work']['cell_counts']['ssx'] > 1


def test_default_depth_uses_remaining_work_to_complete_a_source_loop():
    from examples.ssx.bez_ssx5_case11 import S1,S2
    kwargs = dict(atol=1e-3,rational=False,max_cells=60000)
    limited = ssx.bez_ssx(S1,S2,max_depth=13,**kwargs)
    assert not limited['complete']
    assert limited['status']['reasons'] == ['depth_limit']
    stopped = limited['status']['work']['cells_processed']
    assert stopped < .9*kwargs['max_cells']
    completed = ssx.bez_ssx(S1,S2,**kwargs)
    assert completed['complete'],completed['status']
    branch, = completed['branches']
    assert branch.closed
    length = np.linalg.norm(np.diff(branch.curve[1],axis=0),axis=1).sum()
    assert 1.52 < length < 1.55
    assert stopped < completed['status']['work']['cells_processed'] < kwargs['max_cells']


def test_tangent_witness_at_size_floor_keeps_unresolved_neighborhood(monkeypatch):
    # The exact source paraboloid has one root at the center. A witness
    # there establishes existence; it does not supply a complete local
    # zero-set certificate to the generic singular search.
    a = np.array([[[s, t, x+y] for t, y in zip((0., .5, 1.), (.25, -.25, .25))]
                  for s, x in zip((0., .5, 1.), (.25, -.25, .25))])
    b = np.array([[[s, t, 0.] for t in (0., 1.)] for s in (0., 1.)])
    from mmcore.nurbs import _nurbs_param_tol
    monkeypatch.setattr(_nurbs_param_tol, 'bez_surface_param_tolerance',
                        lambda *args, **kwargs: (.25, .25))
    monkeypatch.setattr(ssx, '_check_loop_free', lambda *args, **kwargs: False)
    monkeypatch.setattr(ssx, '_check_tangency', lambda *args, **kwargs: True)
    witnesses = []
    def witness(*args,**kwargs):
        witnesses.append(args[0])
        return True,[np.full(4,.5)]
    monkeypatch.setattr(ssx, '_emit_tangent_roots',witness)
    monkeypatch.setattr(ssx, '_phi_slice_loop_fragments', lambda *args, **kwargs: [])
    # Exercise the generic witness path; exact quadratic elimination has
    # an independent complete singleton proof for this source pair.
    result = ssx.bez_ssx(a, b, rational=False,max_xyz_step=.1)
    assert witnesses
    assert not result['complete']
    assert 'unresolved_multiplicity' in result['status']['reasons']
    assert any(region['stuv_min'] == (0.,)*4 and region['stuv_max'] == (1.,)*4
               and region['reason'] == 'unresolved_multiplicity'
               for region in result['unresolved_regions'])


def test_strict_corner_cone_isolates_a_regular_clipped_arc():
    from types import SimpleNamespace
    s1 = np.array([[[s, t, 0., 1.] for t in [0., 1.]] for s in [0., 1.]])
    s2 = np.array([[[-u, v, u-v, 1.] for v in [0., 1.]] for u in [0., 1.]])
    cell = SimpleNamespace(g1=SimpleNamespace(surface=s1),
                           g2=SimpleNamespace(surface=s2))
    assert ssx._strict_corner_touch(cell, np.zeros(4))


def test_source_corner_cone_uses_proven_face_membership_and_source_derivatives():
    from types import SimpleNamespace
    from mmcore.numeric.intersection.ssx._ssx_cofactor_bounds import SourceCofactorBounds
    eps = 2.**-20
    a = np.array([[[s, t, 0., 1.] for t in (0., 1.)] for s in (0., 1.)])
    for offset, expected in ((0., True), (eps, False)):
        b = np.array([[[-u+offset, v, u-v, 1.] for v in (0., 1.)] for u in (0., 1.)])
        # The rounded representative says s=0; the authoritative root
        # says s=offset. For a positive offset one germ enters the cell.
        root = np.array([offset, 0., 0., 0.])
        point = ssx.BoundaryPoint(np.zeros(4), np.zeros(3), (0, 0))
        point.root_box = np.repeat(root[:, None], 2, axis=1)
        point._source_root_box = True
        cell = SimpleNamespace(g1=SimpleNamespace(surface=a),
                               g2=SimpleNamespace(surface=b), box=((0., 1.),)*4,
                               source_cofactors=SourceCofactorBounds(a, b))
        assert ssx._strict_corner_touch(cell, np.zeros(4), point) is expected


def test_tangent_output_identity_preserves_close_distinct_preimages(monkeypatch):
    from fractions import Fraction
    from math import comb
    from mmcore.numeric.intersection.ssx._ssx_substrate import GaussMapBern
    # Exact dyadic coefficients of 3*((s-1/4)*(s-3/4))**2
    # +3*(t-1/2)**2 give two isolated touches. Shrinking only x/y
    # puts both spatial images inside one geometric tolerance.
    powers = [Fraction(27, 256), Fraction(-9, 8), Fraction(33, 8), -6, 3]
    height = [sum(powers[j]*Fraction(comb(i, j), comb(4, j))
                  for j in range(i+1)) for i in range(5)]
    assert all(Fraction(float(value)) == value for value in height)
    a = np.array([[[s*2.**-20, t*2.**-20, float(z+y)]
                   for t, y in zip((0., .5, 1.), (Fraction(3,4), Fraction(-3,4), Fraction(3,4)))]
                  for s, z in zip((0., .25, .5, .75, 1.), height)])
    b = np.array([[[s*2.**-20, t*2.**-20, 0.] for t in (0., 1.)] for s in (0., 1.)])
    roots = [np.array([s, .5, s, .5]) for s in (.25, .75)]
    cell = ssx._Cell(GaussMapBern.from_surf(a), GaussMapBern.from_surf(b), [], ((0., 1.),)*4)
    from mmcore.numeric.intersection._root_box_certificate import exact_bernstein_value
    cell.source_root_exact = lambda q: exact_bernstein_value(a, q[:2]) == exact_bernstein_value(b, q[2:])
    monkeypatch.setattr(ssx, '_tangency_witness', lambda *args, **kwargs: (True, roots+roots[:1], 0., False))
    monkeypatch.setattr(ssx, '_delta_float_gn', lambda *args, **kwargs: (lambda x: x, None))
    monkeypatch.setattr(ssx, '_delta_root_local_dimension', lambda *args, **kwargs: False)
    monkeypatch.setattr(ssx, '_cell_ptol4', lambda *args, **kwargs: np.ones(4))
    found = []
    ssx._emit_tangent_roots(cell, .001, np.ones(4), found,
                           overlap_boxes=[np.array([[0., 1.]]*4)])
    assert len(found) == 2
    assert {tuple(point.stuv) for point in found} == {tuple(root) for root in roots}


def test_unproved_tangent_candidate_is_a_local_obligation_not_a_confirmed_point():
    from types import SimpleNamespace
    budget = ssx._SSXSoftBudget(100, 10)
    diagnostics = []
    box = ((.25, .5),)*4
    cell = SimpleNamespace(box=box, work_budget=budget, unresolved_regions=diagnostics,
                           source_root_exact=lambda q: False)
    assert not ssx._record_unproved_tangent(cell, np.full(4, .375))
    assert budget.incomplete
    assert diagnostics == [{'stuv_min': (.25,)*4, 'stuv_max': (.5,)*4,
                            'reason': 'unresolved_multiplicity', 'candidate': (.375,)*4,
                            'candidate_kind': 'tangent_point', 'source_existence': False}]


def test_roundoff_close_distinct_lifted_endpoints_remain_open():
    q = np.array([[.5, .5, .5, .5], [.75, .75, .75, .75],
                  [np.nextafter(.5, 1.), .5, .5, .5]])
    xyz = np.array([[0., 0., 0.], [1., 1., 0.], [0., 0., 0.]])
    branch = ssx._assemble_fragments([ssx._Fragment(None, None, q, xyz)])[0]
    assert not branch.closed


def test_deflated_singleton_keeps_original_global_parameter_registration():
    net = np.zeros((2, 2, 4))
    net[..., 3] = 1.
    zero = np.zeros((1,)*4)
    global_point = ssx.BoundaryPoint(np.array([.5]*4), np.zeros(3), (0, 1))
    local_point = ssx.BoundaryPoint(np.array([1., .5, .5, .5]), np.zeros(3), (0, 1))
    fragments, points = ssx._deflate_tangent_cell(
        net, net, zero, zero, zero, zero, ((0., 1.),)*4,
        [local_point], .001, originals=[global_point])
    assert fragments == [] and len(points) == 1
    np.testing.assert_array_equal(points[0].stuv, global_point.stuv)
    assert points[0]._registered_root_id == id(global_point)


def test_fragment_containment_preserves_distinct_lifted_sheets():
    q = np.array([[.25, 0., .25, 0.], [.25, 1., .25, 1.]])
    xyz = np.array([[0., 0., 0.], [0., 1., 0.]])
    for displacement in (.5, 1e-9):
        mate = q.copy()
        mate[:, 0] += displacement
        first = ssx._Fragment(None, None, q, xyz)
        second = ssx._Fragment(None, None, mate, xyz)
        assert len(ssx._drop_duplicate_fragments([first, second], .001)) == 2
        assert len(ssx._assemble_fragments([first, second])) == 2


def test_identical_approximation_paths_without_shared_source_arc_are_retained():
    q = np.array([[.25, 0., .25, 0.], [.25, 1., .25, 1.]])
    xyz = np.array([[0., 0., 0.], [0., 1., 0.]])
    first = ssx._Fragment(None, None, q, xyz)
    second = ssx._Fragment(None, None, q[::-1], xyz[::-1])
    assert len(ssx._drop_duplicate_fragments([first, second], .001)) == 2


def test_overlap_endpoints_remain_available_to_incident_ordinary_arcs():
    # z=t*(t-s*(1-s)) has the boundary line t=0 AND the ordinary arch
    # t=s*(1-s), whose two endpoints are precisely that overlap's ends.
    q = np.array([0., -.5, 0.])  # Bernstein coefficients of s^2-s
    t = np.array([0., .5, 1.])
    t2 = np.array([0., 0., 1.])
    s1 = np.array([[[i/2, t[j], t2[j]+q[i]*t[j]]
                    for j in range(3)] for i in range(3)])
    s2 = np.array([[[s, v, 0.] for v in [0., 1.]] for s in [0., 1.]])
    crossings, overlaps = ssx._find_ssx_boundary_zeros(
        s1, s2, atol=1e-5, rational=False)
    assert overlaps
    assert any(np.max(np.abs(c.stuv)) < 1e-8 for c in crossings)
    assert any(np.max(np.abs(c.stuv-np.array([1., 0., 1., 0.]))) < 1e-8
               for c in crossings)


def test_restricted_squared_distance_roundoff_cannot_prune_exact_line():
    from mmcore.numeric.bern_sq_dist import surface_surface_distance_squared_net_homog
    from mmcore.numeric._bezier_common import restrict_net_axis_v
    # The exact binary input has z=a*(2*s-1), hence an exact line s=u=.5.
    # Its Gram-product squared net becomes strictly positive after a tight
    # restriction around that line: positivity here is floating error.
    slope = 0.8435019602765045
    a = np.array([[[s, t, z] for t in (0., 1.)]
                  for s, z in ((0., -slope), (1., slope))])
    b = a.copy()
    b[..., 2] = 0.
    f = surface_surface_distance_squared_net_homog(a, b, rational=False)
    lo, hi = .5-1e-9, .5+1e-9
    for axis in range(4):
        f = restrict_net_axis_v(f[..., None], axis, lo, hi, 0., 1.)[..., 0]
    for axis in range(2):
        a = restrict_net_axis_v(a, axis, lo, hi, 0., 1.)
        b = restrict_net_axis_v(b, axis, lo, hi, 0., 1.)
    atol = 1e-11
    assert f.min() > atol**2
    assert not ssx._prune_ssx_cell(a, b, atol, rational=False, F=f)


def test_regular_corner_does_not_launch_displaced_recovery(monkeypatch):
    from types import SimpleNamespace
    a = np.array([[[s, t, 0., 1.] for t in (0., 1.)] for s in (0., 1.)])
    b = np.array([[[-u, v, u-v, 1.] for v in (0., 1.)] for u in (0., 1.)])
    root = ssx.BoundaryPoint(np.zeros(4), np.zeros(3), (0, 0))
    cell = ssx._Cell(SimpleNamespace(surface=a), SimpleNamespace(surface=b),
                     [root], ((0., 1.),)*4)

    def invalid_launch(*args, **kwargs):
        raise AssertionError('strict clipped corner has no inward regular arc')

    monkeypatch.setattr(ssx, '_march_to_boundary', invalid_launch)
    fragments, points = ssx._trace_cell_by_registrations(cell, 1e-4)
    assert not fragments
    assert len(points) == 1
    assert not cell.trace_incomplete


def test_exact_point_cleanup_preserves_close_parameter_preimages():
    root = np.array([.5, .5, .5, .5])
    xyz = np.zeros(3)
    first = ssx.SSXPoint(root.copy(), xyz.copy())
    repeated = ssx.SSXPoint(root.copy(), xyz.copy())
    distinct = ssx.SSXPoint(root+np.array([1e-9, 0., 0., 0.]), xyz.copy())
    result = ssx._deduplicate_ssx_points(
        [first, repeated, distinct], np.full(4, 1e-3), 1e-3,
        exact_topology=True)
    assert len(result) == 2
    assert result[0] is first
    assert result[1] is distinct


def test_exact_boundary_source_with_unique_target_has_one_representation():
    # The intersection is the s=1 boundary of the first plane. Its exact
    # source-curve ownership and the second plane's unique inverse identify
    # regular tracing and boundary overlap as the same correspondence.
    a = np.array([[[s, t, 0.] for t in (0., 1.)] for s in (0., .5)])
    b = np.array([[[u, v, u-.5] for v in (0., 1.)] for u in (0., 1.)])
    result = ssx.bez_ssx(a, b, atol=1e-3, rational=False)
    assert result['complete'], result['status']['reasons']
    assert len(result['branches']) == 1
    path = np.asarray(result['branches'][0].curve[0])
    assert np.all(path[:, 0] == 1.)
    assert np.allclose(path[:, 2], .5, atol=1e-12)


def test_zero_minor_is_not_a_loop_absence_certificate():
    # A rank-three fiber of S1=(circle_factor,0,0), S2=(0,u,v)
    # has a closed (s,t) loop while its u/v tangent minors vanish.
    changing = np.array([-1.,1.])
    zero = np.zeros(2)
    assert ssx._check_monotonicity(changing,changing,zero,zero) == (False,None)
    assert ssx._check_monotonicity(zero,np.array([0.,1.]),zero,zero) == (True,1)


def test_representable_narrow_cell_keeps_its_exact_faces():
    width = 2.**-52
    box = ((0., width),)*4
    assert np.array_equal(ssx._global_to_local(np.zeros(4), box), np.zeros(4))
    assert np.array_equal(ssx._global_to_local(np.full(4, width), box), np.ones(4))


def test_small_physical_arc_requires_both_lifted_source_chords():
    from types import SimpleNamespace
    from mmcore.numeric.intersection._deflate import minors_Tpsi_from_control_nets
    from mmcore.numeric.intersection.ssx._ssx_affine_path import (
        affine_path_representation_bounded, source_box_image_diameter_bounded)
    from mmcore.numeric.intersection.ssx._ssx_cofactor_bounds import SourceCofactorBounds
    scale = 2.**-14
    a = np.array([[[scale*x,scale*y,0.] for y in (1/16,9/16)]
                  for x in (.25,.75)])
    b = np.array([[[scale*x,scale*y,y-xx] for y in (1/16,9/16)]
                  for x,xx in ((.25,1/16),(.5,3/16),(.75,9/16))])
    h = lambda net: np.concatenate((net,np.ones(net.shape[:-1]+(1,))),axis=-1)
    roots=[]
    for value in (0.,1.):
        q=np.full(4,value)
        root=ssx.BoundaryPoint(q,ssx.eval_surface(a,value,value,rational=False),
                              (0,int(value)),root_box=np.column_stack((q,q)))
        root._source_root_box=True
        roots.append(root)
    tensors=[np.asarray(t) for t in minors_Tpsi_from_control_nets(a,b)]
    checked=[]
    def representation(start,end,xyz):
        checked.append(True)
        return affine_path_representation_bounded(a,b,start,end,xyz,1e-3,rational=False)
    cell=SimpleNamespace(g1=SimpleNamespace(surface=h(a)),g2=SimpleNamespace(surface=h(b)),
                         crossings=roots,box=((0.,1.),)*4,boundary_complete=True,
                         T1=tensors[0],T2=tensors[1],T3=tensors[2],T4=tensors[3],
                         source_cofactors=SourceCofactorBounds(h(a),h(b)),
                         arc_image=lambda box,xyz: source_box_image_diameter_bounded(
                             a,b,box,xyz,1e-3,rational=False),
                         path_representation=representation)
    assert ssx._small_regular_cell_arc(cell,1e-3) is None
    assert checked
    assert ssx.eval_surface(b,.5,.5,rational=False)[2] == .0625


def test_postprocess_denial_preserves_discovered_ordinary_branch():
    a = np.array([[[s,t,0.] for t in (0.,1.)] for s in (0.,1.)])
    b = np.array([[[u,v,v-.5] for v in (0.,1.)] for u in (0.,1.)])
    result = ssx.bez_ssx(a,b,atol=1e-3,rational=False,max_postprocess_work=0,
                         max_xyz_step=.1)
    assert not result['complete']
    assert 'postprocess_cap' in result['status']['reasons']
    assert result['status']['work']['postprocess_work'] == 0
    assert result['branches']
    assert any(np.ptp(np.asarray(branch.curve[1])[:,0]) == 1.
               for branch in result['branches'])
