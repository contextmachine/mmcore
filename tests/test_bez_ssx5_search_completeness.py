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
    # Every metric-separated boundary event must remain available at
    # the requested modeling tolerance; these roots are eight atols apart.
    for x in (7.84, 7.848):
        for y in (0., 1.):
            assert any(np.linalg.norm(c.xyz-np.array([x, y, 0.])) <= .001
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


def test_closed_branch_metadata_matches_paired_chain_closure():
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
    assert 'unresolved_regions' not in result
    assert result['status']['work']['cell_counts']['ssx'] > 1


















def test_fragment_containment_preserves_distinct_lifted_sheets():
    q = np.array([[.25, 0., .25, 0.], [.25, 1., .25, 1.]])
    xyz = np.array([[0., 0., 0.], [0., 1., 0.]])
    for displacement in (.5,):
        mate = q.copy()
        mate[:, 0] += displacement
        first = ssx._Fragment(None, None, q, xyz)
        second = ssx._Fragment(None, None, mate, xyz)
        assert len(ssx._drop_duplicate_fragments([first, second], .001)) == 2
        assert len(ssx._assemble_fragments([first, second])) == 2




def test_overlap_endpoints_remain_available_to_incident_ordinary_arcs():
    # z=t*(t-s*(1-s)) has the boundary line t=0 AND the ordinary arch
    # t=s*(1-s), whose two endpoints are precisely that overlap's ends.
    q = np.array([0., -.5, 0.])  # Bernstein coefficients of s^2-s
    t = np.array([0., .5, 1.])
    t2 = np.array([0., 0., 1.])
    s1 = np.array([[[i/2, t[j], t2[j]+q[i]*t[j]]
                    for j in range(3)] for i in range(3)])
    s2 = np.array([[[s, v, 0.] for v in [0., 1.]] for s in [0., 1.]])
    result = ssx.bez_ssx(s1, s2, atol=1e-5, rational=False)
    paths = [np.asarray(branch.curve[1]) for branch in result['branches']]
    assert paths, result['status']
    # The overlap line and incident arch must both be represented in full.
    # Whether their shared endpoints are returned as standalone crossing
    # records is an internal ownership decision, not the CAD contract.
    for u in np.linspace(0., 1., 101):
        for v in (0., u*(1-u)):
            point = np.array([u, v, 0.])
            distances = []
            for path in paths:
                first, delta = path[:-1], np.diff(path, axis=0)
                square = np.maximum(np.einsum('ij,ij->i', delta, delta), 1e-300)
                q = np.clip(np.einsum('ij,ij->i', point-first, delta)/square, 0., 1.)
                distances.append(np.linalg.norm(first+q[:, None]*delta-point, axis=1).min())
            assert min(distances) <= 4e-5


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






def test_boundary_overlap_has_one_representation():
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


def test_valid_parabola_chord_is_not_deleted_by_midpoint_residual_estimate():
    # h=512*g*(2a²-g), g=v-(u-.5)². The lower zero arc g=0
    # lies <=a² from its endpoint chord. Both source evaluations along
    # the published lifted chord are also within 512*a^4=1/2048 of it.
    # At the midpoint grad(h)=0 while h=1/2048, so dividing this
    # residual by a clamped normal angle invents a large distance.
    from fractions import Fraction
    from math import comb
    from mmcore.numeric.intersection._root_box_certificate import exact_bernstein_value
    a = Fraction(1,32)
    square = {(0,0):Fraction(1,4),(1,0):Fraction(-1),(2,0):Fraction(1)}
    g = {key:-value for key,value in square.items()}
    g[0,1] = Fraction(1)
    opposite = {key:-value for key,value in g.items()}
    opposite[0,0] += 2*a*a
    power = {}
    for (i,j),left in g.items():
        for (k,l),right in opposite.items():
            key = i+k,j+l
            power[key] = power.get(key,Fraction(0))+512*left*right
    graph = np.zeros((5,3,4))
    for i in range(5):
        for j in range(3):
            h = sum(value*Fraction(comb(i,k),comb(4,k))*Fraction(comb(j,l),comb(2,l))
                    for (k,l),value in power.items() if k <= i and l <= j)
            exact = (3*i,6*j,12*h,12)
            graph[i,j] = list(map(float,exact))
            assert all(Fraction(float(x)) == y for x,y in zip(graph[i,j],exact))
    plane = np.array([[[s,t,0.,1.] for t in (0.,1.)] for s in (0.,1.)])
    q = np.array([[float(Fraction(1,2)+sign*a),float(a*a)]*2 for sign in (-1,1)])
    xyz = np.column_stack((q[:,:2],np.zeros(2)))
    for point in q:
        assert exact_bernstein_value(graph,point[2:])[2] == 0
    assert a*a < Fraction(1,1000)
    assert 512*a**4 < Fraction(1,1000)
    left,right = [ssx.BoundaryPoint(s,x,(0,-1)) for s,x in zip(q,xyz)]
    branch = ssx._assemble_fragments(
        [ssx._Fragment(left,right,q,xyz)], S1_full=plane,S2_full=graph,
        atol_full=.001,rational_full=True)
    assert len(branch) == 1
    np.testing.assert_array_equal(branch[0].curve[0],q)
