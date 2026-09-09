"""Boundary-root identity requires a common slice and isolation proof."""
import numpy as np
import pytest
from fractions import Fraction

from mmcore.numeric.intersection.ssx._ssx_root_identity import BoundaryRootIdentity
from mmcore.numeric.intersection.ssx._ssx_root_identity import _affine_interval
from mmcore.numeric.intersection.ssx._bez_ssx5 import BoundaryPoint, _dedup_crossings


def _homogeneous(net):
    return np.concatenate([net, np.ones(net.shape[:-1]+(1,))], axis=-1)


@pytest.mark.parametrize('side',[0,1])
def test_broad_source_enclosure_refines_to_the_same_algebraic_boundary_root(side):
    from examples.ssx.ssx5_analytic_audit import graph_pair,case_components
    from mmcore.numeric.intersection.ssx._ssx_planar_cut import exact_source_planar_cut
    a,b = map(_homogeneous,graph_pair(case_components('four_lines')))
    identity = BoundaryRootIdentity(a,b)
    identity.source_census = lambda axis,value,box: exact_source_planar_cut(
        a,b,axis,value,box,max_cells=1000)
    q = np.array([.125,float(side),.25,(side+.25)/1.5])
    old = np.column_stack((q-.001,q+.001))
    old[1] = side,side
    old[3] = ((1/6,np.nextafter(1/6,np.inf)) if side == 0 else
              (np.nextafter(5/6,-np.inf),5/6))
    point = BoundaryPoint(q,np.array([.125,float(side),0.]),(1,side),root_box=old.copy())
    point._source_root_box = True
    refined = identity.refine_enclosure(point,np.full(4,1e-4))
    assert refined is not None
    assert np.all(refined[:,0] >= old[:,0]) and np.all(refined[:,1] <= old[:,1])
    assert refined[0,1]-refined[0,0] < 1e-14
    assert identity.source_certificates[id(point)]['owner'] is point
    np.testing.assert_array_equal(point.root_box,old)  # Caller commits the box.


def test_failed_enclosure_refinement_does_not_replace_old_source_identity(monkeypatch):
    a = _homogeneous(np.array([[[s,t,0.] for t in (0.,1.)] for s in (0.,1.)]))
    identity = BoundaryRootIdentity(a,a)
    point = BoundaryPoint(np.array([.25,0.,.25,0.]),np.zeros(3),(1,0),
                          root_box=np.array([[.24,.26],[0.,0.],[.24,.26],[0.,0.]]))
    point._source_root_box = True
    original = {'owner':point,'tag':'old-source-proof'}
    identity.source_certificates[id(point)] = original
    def another_root(trial,radii):
        identity.source_certificates[id(trial)] = {'owner':trial,'tag':'different-root'}
        return np.array([[.74,.76],[0.,0.],[.74,.76],[0.,0.]])
    monkeypatch.setattr(identity,'enclose',another_root)
    assert identity.refine_enclosure(point,np.full(4,.001)) is None
    assert identity.source_certificates == {id(point):original}


def test_denied_source_refinement_preserves_box_and_skips_candidate_work(monkeypatch):
    a = _homogeneous(np.array([[[s,t,0.] for t in (0.,1.)] for s in (0.,1.)]))
    identity = BoundaryRootIdentity(a,a,charge=lambda _:False)
    point = BoundaryPoint(np.full(4,.5),np.zeros(3),(1,0),root_box=np.array([[.4,.6]]*4))
    point._source_root_box = True
    def forbidden(*args,**kwargs):
        raise AssertionError('Unpaid candidate isolation')
    monkeypatch.setattr(identity,'enclose',forbidden)
    assert identity.refine_enclosure(point,np.full(4,.01)) is None
    np.testing.assert_array_equal(point.root_box,np.array([[.4,.6]]*4))


def test_independent_representatives_on_equivalent_exact_slices_coalesce():
    a = np.array([[[s,t,0.] for t in (0.,1.)] for s in (0.,1.)])
    b = np.array([[[u,v,v-.5] for v in (0.,1.)] for u in (0.,1.)])
    identity = BoundaryRootIdentity(_homogeneous(a), _homogeneous(b))
    first = BoundaryPoint(np.array([.25,.5,.25+1e-15,.5+1e-15]),
                          np.array([.25,.5,0.]), (0,-1))
    second = BoundaryPoint(np.array([.25+1e-15,.5-1e-15,.25,.5]),
                           np.array([.25,.5,0.]), (2,-1))
    assert identity(first, second, np.full(4,1e-3), 1e-3)


def test_close_distinct_roots_fail_common_face_uniqueness():
    left, right = .5-2.**-20, .5+2.**-20
    z = np.array([left*right, left*right-(left+right)/2,
                  (1-left)*(1-right)])
    a = np.array([[[s,t,z[i]] for t in (0.,1.)]
                  for i,s in enumerate((0.,.5,1.))])
    b = np.array([[[u,v,0.] for v in (0.,1.)] for u in (0.,1.)])
    identity = BoundaryRootIdentity(_homogeneous(a), _homogeneous(b))
    points = []
    for x in (left,right):
        root = np.array([x,0.,x,0.])
        points.append(BoundaryPoint(root,np.array([x,0.,0.]),(1,0),
                                    root_box=np.column_stack((root,root))))
    assert not identity(*points,np.full(4,1e-3),1e-3)
    assert len(_dedup_crossings(points,1e-3,param_tol=np.full(4,1e-3),
                                root_matcher=identity)) == 2


def test_two_near_corner_crossings_of_one_arc_are_distinct_events():
    tiny = 2.**-20
    a = np.array([[[s,t,0.] for t in (0.,1.)] for s in (0.,1.)])
    b = np.array([[[u,v,u+v-tiny] for v in (0.,1.)] for u in (0.,1.)])
    identity = BoundaryRootIdentity(_homogeneous(a), _homogeneous(b))
    first = BoundaryPoint(np.array([0.,tiny,0.,tiny]),np.array([0.,tiny,0.]),(0,0))
    second = BoundaryPoint(np.array([tiny,0.,tiny,0.]),np.array([tiny,0.,0.]),(1,0))
    assert not identity(first,second,np.full(4,1e-3),1e-3)


def _certified_cut_root(identity, first, second, axis, cut, choose=0):
    from mmcore.numeric.intersection.ssx._ssx_planar_cut import exact_source_planar_cut
    result = exact_source_planar_cut(first, second, axis, cut, ((0.,1.),)*4,
                                     max_cells=1000)
    assert result['boundary_topology_complete']
    root = result['isolated'][choose]
    point = BoundaryPoint(np.asarray(root['stuv']), np.asarray(root['point']),
                          (axis,-1), root_box=np.asarray(root['parameter_root_box']))
    point._source_root_box = True
    assert identity.register_source_root(point, root['source_cut_certificate'])
    return point


def test_exact_singleton_source_root_is_identical_across_different_face_polynomials():
    coordinates, square = (0.,.5,1.), (1.,-1.,1.)
    graph = np.array([[[s,t,square[i]-square[j]] for j,t in enumerate(coordinates)]
                      for i,s in enumerate(coordinates)])
    plane = np.array([[[u,v,0.] for v in (-.5,1.5)] for u in (-.5,1.5)])
    first, second = _homogeneous(graph), _homogeneous(plane)
    identity = BoundaryRootIdentity(first, second)
    a = _certified_cut_root(identity, first, second, 0, 0.)
    b = _certified_cut_root(identity, first, second, 1, 0.)
    np.testing.assert_array_equal(a.stuv, [0.,0.,.25,.25])
    assert identity.source_certificates[id(a)]['varying_axis'] != identity.source_certificates[id(b)]['varying_axis']
    assert identity(a,b,np.full(4,1e-3),1e-3)
    assert len(_dedup_crossings([a,b],1e-3,param_tol=np.full(4,1e-3),
                                root_matcher=identity)) == 1


def test_different_exact_singleton_source_roots_at_nearby_faces_stay_distinct():
    tiny = 2.**-20
    graph = np.array([[[s,t,s+t-tiny] for t in (0.,1.)] for s in (0.,1.)])
    plane = np.array([[[u,v,0.] for v in (0.,1.)] for u in (0.,1.)])
    first, second = _homogeneous(graph), _homogeneous(plane)
    identity = BoundaryRootIdentity(first, second)
    a = _certified_cut_root(identity, first, second, 0, 0.)
    b = _certified_cut_root(identity, first, second, 1, 0.)
    assert not identity(a,b,np.full(4,1e-3),1e-3)


def test_incident_corner_faces_obtain_own_source_certificates_before_identity():
    from mmcore.numeric.intersection.ssx._ssx_planar_cut import exact_source_planar_cut
    graph = np.array([[[s,t,s+t] for t in (0.,1.)] for s in (0.,1.)])
    plane = np.array([[[u,v,0.] for v in (0.,1.)] for u in (0.,1.)])
    first, second = _homogeneous(graph), _homogeneous(plane)
    identity = BoundaryRootIdentity(first, second)
    identity.source_census = lambda axis,value,box: exact_source_planar_cut(
        first, second, axis, value, box, max_cells=1000)
    a = BoundaryPoint(np.zeros(4),np.zeros(3),(0,0))
    b = BoundaryPoint(np.zeros(4),np.zeros(3),(1,0))
    assert identity(a,b,np.full(4,1e-3),1e-3)
    assert id(a) in identity.source_certificates and id(b) in identity.source_certificates


def test_affine_root_interval_encloses_exact_binary_mapping():
    lo, hi, parameter = .20323707831873447, .7064770887111032, .569861747188787
    lower, upper = _affine_interval(lo, hi, parameter, parameter)
    exact = Fraction(lo)+(Fraction(hi)-Fraction(lo))*Fraction(parameter)
    assert Fraction(lower) <= exact <= Fraction(upper)


def test_denied_identity_work_retains_events_before_certificate_allocation():
    a = np.array([[[s,t,0.] for t in (0.,1.)] for s in (0.,1.)])
    b = np.array([[[u,v,v-.5] for v in (0.,1.)] for u in (0.,1.)])
    charges = []
    identity = BoundaryRootIdentity(_homogeneous(a), _homogeneous(b),
                                    charge=lambda amount: charges.append(amount) or False)
    first = BoundaryPoint(np.array([.25,.5,.25,.5]), np.array([.25,.5,0.]), (0,-1))
    second = BoundaryPoint(np.array([.25,.5,.25,.5]), np.array([.25,.5,0.]), (2,-1))
    assert not identity(first, second, np.full(4,1e-3), 1e-3)
    assert charges == [1]
    assert not identity.faces and not identity.enclosures


def test_exact_affine_boundary_reduction_certifies_irrational_root_identity():
    # The source t=0 face and target v=0 face share the irrational endpoint
    # s=u=sqrt(1/2). Full 3D strict inclusion cannot cover v=0; the exact
    # y=t=v identity reduces this to the square interior problem (x,z).
    a = np.array([[[s,t,z] for t in (0.,1.)]
                  for s,z in ((0.,-.5),(.5,-.5),(1.,.5))])
    b = np.array([[[u,v,0.] for v in (0.,1.)] for u in (0.,1.)])
    identity = BoundaryRootIdentity(_homogeneous(a), _homogeneous(b))
    root = np.sqrt(.5)
    first = BoundaryPoint(np.array([root,0.,np.nextafter(root,1.),0.]),
                          np.array([root,0.,0.]), (1,0))
    second = BoundaryPoint(np.array([np.nextafter(root,1.),0.,root,0.]),
                           np.array([root,0.,0.]), (3,0))
    radii = np.full(4,1e-3)
    assert identity(first,second,radii,1e-3)
    enclosure = identity.enclose(first,radii)
    assert enclosure is not None
    assert np.all(enclosure[[1,3]] == 0.)


def test_root_identity_refines_oversized_proposal_without_merging_neighbor():
    left, right = 15/32, 17/32
    z = [left*right, left*right-.5, (1-left)*(1-right)]
    a = np.array([[[s,t,z[i]] for t in (0.,1.)]
                  for i,s in enumerate((0.,.5,1.))])
    b = np.array([[[u,v,0.] for v in (0.,1.)] for u in (0.,1.)])
    identity = BoundaryRootIdentity(_homogeneous(a), _homogeneous(b))
    first = BoundaryPoint(np.array([left,.5,left,.5]),np.array([left,.5,0.]),(1,-1))
    second = BoundaryPoint(np.array([np.nextafter(left,1.),.5,left,.5]),
                           np.array([left,.5,0.]),(1,-1))
    neighbor = BoundaryPoint(np.array([right,.5,right,.5]),
                             np.array([right,.5,0.]),(1,-1))
    # The initial proposal contains both roots. Refinement must isolate
    # each existing root; the final union still contains both enclosures
    # when different roots are compared, so they cannot coalesce.
    assert identity(first,second,np.full(4,.25),1.)
    assert not identity(first,neighbor,np.full(4,.25),1.)


def test_singular_root_refusal_stops_at_source_precision_without_spending_ledger():
    # z=(s-.5)^2 has an exact repeated face root. Regular Krawczyk
    # inclusion cannot isolate it; shrinking to float spacing only spends
    # work while the inherited source uncertainty already dominates z.
    a = np.array([[[s,t,z] for t in (0.,1.)]
                  for s,z in ((0.,.25),(.5,-.25),(1.,.25))])
    b = np.array([[[u,v,0.] for v in (0.,1.)] for u in (0.,1.)])
    spent = [0]

    def charge(amount):
        if spent[0]+amount > 30:
            return False
        spent[0] += amount
        return True

    identity = BoundaryRootIdentity(_homogeneous(a), _homogeneous(b), charge=charge)
    root = np.array([.5,.5,.5,.5])
    first = BoundaryPoint(root.copy(),np.array([.5,.5,0.]),(1,-1))
    second = BoundaryPoint(root.copy(),np.array([.5,.5,0.]),(1,-1))
    assert identity.enclose(first,np.full(4,1e-3)) is None
    assert not identity.exhausted
    # Refusal preserves the independent events; it does not identify them
    # or assert that the singular root is absent.
    assert not identity(first,second,np.full(4,1e-3),1e-3)
    assert len(_dedup_crossings([first,second],1e-3,
                                param_tol=np.full(4,1e-3),root_matcher=identity)) == 2
