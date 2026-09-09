"""Exact source events can share every displayed floating coordinate."""
from fractions import Fraction

import numpy as np

from mmcore.numeric.intersection.ssx._bez_ssx5 import BoundaryPoint
from mmcore.numeric.intersection.ssx._ssx_affine_constraints import AffineParameterConstraints
from mmcore.numeric.intersection.ssx._ssx_planar_cut import exact_source_planar_cut
from mmcore.numeric.intersection.ssx._ssx_root_identity import BoundaryRootIdentity


def _source_events(cuts):
    coordinates, square = (0.,.5,1.), (1.,-1.,1.)
    graph = np.array([[[s,t,square[i]+square[j]] for j,t in enumerate(coordinates)]
                      for i,s in enumerate(coordinates)])
    plane = np.array([[[u,v,.5] for v in (-.5,1.5)] for u in (-.5,1.5)])
    first,second = [np.concatenate((net*2e-4,np.ones(net.shape[:2]+(1,))),axis=2)
                    for net in (graph,plane)]
    identity = BoundaryRootIdentity(first,second)
    identity.affine_constraints = AffineParameterConstraints(identity.affine)
    points = []
    for axis,value in cuts:
        result = exact_source_planar_cut(first,second,axis,value,((0.,1.),)*4,max_cells=2000)
        assert result['boundary_topology_complete']
        root = min(result['isolated'],key=lambda r:np.linalg.norm(
            r['stuv']-np.array([.25,.75,.375,.625])))
        point = BoundaryPoint(root['stuv'],root['point'],(axis,-1))
        assert identity.register_source_root(point,root['source_cut_certificate'])
        points.append(point)
    boxes = tuple(identity.source_certificates[id(point)]['exact_box'] for point in points)
    return identity,points,boxes


def _disjoint(first,second,axis):
    a,b = first[axis],second[axis]
    return a[1] < b[0] or b[1] < a[0]


def test_exact_pins_survive_identical_display_parameters():
    identity,points,boxes = _source_events(((0,.25),(2,.375)))
    np.testing.assert_array_equal(points[0].stuv,points[1].stuv)
    assert boxes[0][0] == (Fraction(1,4),)*2
    assert boxes[1][0] == (Fraction(9838263505978429,39353054023913712),)*2
    assert _disjoint(*boxes,0)
    refined = identity.separate_source_boxes(points,boxes)
    assert _disjoint(*refined,0)
    assert len(refined) == 2


def test_owner_refinement_preserves_nonbinary_fraction_separator():
    identity,points,boxes = _source_events(((3,.625),(2,.375)))
    separator = boxes[1][0][0]
    assert Fraction(float(separator)) != separator
    assert boxes[0][0][0] < separator < boxes[0][0][1]
    refined = identity.refine_source_box(points[0],boxes[0],boxes[1])
    assert refined[0][0] == separator
    assert identity.source_certificates[id(points[0])]['exact_box'] == boxes[0]


def test_sturm_refinement_separates_distinct_roots_below_one_ulp():
    identity,points,boxes = _source_events(((3,.625),(2,.375)))
    np.testing.assert_array_equal(points[0].stuv,points[1].stuv)
    assert not any(_disjoint(*boxes,axis) for axis in range(4))
    refined = identity.separate_source_boxes(points,boxes)
    assert _disjoint(*refined,0) and _disjoint(*refined,1)
    for point,old,new in zip(points,boxes,refined):
        certificate = identity.source_certificates[id(point)]
        assert certificate['exact_box'] == old
        assert all(a <= lo <= hi <= b for (a,b),(lo,hi) in zip(old,new))
        varying = certificate['varying_axis']
        polynomial = certificate['exact_polynomial']
        def value(x):
            return sum(coefficient*x**i for i,coefficient in enumerate(polynomial))
        assert value(new[varying][0])*value(new[varying][1]) <= 0
    np.testing.assert_array_equal(points[0].stuv,points[1].stuv)


def test_exact_root_at_separator_is_retained_not_declared_distinct():
    identity,points,boxes = _source_events(((0,.25),(0,.25)))
    refined = identity.separate_source_boxes(points,boxes)
    assert refined == boxes
    assert not any(_disjoint(*refined,axis) for axis in range(4))


def test_denied_exact_separation_retains_both_original_source_boxes():
    identity,points,boxes = _source_events(((3,.625),(2,.375)))
    identity.charge = lambda amount:False
    assert identity.separate_source_boxes(points,boxes) == boxes
    assert identity.exhausted
    assert all(identity.source_certificates[id(p)]['exact_box'] == b
               for p,b in zip(points,boxes))


def test_missing_source_certificate_keeps_the_supplied_enclosure():
    identity,points,boxes = _source_events(((3,.625),(2,.375)))
    identity.source_certificates.clear()
    assert identity.separate_source_boxes(points,boxes) == boxes


def test_work_exhaustion_during_sturm_refinement_keeps_both_root_identities():
    identity,points,boxes = _source_events(((3,.625),(2,.375)))
    spent = 0
    def charge(amount):
        nonlocal spent
        if spent+amount > 8:
            return False
        spent += amount
        return True
    identity.charge = charge
    refined = identity.separate_source_boxes(points,boxes)
    assert identity.exhausted and 0 < spent <= 8
    assert refined == boxes
    assert all(identity.source_certificates[id(p)]['exact_box'] == b
               for p,b in zip(points,boxes))
