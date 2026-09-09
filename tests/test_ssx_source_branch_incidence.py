"""Two true zero arcs can have identical valid approximation chords."""
from fractions import Fraction
from math import comb

import numpy as np

from mmcore.numeric.intersection._root_box_certificate import exact_bernstein_value
from mmcore.numeric.intersection.ssx import _bez_ssx5 as bez
from mmcore.numeric.intersection.ssx import _nssx5 as nurbs


def _lens():
    # F=(t-x^2)*(t+x^2-2a^2), x=s-.5. The target plane clips
    # x to [-a,a], producing the two arcs of one exact closed lens.
    a = Fraction(1,32)
    x2 = [Fraction(1,4),Fraction(-1),Fraction(1)]
    x4 = [Fraction(1,16),Fraction(-1,2),Fraction(3,2),Fraction(-2),Fraction(1)]
    power = {(i,0):-value for i,value in enumerate(x4)}
    for i,value in enumerate(x2):
        power[i,0] += 2*a*a*value
    power[0,1],power[0,2] = -2*a*a,Fraction(1)
    graph = np.zeros((5,3,4))
    for i in range(5):
        for j in range(3):
            z = sum(value*Fraction(comb(i,k),comb(4,k))*Fraction(comb(j,l),comb(2,l))
                    for (k,l),value in power.items() if k <= i and l <= j)
            exact = (3*i,6*j,12*z,12)
            graph[i,j] = list(map(float,exact))
            assert all(Fraction.from_float(float(x)) == value for x,value in zip(graph[i,j],exact))
    plane = np.array([[[float(Fraction(1,2)-a+2*a*u),float(2*a*a*v),0.,1.]
                       for v in (0,1)] for u in (0,1)])
    q = np.array([[float(Fraction(1,2)-a),float(a*a),0.,.5],
                  [float(Fraction(1,2)+a),float(a*a),1.,.5]])
    xyz = np.column_stack((q[:,:2],np.zeros(2)))
    for parameters in q:
        first = exact_bernstein_value(graph,parameters[:2])
        second = exact_bernstein_value(plane,parameters[2:])
        assert all(first[k]*second[3] == second[k]*first[3] for k in range(3))
    # Both genuine arcs lie at distance <=a² from this same chord.
    assert a*a < Fraction(1,1000)
    return q,xyz


def test_bezier_lens_arcs_with_shared_endpoints_survive_fragment_dedup():
    q,xyz = _lens()
    left,right = [bez.BoundaryPoint(s,x,(2,side)) for s,x,side in zip(q,xyz,(0,1))]
    first = bez._Fragment(left,right,q,xyz)
    second = bez._Fragment(right,left,q[::-1].copy(),xyz[::-1].copy())
    assert len(bez._drop_duplicate_fragments([first,second],.001)) == 2
    branches = bez._assemble_fragments([first,second],unify_tol=np.full(4,.001))
    assert len(branches) == 1 and branches[0].closed
    assert len(branches[0].curve[0]) == 3


def test_nurbs_lens_arcs_survive_and_form_their_registered_cycle():
    q,xyz = _lens()
    fragments = [nurbs._Frag(q,xyz,'transversal',False),
                 nurbs._Frag(q[::-1].copy(),xyz[::-1].copy(),'transversal',False)]
    context = nurbs._DomainCtx(np.zeros(4),np.ones(4),np.ones(4),np.full(4,.001),(False,)*4)
    assert len(nurbs._containment_dedup(fragments,.001,nurbs._make_aggregate({},1),context)) == 2
    branches = nurbs._assemble_branches(fragments,context,.001,nurbs._make_aggregate({},1))
    assert len(branches) == 1 and branches[0].closed
    assert len(branches[0].curve[0]) == 3


def test_repeated_same_fragment_object_has_actual_shared_provenance():
    q,xyz = _lens()
    fragment = bez._Fragment(None,None,q,xyz)
    assert bez._drop_duplicate_fragments([fragment,fragment],.001) == [fragment]
    mapped = nurbs._Frag(q,xyz,'transversal',False)
    assert nurbs._containment_dedup([mapped,mapped],.001,nurbs._make_aggregate({},1)) == [mapped]


def test_boundary_arcs_on_a_folded_chart_are_not_identified_by_their_chords():
    # S1(s,t)=(t,s,0), S2(u,v)=(u,(u-.5)^2+(v-.5)^2-a²,0).
    # The boundary s=0 has a circular parameter preimage on S2. Its upper
    # and lower arcs have the same endpoints and this same approximate chord.
    a = 1/32
    q = np.array([[0.,.5-a,.5-a,.5],[0.,.5+a,.5+a,.5]])
    first = np.array([[[t,s,0.] for t in (0.,1.)] for s in (0.,1.)])
    overlaps = [bez.BoundaryOverlap(q[0],q[1],(0,0)),
                bez.BoundaryOverlap(q[1].copy(),q[0].copy(),(0,0))]
    for parameters in [q[0],q[1],np.array([0.,.5,.5,.5-a]),np.array([0.,.5,.5,.5+a])]:
        s,t,u,v = map(Fraction.from_float,parameters)
        assert s == (u-Fraction(1,2))**2+(v-Fraction(1,2))**2-Fraction(1,32)**2
        assert t == u
    branches = bez._overlaps_to_branches(overlaps,first,.001,False)
    assert len(branches) == 2


def test_valid_parabola_chord_is_not_deleted_by_midpoint_residual_estimate():
    # h=512*g*(2a²-g), g=v-(u-.5)². The lower zero arc g=0
    # lies <=a² from its endpoint chord. Both source evaluations along
    # the published lifted chord are also within 512*a^4=1/2048 of it.
    # At the midpoint grad(h)=0 while h=1/2048, so dividing this
    # residual by a clamped normal angle invents a large distance.
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
    left,right = [bez.BoundaryPoint(s,x,(0,-1)) for s,x in zip(q,xyz)]
    branch = bez._assemble_fragments(
        [bez._Fragment(left,right,q,xyz)], S1_full=plane,S2_full=graph,
        atol_full=.001,rational_full=True)
    assert len(branch) == 1
    np.testing.assert_array_equal(branch[0].curve[0],q)


def test_failed_endpoint_source_identity_is_not_overridden_by_equal_samples():
    # Two independent source enclosures can have identical rounded display
    # tuples. A declined root identity must survive every assembly phase.
    first = np.array([[.2,.5,.2,.5],[.5,.5,.5,.5]])
    second = np.array([[.5,.5,.5,.5],[.8,.5,.8,.5]])
    fragments = []
    for index,q in enumerate((first,second)):
        xyz = np.column_stack((q[:,0],np.zeros((2,2))))
        endpoints = [bez.BoundaryPoint(p,x,(index,0)) for p,x in zip(q,xyz)]
        fragments.append(bez._Fragment(*endpoints,q,xyz))
    branches = bez._assemble_fragments(
        fragments,unify_tol=np.full(4,.001),root_matcher=lambda *args:False)
    assert len(branches) == 2
