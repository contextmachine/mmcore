"""An approximation chord cannot own an independent isolated source root."""
from fractions import Fraction
from math import comb

import numpy as np

from mmcore.numeric._work_budget import SoftWorkBudget
from mmcore.numeric.intersection._root_box_certificate import exact_bernstein_value
from mmcore.numeric.intersection.ssx._bez_ssx5 import (
    BoundaryPoint, _registered_point, _remove_source_incident_points,
)
from mmcore.numeric.intersection.ssx._nssx5 import _assemble_points, _DomainCtx, _make_aggregate
from mmcore.numeric.intersection.ssx._ssx_polyline import point_matches_polyline
from mmcore.numeric.intersection.ssx._ssx_substrate import SSXBranch, SSXPoint


def _parabola_and_isolated_point():
    # z=(t-(s-.5)^2)*((s-.5)^2+(t-a^2)^2). Its exact real zero
    # set is the parabola plus the distinct isolated point (.5,a^2).
    a = Fraction(1,32)
    square = {(0,0):Fraction(1,4),(1,0):Fraction(-1),(2,0):Fraction(1)}
    first = {key:-value for key,value in square.items()}
    first[0,1] = Fraction(1)
    second = dict(square)
    second[0,0] += a**4
    second[0,1],second[0,2] = -2*a*a,Fraction(1)
    power = {}
    for (i,j),left in first.items():
        for (k,l),right in second.items():
            key = i+k,j+l
            power[key] = power.get(key,Fraction(0))+left*right
    # Uniform homogeneous scale36 makes all Bernstein coefficients dyadic,
    # including the degree-three coordinate chart. No source rounding.
    graph = np.zeros((5,4,4))
    for i in range(5):
        for j in range(4):
            z = sum(value*Fraction(comb(i,k),comb(4,k))*Fraction(comb(j,l),comb(3,l))
                    for (k,l),value in power.items() if k <= i and l <= j)
            exact = (9*i,12*j,36*z,36)
            graph[i,j] = [float(value) for value in exact]
            assert all(Fraction.from_float(float(value)) == target
                       for value,target in zip(graph[i,j],exact))
    plane = np.array([[[s,t,0.,1.] for t in (0.,1.)] for s in (0.,1.)])
    u = np.array([float(Fraction(1,2)-a),float(Fraction(1,2)+a)])
    v = float(a*a)
    stuv = np.column_stack((u,np.full(2,v),u,np.full(2,v)))
    xyz = np.column_stack((u,np.full(2,v),np.zeros(2)))
    branch = SSXBranch((stuv,xyz))
    point = SSXPoint(np.array([.5,v,.5,v]),np.array([.5,v,0.]))
    for p in [point.stuv,*stuv]:
        first_value = exact_bernstein_value(graph,p[:2])
        second_value = exact_bernstein_value(plane,p[2:])
        assert all(first_value[k]*second_value[3] == second_value[k]*first_value[3] for k in range(3))
    # The parabola-to-chord error is exactly a^2, within the advertised atol.
    assert a*a < Fraction(1,1000)
    assert point_matches_polyline(point.stuv,point.xyz,stuv,xyz,np.zeros(4),1e-3)
    return point,branch


def test_bezier_cleanup_keeps_actual_isolated_root_on_an_approximation_chord():
    point,branch = _parabola_and_isolated_point()
    budget = SoftWorkBudget(100,100,max_postprocess_work=0)
    kept = _remove_source_incident_points([point],[branch],np.ones(4)*1e-3,1e-3,None,budget)
    assert len(kept) == 1 and kept[0] is point
    assert budget.result_fields()['complete']


def test_nurbs_cleanup_keeps_actual_isolated_root_on_an_approximation_chord():
    point,branch = _parabola_and_isolated_point()
    context = _DomainCtx(np.zeros(4),np.ones(4),np.ones(4),np.full(4,.001),(False,)*4)
    aggregate = _make_aggregate({},1)
    kept = _assemble_points([point],[branch],context,1e-3,aggregate)
    assert len(kept) == 1 and kept[0] is point


def test_source_registration_still_proves_point_incidence():
    _,branch = _parabola_and_isolated_point()
    # This separate control supplies the actual shared registration, not
    # just an equal float tuple or membership in an approximate chord.
    token = BoundaryPoint(branch.curve[0][0],branch.curve[1][0],(0,-1))
    point = _registered_point(token)
    branch._registered_root_ids = frozenset((id(token),))
    branch._registered_root_points = {id(token):token}
    assert _remove_source_incident_points([point],[branch],np.ones(4)*1e-3,1e-3,None) == []
