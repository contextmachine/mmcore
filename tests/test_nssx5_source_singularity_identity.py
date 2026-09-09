"""Geometric proximity does not identify isolated roots or singular strata."""
from fractions import Fraction

import numpy as np

from mmcore.numeric.intersection.ssx import _nssx5 as ssx
from mmcore.numeric.intersection.ssx._bez_ssx5 import SSXSingularity


def _context(closed=False):
    return ssx._DomainCtx(np.zeros(4),np.ones(4),np.ones(4),np.full(4,.001),(closed,False,False,False))


def test_distinct_exact_nearby_tangent_points_remain_distinct():
    # Plane vs z=((s-c)^2+(t-.5)^2)*((s-d)^2+(t-.5)^2)
    # has exactly these two isolated tangent points, however close c,d are.
    c,d = Fraction(1,2),Fraction(1,2)+Fraction(1,2**20)
    singularities = []
    for x in (c,d):
        assert ((x-c)**2)*((x-d)**2) == 0
        q = np.array([float(x),.5,float(x),.5])
        singularities.append(SSXSingularity('tangent_point',q,np.array([float(x),.5,0.])))
    result = ssx._assemble_singularities(singularities,[],_context(),.001,ssx._make_aggregate({},1))
    assert len(result) == 2


def test_identical_cusp_samples_do_not_own_a_whole_source_stratum():
    q = np.array([.5,.5,.5,.5])
    first = SSXSingularity('cusp_curve',q,np.zeros(3),samples=np.array([q]))
    second = SSXSingularity('cusp_curve',q.copy(),np.zeros(3),samples=np.array([q]))
    result = ssx._assemble_singularities([first,second,first],[],_context(),.001,ssx._make_aggregate({},1))
    assert len(result) == 2 and result[0] is first and result[1] is second


def test_cusps_on_different_source_charts_keep_both_surface_owners():
    q = np.full(4,.5)
    values = [SSXSingularity('cusp',q.copy(),np.zeros(3),surface=owner) for owner in (1,2)]
    result = ssx._assemble_singularities(values,[],_context(),.001,ssx._make_aggregate({},1))
    assert len(result) == 2 and {value.surface for value in result} == {1,2}


def test_sub_ulp_distance_from_seam_is_not_exact_endpoint_identity():
    ctx = _context(True)
    first = np.array([1.,.5,.5,.5])
    second = np.array([2.**-60,.5,.5,.5])
    assert abs(first[0]-second[0]) == 1.  # The former distance test cancelled.
    assert not ssx._dup_stuv(first,second,np.zeros(3),np.zeros(3),ctx,.001)
    assert not ssx._mate_matches(first,second,ctx)
    second[0] = 0.
    assert ssx._dup_stuv(first,second,np.zeros(3),np.zeros(3),ctx,.001)


def test_sub_ulp_seam_gap_does_not_join_two_unknown_arc_endpoints():
    first = np.array([[.9,.5,.5,.5],[1.,.5,.5,.5]])
    second = np.array([[2.**-60,.5,.5,.5],[.1,.5,.5,.5]])
    fragments = [ssx._Frag(q,np.column_stack((np.array([-.1,0.]) if i == 0 else np.array([0.,.1]),
                                             np.zeros((2,2)))),'transversal',False)
                 for i,q in enumerate((first,second))]
    result = ssx._assemble_branches(fragments,_context(True),.001,ssx._make_aggregate({},1))
    assert len(result) == 2 and not any(branch.closed for branch in result)
