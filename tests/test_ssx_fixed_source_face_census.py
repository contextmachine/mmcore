"""Exact affine faces can exhaust a cell without zero-width patch nets."""
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from examples.ssx.ssx5_analytic_audit import (
    audit_result, case_components, graph_pair, solve_case,
)
from mmcore.numeric.intersection._exact_univariate import _value
from mmcore.numeric.intersection.ssx._ssx_affine_constraints import AffineParameterConstraints
from mmcore.numeric.intersection.ssx._ssx_planar_cut import exact_source_planar_cut
from mmcore.numeric.intersection.ssx._ssx_root_identity import _pure_affine_coordinates
from mmcore.numeric.intersection.ssx._ssx_root_identity import BoundaryRootIdentity
from mmcore.numeric.intersection.ssx import _bez_ssx5 as ssx


def test_exact_fixed_face_census_excludes_a_rounded_corner_alias():
    surfaces = tuple(np.concatenate((net,np.ones(net.shape[:2]+(1,))),axis=-1)
                     for net in graph_pair(case_components('nested_circles')))
    owner = ((.375,.5),(.25,.5),(.5,np.nextafter(2/3,np.inf)),(.25,.375))
    constraints = AffineParameterConstraints(tuple(_pure_affine_coordinates(net)
                                                  for net in surfaces))
    necessary = constraints.contract(owner)
    assert necessary[0] == necessary[2] == (.5,.5)
    assert necessary[1][1] == .3125
    whole = exact_source_planar_cut(*surfaces,0,.5,((0.,1.),)*4)
    rounded_corner = whole['isolated'][1]
    np.testing.assert_array_equal(rounded_corner['stuv'],[.5,.3125,.5,.375])
    certificate = rounded_corner['source_cut_certificate']
    polynomial = [Fraction(int(a),int(b)) for a,b in certificate['polynomial']]
    assert _value(polynomial,Fraction(5,16)) != 0
    # The exact root is above the owner's t/v bounds, despite its rounded
    # representative landing on their shared corner. The original source
    # face solves the entire necessary zero set of this positive-width box.
    local = exact_source_planar_cut(*surfaces,0,.5,owner)
    assert local['boundary_topology_complete']
    assert not local['isolated']
    assert not local['budget_exhausted']


@pytest.mark.parametrize('variant', ['identity', 'swap'])
def test_fixed_face_reduction_preserves_complete_nested_closed_components(variant):
    result = solve_case('nested_circles',variant)
    report = audit_result('nested_circles',result)
    assert report['passed'],report


def test_nonroot_sturm_endpoint_excludes_a_borrowed_rounded_corner():
    surfaces = tuple(np.concatenate((net,np.ones(net.shape[:2]+(1,))),axis=-1)
                     for net in reversed(graph_pair(case_components('nested_circles'))))
    census = exact_source_planar_cut(*surfaces,1,.5,((0.,1.),)*4)
    root = next(root for root in census['isolated']
                if tuple(root['stuv']) == (.625,.5,.6875,.5))
    point = ssx.BoundaryPoint(root['stuv'],root['point'],(1,-1),
                              root_box=np.asarray(root['parameter_root_box']))
    matcher = BoundaryRootIdentity(*surfaces)
    assert matcher.register_source_root(point,root['source_cut_certificate'])
    owner = ((.5625,.625),(.375,.5),(.59375,.6875),(.3125,.5))
    cell = SimpleNamespace(box=owner,root_matcher=matcher)
    certificate = matcher.source_certificates[id(point)]
    axis = certificate['varying_axis']
    low, high = certificate['exact_interval']
    assert low == Fraction(owner[axis][1]) < high
    assert _value(certificate['exact_polynomial'],low) != 0
    assert ssx._source_boundary_outside(cell,point)
    # Its other closed child owns the actual root. Denied/absent source
    # evidence is never a reason to discard the borrowed representative.
    cell.box = ((.625,.75),(.375,.5),(.6875,.875),(.3125,.5))
    assert not ssx._source_boundary_outside(cell,point)
    cell.box = owner
    matcher.charge = lambda amount: False
    assert not ssx._source_boundary_outside(cell,point)
    matcher.source_certificates.clear()
    assert not ssx._source_boundary_outside(cell,point)


def test_an_actual_endpoint_root_is_retained_in_both_closed_owners():
    first = np.array([[[s,t,2*t-1,1.] for t in (0.,1.)] for s in (0.,1.)])
    second = np.array([[[s,t,0.,1.] for t in (0.,1.)] for s in (0.,1.)])
    matcher = BoundaryRootIdentity(first,second)
    root = exact_source_planar_cut(first,second,0,.5,((0.,1.),)*4)['isolated'][0]
    point = ssx.BoundaryPoint(root['stuv'],root['point'],(0,-1),
                              root_box=np.asarray(root['parameter_root_box']))
    assert matcher.register_source_root(point,root['source_cut_certificate'])
    # The varying root t=v=1/2 is an actual source zero on the shared face.
    for owner in (((0.,1.),(0.,.5),(0.,1.),(0.,.5)),
                  ((0.,1.),(.5,1.),(0.,1.),(.5,1.))):
        assert not ssx._source_boundary_outside(SimpleNamespace(box=owner,root_matcher=matcher),point)


def test_open_sturm_endpoints_do_not_make_false_linked_corner_faces():
    surfaces = tuple(np.concatenate((net,np.ones(net.shape[:2]+(1,))),axis=-1)
                     for net in graph_pair(case_components('four_lines')))
    matcher = BoundaryRootIdentity(*surfaces)
    matcher.affine_constraints = AffineParameterConstraints(matcher.affine)
    census = exact_source_planar_cut(*surfaces,1,0.,((0.,1.),)*4)
    root = next(root for root in census['isolated'] if root['stuv'][0] == .875)
    point = ssx.BoundaryPoint(root['stuv'],root['point'],(1,-1),
                              root_box=np.asarray(root['parameter_root_box']))
    assert matcher.register_source_root(point,root['source_cut_certificate'])
    owner = ((.875,1.),(0.,1.),(.75,1.),(0.,1.))
    cell = SimpleNamespace(box=owner,root_matcher=matcher)
    exact_box = matcher.source_certificates[id(point)]['exact_box']
    assert exact_box[0][0] == Fraction(7,8)
    assert exact_box[2][0] == Fraction(3,4)
    assert ssx._source_boundary_possible_faces(cell,point,exact_box) == [(1,1.)]
    # Without paid scalar/affine evidence the extra possibly active faces
    # remain constraints; they are never guessed away from a tiny residual.
    matcher.charge = lambda amount: False
    faces = ssx._source_boundary_possible_faces(cell,point,exact_box)
    assert (0,1.) in faces and (2,1.) in faces


def test_generic_source_inclusion_can_exclude_an_unrelated_closed_child():
    first = np.array([[[s,t,2*t-1,1.] for t in (0.,1.)] for s in (0.,1.)])
    second = np.array([[[s,t,0.,1.] for t in (0.,1.)] for s in (0.,1.)])
    matcher = BoundaryRootIdentity(first,second)
    point = ssx.BoundaryPoint(np.array([.25,.5,.25,.5]),np.array([.25,.5,0.]),(0,-1))
    point.root_box = matcher.enclose(point,np.full(4,.001))
    assert point.root_box is not None
    assert not matcher.source_certificates  # Generic source inclusion.
    cell = SimpleNamespace(box=((.5,1.),(0.,1.),(0.,1.),(0.,1.)),root_matcher=matcher)
    assert not ssx._source_boundary_outside(cell,point)  # No source flag.
    point._source_root_box = True
    assert ssx._source_boundary_outside(cell,point)
    cell.box = ((0.,.5),(0.,1.),(0.,1.),(0.,1.))
    assert not ssx._source_boundary_outside(cell,point)
