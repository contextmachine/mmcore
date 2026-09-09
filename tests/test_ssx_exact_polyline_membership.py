"""Deleting output requires exact lifted membership at zero parameter tolerance."""
import numpy as np

from mmcore.numeric.intersection.ssx._ssx_polyline import (
    point_matches_polyline, polyline_contained,
)


def _diagonal():
    return np.array([[1.,1.,1.,1.],[0.,0.,0.,0.]]), np.array([[1.,0.,0.],[0.,0.,0.]])


def test_cancellation_cannot_move_a_distinct_parameter_point_onto_a_segment():
    s,x = _diagonal()
    e = 2.**-60
    assert not point_matches_polyline([e,2*e,e,e],np.zeros(3),s,x,np.zeros(4),1e-12)


def test_close_point_on_the_exact_lifted_segment_is_retained_as_incidence():
    s,x = _diagonal()
    e = 2.**-60
    assert point_matches_polyline([e]*4,[e,0.,0.],s,x,np.zeros(4),0.)


def test_geometric_comparison_uses_the_same_exact_parameter_fraction():
    s,x = _diagonal()
    e = 2.**-60
    assert not point_matches_polyline([e]*4,np.zeros(3),s,x,np.zeros(4),e/2)


def test_disabled_seam_segments_keep_only_their_explicit_vertices():
    s,x = _diagonal()
    assert not point_matches_polyline([.5]*4,[.5,0.,0.],s,x,np.zeros(4),0.,segment_mask=[False])
    assert point_matches_polyline(s[0],x[0],s,x,np.zeros(4),0.,segment_mask=[False])


def test_constant_parameter_segment_uses_exact_world_projection():
    s,x = _diagonal()
    s[:] = .5
    assert point_matches_polyline([.5]*4,[.25,0.,0.],s,x,np.zeros(4),0.)
    assert not point_matches_polyline([.5]*4,[.25,1e-15,0.],s,x,np.zeros(4),0.)


def test_cancellation_cannot_cover_a_distinct_lifted_path():
    s,x = _diagonal()
    e = 2.**-60
    source = np.array([[e,2*e,e,e],[2*e,3*e,2*e,2*e]])
    assert not polyline_contained(source,np.zeros((2,3)),s,x,np.zeros(4),1e-12)


def test_exact_collinear_path_coverage_survives_varied_sampling_and_reversal():
    parameters = np.array([0.,2.**-60,.25,.75,1.])
    s = np.repeat(parameters[:,None],4,axis=1)
    x = np.column_stack((parameters,np.zeros((len(parameters),2))))
    ks,kx = _diagonal()
    assert polyline_contained(s,x,ks,kx,np.zeros(4),0.)
    assert polyline_contained(ks,kx,s,x,np.zeros(4),0.)
    assert polyline_contained(s,x,s[::-1],x[::-1],np.zeros(4),0.)


def test_exact_parameter_coverage_keeps_a_subnormal_gap():
    # The disabled connector is a real unrepresented parameter interval.
    e = np.nextafter(0.,1.)
    source = np.array([[0.]*4,[2*e]*4])
    keeper = np.array([[0.]*4,[e]*4,[2*e]*4])
    assert not polyline_contained(source,np.zeros((2,3)),keeper,np.zeros((3,3)),
                                   np.zeros(4),0.,keeper_mask=[True,False])


def test_identical_path_fast_proof_preserves_geometric_and_budget_limits():
    s,x = _diagonal()
    shifted = x.copy()
    shifted[:,1] += .125
    assert polyline_contained(s,x,s,shifted,np.zeros(4),.125)
    assert not polyline_contained(s,x,s,shifted,np.zeros(4),.0625)
    assert polyline_contained(s,x,s,x,np.zeros(4),0.,charge=lambda _:False) is None
