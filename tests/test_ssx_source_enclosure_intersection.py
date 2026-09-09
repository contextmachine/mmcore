"""Source certificates retain independent exact enclosure contractions."""
from fractions import Fraction
from types import SimpleNamespace

import numpy as np

from mmcore.numeric.intersection.ssx import _bez_ssx5 as ssx
from mmcore.numeric.intersection.ssx._ssx_affine_constraints import AffineParameterConstraints
from mmcore.numeric.intersection.ssx._ssx_planar_cut import exact_source_planar_cut
from mmcore.numeric.intersection.ssx._ssx_root_identity import BoundaryRootIdentity


def _point(first, second):
    matcher = BoundaryRootIdentity(first, second)
    matcher.affine_constraints = AffineParameterConstraints(matcher.affine)
    root = exact_source_planar_cut(first, second, 1, 0., ((0., 1.),)*4)['isolated'][0]
    point = ssx.BoundaryPoint(np.asarray(root['stuv']), np.asarray(root['point']), (1, -1),
                              root_box=np.asarray(root['parameter_root_box']))
    point._source_root_box = True
    assert matcher.register_source_root(point, root['source_cut_certificate'])
    return matcher, point


def test_source_sturm_box_keeps_independently_proved_affine_contraction():
    weight = np.sqrt(.5)
    first = np.array([[[x*w, t*w, z*w, w] for t in (0., 1.)]
                      for x, z, w in ((1., 0., 1.), (1., 1., weight), (0., 1., 1.))])
    second = np.array([[[2*u-.5, 2*v-.5, .5, 1.] for v in (0., 1.)] for u in (0., 1.)])
    matcher, point = _point(first, second)
    certificate = matcher.source_certificates[id(point)]
    original = certificate['exact_box']
    assert original[3][0] < Fraction(1, 4) < original[3][1]
    point.root_box = np.asarray(matcher.affine_constraints.contract(point.root_box))
    np.testing.assert_array_equal(point.root_box[3], [.25, .25])
    cell = SimpleNamespace(box=((0., 1.),)*3+((.25, .75),), root_matcher=matcher)
    exact = ssx._source_boundary_intervals(cell, point, point.root_box)
    assert exact[3] == (Fraction(1, 4), Fraction(1, 4))
    assert certificate['exact_box'] == original


def test_intersection_never_clips_the_source_enclosure_to_its_owner():
    first = np.array([[[s, t, 3*s-1, 1.] for t in (0., 1.)] for s in (0., 1.)])
    second = np.array([[[s, t, 0., 1.] for t in (0., 1.)] for s in (0., 1.)])
    matcher, point = _point(first, second)
    cell = SimpleNamespace(box=((0., float(Fraction(1, 3))),)+((0., 1.),)*3,
                           root_matcher=matcher)
    exact = ssx._source_boundary_intervals(cell, point, point.root_box)
    assert exact[0] == (Fraction(1, 3), Fraction(1, 3))
    assert exact[0][0] > cell.box[0][1]


def test_unproved_or_unpaid_enclosure_cannot_tighten_a_source_certificate():
    first = np.array([[[s, t, 3*s-1, 1.] for t in (0., 1.)] for s in (0., 1.)])
    second = np.array([[[s, t, 0., 1.] for t in (0., 1.)] for s in (0., 1.)])
    matcher, point = _point(first, second)
    cell = SimpleNamespace(box=((0., 1.),)*4, root_matcher=matcher)
    original = matcher.source_certificates[id(point)]['exact_box']
    point._source_root_box = False
    assert ssx._source_boundary_intervals(cell, point, np.zeros((4, 2))) == original
    point._source_root_box = True
    matcher.charge = lambda amount: False
    assert ssx._source_boundary_intervals(cell, point, point.root_box) == original


def _nearby_circle_source_root():
    from examples.ssx.ssx5_analytic_audit import graph_pair, case_components
    first, second = [np.concatenate((net, np.ones(net.shape[:2]+(1,))), axis=2)
                     for net in reversed(graph_pair(case_components('nearby_circles')))]
    matcher = BoundaryRootIdentity(first, second)
    matcher.affine_constraints = AffineParameterConstraints(matcher.affine)
    root = exact_source_planar_cut(first, second, 1, .5, ((0., 1.),)*4)['isolated'][2]
    point = ssx.BoundaryPoint(np.asarray(root['stuv']), np.asarray(root['point']), (1, -1),
                              root_box=np.asarray(root['parameter_root_box']))
    point._source_root_box = True
    assert matcher.register_source_root(point, root['source_cut_certificate'])
    return matcher, point


def test_sturm_owner_face_refines_linked_axis_without_changing_source_identity():
    matcher, point = _nearby_circle_source_root()
    certificate = matcher.source_certificates[id(point)]
    original = certificate['exact_box']
    face = float(Fraction(31, 48))
    assert original[0][0] < face < original[0][1]
    owner = ((.64, face), (.48, .5), (.71, .72), (.48, .5))
    refined = matcher.refine_source_box(point, original, owner)
    assert refined[0][1] == Fraction(face)
    assert refined[2][1] == Fraction(3, 2)*Fraction(face)-Fraction(1, 4)
    assert refined[2][1] < original[2][1]
    # The root lies strictly to the left of this rounded face. Exact
    # polynomial signs at the isolated interval endpoints prove it remains
    # bracketed, independently of the refinement's Sturm implementation.
    polynomial = certificate['exact_polynomial']
    def value(x):
        return sum(c*x**i for i, c in enumerate(polynomial))
    assert value(refined[2][0])*value(refined[2][1]) < 0
    assert certificate['exact_box'] == original


def test_sturm_owner_refinement_retains_root_outside_opposite_owner():
    matcher, point = _nearby_circle_source_root()
    original = matcher.source_certificates[id(point)]['exact_box']
    face = float(Fraction(31, 48))
    owner = ((face, .66), (.48, .5), (.71, .72), (.48, .5))
    refined = matcher.refine_source_box(point, original, owner)
    assert refined[0][0] < face
    assert refined[0][1] == Fraction(face)
    assert refined[0][0] == original[0][0]


def test_unpaid_sturm_owner_refinement_keeps_original_enclosure():
    matcher, point = _nearby_circle_source_root()
    original = matcher.source_certificates[id(point)]['exact_box']
    matcher.charge = lambda amount: False
    owner = ((.64, float(Fraction(31, 48))),)+((0., 1.),)*3
    assert matcher.refine_source_box(point, original, owner) == original
