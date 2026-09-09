"""Equivalent faces transport owned roots, not fresh proposal associations."""
import numpy as np
import pytest

from mmcore.numeric.intersection.ssx._bez_ssx5 import BoundaryPoint
from mmcore.numeric.intersection.ssx._ssx_affine_constraints import AffineParameterConstraints
from mmcore.numeric.intersection.ssx._ssx_root_identity import BoundaryRootIdentity


def matcher_and_points(second_interval):
    first = np.array([[[s, t, 0., 1.] for t in (0., 1.)] for s in (0., 1.)])
    # The exact face s=u=1/2 contains t=v=1/4 and t=v=3/4.
    height = (.1875, -.3125, .1875)
    second = np.array([[[u, v, height[j], 1.] for j, v in enumerate((0., .5, 1.))]
                       for u in (0., 1.)])
    matcher = BoundaryRootIdentity(first, second)
    matcher.affine_constraints = AffineParameterConstraints(matcher.affine)
    # Source ownership is independent of the displayed proposal. A fresh
    # Newton solve at either of these proposals can converge to root 1/4.
    a = BoundaryPoint(np.array([.5, .25, .5, .25]), np.array([.5, .25, 0.]), (0, -1))
    b = BoundaryPoint(a.stuv.copy(), np.array([.5, .75, 0.]), (2, -1))
    for point, interval in ((a, (.24, .26)), (b, second_interval)):
        point.root_box = np.array([(.5, .5), interval, (.5, .5), interval])
        point._source_root_box = True
    return matcher, a, b


def test_new_proposal_enclosure_cannot_replace_another_owned_root(monkeypatch):
    # Both boxes contain exactly one source root, but overlap in a root-free
    # interval. Independent fresh proposal solves can select the SAME root
    # despite these two different immutable ownership associations.
    matcher, first, second = matcher_and_points((.255, .76))
    original_boxes = [point.root_box.copy() for point in (first, second)]
    monkeypatch.setattr(matcher, 'enclose', lambda *a, **k: first.root_box.copy())
    assert not matcher(first, second, np.full(4, .1), 1.)
    for point, original in zip((first, second), original_boxes):
        np.testing.assert_array_equal(point.root_box, original)


def test_same_owned_root_on_equivalent_face_uses_existing_existence(monkeypatch):
    matcher, first, second = matcher_and_points((.245, .3))
    second.xyz = first.xyz.copy()
    monkeypatch.setattr(matcher, 'enclose',
                        lambda *a, **k: pytest.fail('owned root must not be reassociated'))
    assert matcher(first, second, np.full(4, .1), .01)


def test_unproved_original_face_membership_cannot_be_replaced_by_a_fresh_root(monkeypatch):
    matcher, first, second = matcher_and_points((.245, .3))
    second.root_box[2] = (.4, .6)
    second.xyz = first.xyz.copy()
    monkeypatch.setattr(matcher, 'enclose',
                        lambda *a, **k: pytest.fail('missing owned-face proof must not be guessed'))
    assert not matcher(first, second, np.full(4, .1), .01)


def test_denied_union_proof_preserves_distinct_registrations():
    matcher, first, second = matcher_and_points((.245, .3))
    matcher.charge = lambda amount: False
    assert not matcher(first, second, np.full(4, .1), 1.)
    assert matcher.exhausted
