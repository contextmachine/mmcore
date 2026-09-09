from types import SimpleNamespace

import numpy as np
import pytest

from mmcore.numeric.intersection.ssx._bez_ssx5 import _balanced_root_avoiding_cut


def _event(lo, hi):
    enclosure = np.zeros((4, 2))
    enclosure[0] = lo, hi
    return SimpleNamespace(root_box=enclosure, stuv=enclosure.mean(axis=1))


def test_balanced_cut_avoids_a_known_continuous_face_coordinate():
    events = [_event(.5, .5)]
    cut = _balanced_root_avoiding_cut(events, 0, ((0., 1.),)*4)
    assert .25 <= cut <= .75
    assert cut != .5


@pytest.mark.parametrize('bands', [[], [(1e-20, 2e-20)], [(.49, .51)],
                                     [(.25, .5), (.6, .75)], [(0., 1.)]])
def test_root_guidance_cannot_prevent_whole_owner_contraction(bands):
    cut = _balanced_root_avoiding_cut([_event(*pair) for pair in bands], 0, ((0., 1.),)*4)
    assert .25 <= cut <= .75
    assert max(cut, 1.-cut) <= .75
    if bands == [(1e-20, 2e-20)]:
        assert cut == .5


def test_adjacent_endpoints_are_left_for_explicit_representation_refusal():
    lo, hi = .5, np.nextafter(.5, 1.)
    cut = _balanced_root_avoiding_cut([], 0, ((lo, hi),)+((0., 1.),)*3)
    assert not lo < cut < hi
