"""Existing trace coverage must preserve paired geometry and search budgets."""
from types import SimpleNamespace

import numpy as np
import pytest

from mmcore.numeric.intersection.ssx._ssx_bernstein_clip import (
    residual_coordinate_scale, restrict_source_pair,
)
from mmcore.numeric.intersection.ssx._ssx_trace_coverage import TangentialTraceCoverage


def _fragment(low=0., high=1.):
    parameters = np.array([[low, .5, low, .5], [high, .5, high, .5]])
    xyz = np.array([[low, .5, 0.], [high, .5, 0.]])
    return SimpleNamespace(stuv_path=parameters, xyz_path=xyz, tangential=True)


def _sources():
    graph = np.array([[[s, t, z, 1.] for t, z in zip(
        (0., .5, 1.), (.25, -.25, .25))] for s in (0., 1.)])
    plane = np.array([[[s, t, 0., 1.] for t in (0., 1.)] for s in (0., 1.)])
    return graph, plane


def _coverage(fragments=None, charge=lambda amount: True):
    coverage = TangentialTraceCoverage(1e-3, np.full(4, 1e-3),
                                       residual_coordinate_scale(*_sources()), charge)
    coverage.update([_fragment()] if fragments is None else fragments)
    return coverage


def test_clipped_both_source_graphs_cover_existing_tangent_line():
    first, second = _sources()
    box = ((.25, .75), (.5, .75), (.25, .75), (.5, .75))
    restricted = restrict_source_pair(first, second, box)
    # Raw source hulls extend far outside the line tube. The residual
    # forces both transverse parameters close to the existing line.
    assert np.max(abs(restricted[1][..., 1] - .5)) > 1e-3
    assert _coverage().covers_cell(*restricted, box)


def test_same_xyz_on_a_different_partner_chart_is_not_covered():
    first, second = _sources()
    local_box = ((.25, .75), (.5, .75), (.25, .75), (.5, .75))
    restricted = restrict_source_pair(first, second, local_box)
    other_chart = ((.25, .75), (.5, .75), (.65, .95), (.5, .75))
    assert not _coverage().covers_cell(*restricted, other_chart)


def test_separate_trace_fragments_do_not_fill_their_gap():
    first, second = _sources()
    box = ((.45, .55), (.5, .6), (.45, .55), (.5, .6))
    restricted = restrict_source_pair(first, second, box)
    assert not _coverage([_fragment(0., .4), _fragment(.6, 1.)]).covers_cell(*restricted, box)


def test_whole_source_bulge_cannot_be_hidden_by_covered_endpoints():
    surface = np.array([[[s, t, z, 1.] for t in (.5, .50001)]
                        for s, z in zip((0., .5, 1.), (0., .1, 0.))])
    box = ((0., 1.), (.5, .50001), (0., 1.), (.5, .50001))
    assert not _coverage().covers_cell(surface, surface, box)


def test_nonuniform_weights_leave_the_cell_to_ordinary_search():
    first, second = _sources()
    first[0] *= 2.
    assert not _coverage().covers_cell(first, second, ((0., 1.),) * 4)


@pytest.mark.parametrize('parameters, xyz', [
    ([[.2, .5, .2, .5]], [[.8, .5, 0.]]),
    ([[.2, .5, .8, .5]], [[.2, .5, 0.]]),
    ([], []),
])
def test_known_points_require_one_paired_segment_location(parameters, xyz):
    assert not _coverage().covers_points(parameters, xyz)


def test_unknown_seed_keeps_the_tangency_discovery_path_enabled():
    coverage = _coverage()
    assert coverage.covers_points([[.2, .5, .2, .5]], [[.2, .5, 0.]])
    assert not coverage.covers_points(
        [[.2, .5, .2, .5], [.8, .2, .8, .2]],
        [[.2, .5, 0.], [.8, .2, 0.]])


def test_denied_optional_coverage_does_not_retire_a_cell(monkeypatch):
    from mmcore.numeric.intersection.ssx import _ssx_bernstein_clip as clip
    enabled = [True]
    coverage = _coverage(charge=lambda amount: enabled[0])
    enabled[0] = False
    first, second = _sources()
    def forbidden(*args):
        raise AssertionError('residual setup after work denial')
    monkeypatch.setattr(clip, 'psi_vector_net', forbidden)
    assert not coverage.covers_cell(first, second, ((0., 1.),) * 4)


@pytest.mark.parametrize('extension, expected', [(5e-4, True), (2e-3, False)])
def test_closed_segment_endpoint_caps_use_cad_distance(extension, expected):
    first, second = _sources()
    box = ((.2-extension, .5), (.5, .75), (.2-extension, .5), (.5, .75))
    restricted = restrict_source_pair(first, second, box)
    coverage = _coverage([_fragment(.2, .8)])
    assert coverage.covers_cell(*restricted, box) is expected


def test_empty_residual_cannot_overrule_a_registered_contact():
    first, second = _sources()
    # The XY equations force t=v strictly above .5; the graph's height
    # is then positive. The box begins within the known path's tolerance
    # neighborhood but does not carry another equation branch.
    box = ((0., .01), (.5, .515625), (0., .01), (.5005, .53125))
    restricted = restrict_source_pair(first, second, box)
    coverage = _coverage()
    assert not coverage.covers_cell(*restricted, box)
    assert coverage.covers_cell(*restricted, box, allow_empty=True)


def test_point_match_searches_the_shared_parameter_and_xyz_interval():
    fragment = SimpleNamespace(
        stuv_path=np.array([[0., 0., 0., 0.], [1., 0., 1., 0.]]),
        xyz_path=np.array([[0., 0., 0.], [10., 0., 0.]]), tangential=True)
    # The parameter least-squares fraction is .5004; its XYZ error is
    # above atol. Fraction .5008 satisfies all four parameter bounds and
    # the world-space bound simultaneously.
    coverage = _coverage([fragment])
    assert coverage.covers_points([[.5, .0008, .5008, .0008]], [[5.008, .0008, 0.]])
