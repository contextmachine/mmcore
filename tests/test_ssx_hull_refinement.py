"""Independent analytic coverage for private rational child-hull exclusions."""
import numpy as np
import pytest

from mmcore.nurbs._nurbs_eval import NURBSSurfaceTuple
from mmcore.numeric.intersection.ssx import _ssx_hull_refine as refine


ATOL = 1e-3


def _patch(cp, weights=None):
    cp = np.asarray(cp, dtype=float)
    nu, nv = cp.shape[:2]
    return NURBSSurfaceTuple(nu, nv, np.r_[np.zeros(nu), np.ones(nu)],
                            np.r_[np.zeros(nv), np.ones(nv)], cp,
                            np.ones((nu, nv)) if weights is None else np.asarray(weights, float))


def _graph(gap, rational=False):
    # z=(2u-1)^2 for equal weights, or that same numerator divided by
    # the positive quadratic weight polynomial. Adding gap shifts z by gap.
    weights = np.array([1., .2 if rational else 1., 1.])
    z = np.array([1., -1., 1.]) / weights + gap
    return _patch([[[x, y, z[i]] for y in (-1., 1.)]
                   for i, x in enumerate((-1., 0., 1.))],
                  np.repeat(weights[:, None], 2, axis=1))


def _plane():
    return _patch([[[-1., -1., 0.], [-1., 1., 0.]],
                   [[1., -1., 0.], [1., 1., 0.]]])


def _projection_filter(directions=None, calls=None):
    # Independent hull oracle: interval separation along supplied unit axes.
    # It is deliberately incomplete but every exclusion is conservative.
    axes = np.eye(3) if directions is None else np.asarray(directions, float)
    axes = axes / np.linalg.norm(axes, axis=1)[:, None]

    def filter_hulls(first, second, pairs, atol):
        if calls is not None:
            calls.append((len(pairs), atol))
        result = []
        for i, j in pairs:
            a = first[i].control_points.reshape(-1, 3) @ axes.T
            b = second[j].control_points.reshape(-1, 3) @ axes.T
            gap = np.maximum(a.min(0) - b.max(0), b.min(0) - a.max(0))
            if not np.any(np.isfinite(gap) & (gap > 2 * atol)):
                result.append((i, j))
        return result
    return filter_hulls


@pytest.mark.parametrize('rational', [False, True])
@pytest.mark.parametrize('depth', [1, 2])
def test_curved_positive_gap_excludes_only_after_real_subdivision(rational, depth):
    first, second = _graph(4 * ATOL, rational), _plane()
    oracle = _projection_filter()
    assert oracle([first], [second], [(0, 0)], ATOL) == [(0, 0)]
    before = first.control_points.copy(), first.weights.copy()
    assert refine._refine_patch_pairs([first], [second], [(0, 0)], ATOL,
                                     filter_hulls=oracle, max_depth=depth) == []
    assert np.array_equal(first.control_points, before[0])
    assert np.array_equal(first.weights, before[1])


@pytest.mark.parametrize('rational', [False, True])
@pytest.mark.parametrize('gap', [0., 1.9 * ATOL, 2 * ATOL])
def test_real_tangent_and_two_sided_tolerance_contacts_are_retained(rational, gap):
    assert refine._refine_patch_pairs([_graph(gap, rational)], [_plane()], [(0, 0)],
                                     ATOL, filter_hulls=_projection_filter()) == [(0, 0)]


def test_nongauge_rational_arc_is_separated_from_plane_beyond_its_tangent():
    cp = np.array([[[1., -1., 0.], [1., 1., 0.]],
                   [[1., -1., 1.], [1., 1., 1.]],
                   [[0., -1., 1.], [0., 1., 1.]]])
    arc = _patch(cp, [[1., 1.], [.2, .2], [1., 1.]])
    normal = np.array([1., 0., 1.]) / np.sqrt(2.)
    tangent = np.array([-1., 0., 1.]) / np.sqrt(2.)
    center = np.array([7. / 12., 0., 7. / 12.]) + 4 * ATOL * normal
    v = np.array([0., 1., 0.])
    plane = _patch([[center - tangent - v, center - tangent + v],
                    [center + tangent - v, center + tangent + v]])
    oracle = _projection_filter([normal])
    assert oracle([arc], [plane], [(0, 0)], ATOL) == [(0, 0)]
    assert refine._refine_patch_pairs([arc], [plane], [(0, 0)], ATOL,
                                     filter_hulls=oracle) == []


@pytest.mark.parametrize('variant', ['rotate', 'translate', 'small_units', 'large_units',
                                     'swap', 'transpose', 'reverse'])
def test_original_pair_decisions_survive_geometry_and_chart_changes(variant):
    angle = .71
    rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                         [-np.sin(angle), 0., np.cos(angle)]])
    for gap, keep in [(4 * ATOL, False), (1.9 * ATOL, True)]:
        original = [_graph(gap, True), _plane()]
        atol, axes = ATOL, np.eye(3)
        transformed = []
        for patch in original:
            points, weights = patch.control_points.copy(), patch.weights.copy()
            if variant == 'rotate':
                points = points @ rotation.T
                axes = rotation.T
            elif variant == 'translate':
                points += [1.e6, -2.e6, 3.e6]
            elif variant in ('small_units', 'large_units'):
                scale = .01 if variant == 'small_units' else 1000.
                points *= scale
                atol = ATOL * scale
            elif variant == 'transpose':
                points, weights = points.swapaxes(0, 1), weights.T
            elif variant == 'reverse':
                points, weights = points[::-1], weights[::-1]
            transformed.append(_patch(points, weights))
        if variant == 'swap':
            transformed.reverse()
        result = refine._refine_patch_pairs([transformed[0]], [transformed[1]], [(0, 0)],
                                           atol, filter_hulls=_projection_filter(axes))
        assert result == ([(0, 0)] if keep else [])


def test_only_original_identifiers_and_order_leave_the_refinement():
    patches = [_graph(0.), _graph(4 * ATOL), _graph(ATOL)]
    pairs = [(2, 1), (1, 1), (0, 1), (2, 1)]
    result = refine._refine_patch_pairs(patches, [_plane(), _plane()], pairs, ATOL,
                                      filter_hulls=_projection_filter())
    assert result == [(2, 1), (0, 1), (2, 1)]


def test_subnormal_normalized_weights_decline_subdivision():
    patch = _graph(4 * ATOL)
    weights = patch.weights.copy()
    weights[1] = np.finfo(float).tiny / 2.
    patch = _patch(patch.control_points, weights)
    assert refine._split_hull(patch) is None
    assert refine._refine_patch_pairs([patch], [_plane()], [(0, 0)], ATOL,
                                     filter_hulls=_projection_filter()) == [(0, 0)]


def test_split_work_limit_leaves_the_original_hull_available(monkeypatch):
    monkeypatch.setattr(refine, '_MAX_SPLIT_WORK', 1)
    patch = _graph(4*ATOL)
    assert refine._split_hull(patch) is None
    assert refine._refine_patch_pairs([patch], [_plane()], [(0, 0)], ATOL,
                                     filter_hulls=_projection_filter()) == [(0, 0)]


def test_extreme_weight_gauge_is_normalized_before_homogeneous_products():
    patch = _graph(4 * ATOL)
    scaled_weights = patch.weights * np.finfo(float).max
    patch = _patch(patch.control_points, scaled_weights)
    # Some Cartesian controls exceed one: multiplying by these original
    # weights would overflow, although their common gauge changes no geometry.
    assert refine._refine_patch_pairs([patch], [_plane()], [(0, 0)], ATOL,
                                     filter_hulls=_projection_filter()) == []


def test_comparison_budget_preserves_any_parent_with_untested_descendants(monkeypatch):
    monkeypatch.setattr(refine, '_MAX_CHILD_PAIRS', 5)
    pairs = [(0, 0), (1, 0)]
    result = refine._refine_patch_pairs([_graph(4 * ATOL), _graph(4 * ATOL)],
                                      [_plane()], pairs, ATOL,
                                      filter_hulls=_projection_filter())
    assert result == [(1, 0)]


def test_failed_hull_comparison_retains_original_parent():
    def failed(*args):
        raise FloatingPointError('synthetic unavailable certificate')

    assert refine._refine_patch_pairs([_graph(4 * ATOL)], [_plane()], [(0, 0)], ATOL,
                                     filter_hulls=failed) == [(0, 0)]


def test_zero_depth_preserves_original_pairs_without_callback():
    def forbidden(*args):
        raise AssertionError('no hull refinement was requested')

    pairs = [(0, 0)]
    assert refine._refine_patch_pairs([_graph(4 * ATOL)], [_plane()], pairs, ATOL,
                                     filter_hulls=forbidden, max_depth=0) == pairs


def test_roundoff_cushion_only_inflates_the_exclusion_callback_tolerance():
    calls = []
    first, second = _graph(2 * ATOL, True), _plane()
    result = refine._refine_patch_pairs([first], [second], [(0, 0)], ATOL,
                                      filter_hulls=_projection_filter(calls=calls))
    assert result == [(0, 0)]
    assert calls and all(tolerance > ATOL for _, tolerance in calls)
    assert calls[-1][1] >= calls[0][1]


def test_budget_exhaustion_after_one_descendant_keeps_the_same_parent(monkeypatch):
    monkeypatch.setattr(refine, '_MAX_CHILD_PAIRS', 5)
    calls = []
    oracle = _projection_filter()

    def initially_ambiguous(first, second, pairs, tolerance):
        calls.append(len(pairs))
        # An incomplete certificate may retain every first-level child.
        # At level two the budget can cover only one of those descendants.
        if len(calls) == 1:
            return pairs
        return oracle(first, second, pairs, tolerance)

    assert refine._refine_patch_pairs([_graph(4 * ATOL)], [_plane()], [(0, 0)], ATOL,
                                     filter_hulls=initially_ambiguous) == [(0, 0)]
    assert len(calls) == 2
