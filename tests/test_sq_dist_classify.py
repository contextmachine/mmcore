import numpy as np
import pytest
from mmcore.numeric.bern_sq_dist import curve_curve_squared_net_homog
from mmcore.numeric.intersection._sq_dist_classify import (
    classify_sq_dist_net, Classification,
    NO_INTERSECTION, UNIQUE_ISOLATED, OVERLAP, INDETERMINATE,
)

def test_no_intersection_disjoint_curves():
    C1 = np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]])
    C2 = np.array([[0.0, 10.0, 0.0, 1.0], [1.0, 10.0, 0.0, 1.0]])
    F = curve_curve_squared_net_homog(C1, C2, rational=True)
    Pw, Qw = C1[:, -1], C2[:, -1]
    result = classify_sq_dist_net(F, atol=1e-3, Pw=Pw, Qw=Qw)
    assert result.kind == NO_INTERSECTION

def test_no_intersection_parallel_offset():
    # Offset must be large enough that Bernstein coefficients are all positive
    # (a 0.2 offset produces negative coefficients due to Bernstein overestimation
    #  and would need subdivision to classify; offset=2.0 is provable from the net).
    C1 = np.array([[0.0, 0.0, 0.0, 1.0], [0.5, 0.5, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]])
    C2 = np.array([[0.0, 2.0, 0.0, 1.0], [0.5, 2.5, 0.0, 1.0], [1.0, 2.0, 0.0, 1.0]])
    F = curve_curve_squared_net_homog(C1, C2, rational=True)
    Pw, Qw = C1[:, -1], C2[:, -1]
    result = classify_sq_dist_net(F, atol=1e-3, Pw=Pw, Qw=Qw)
    assert result.kind == NO_INTERSECTION

def test_intersecting_curves_not_pruned():
    C1 = np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]])
    C2 = np.array([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [2.0, 1.0, 0.0, 1.0]])
    F = curve_curve_squared_net_homog(C1, C2, rational=True)
    Pw, Qw = C1[:, -1], C2[:, -1]
    result = classify_sq_dist_net(F, atol=1e-3, Pw=Pw, Qw=Qw)
    assert result.kind != NO_INTERSECTION


from mmcore.numeric.bern import de_casteljau_split_nd

def test_unique_isolated_single_crossing():
    """Full domain with two crossings — uniqueness likely unprovable."""
    C1 = np.array([[0.0, 0.0, 0.0, 1.0], [0.5, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]])
    C2 = np.array([[0.0, 1.0, 0.0, 1.0], [0.5, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]])
    F = curve_curve_squared_net_homog(C1, C2, rational=True)
    Pw, Qw = C1[:, -1], C2[:, -1]
    result = classify_sq_dist_net(F, atol=1e-3, Pw=Pw, Qw=Qw)
    # Full domain has two crossings, so should NOT be NO_INTERSECTION
    assert result.kind != NO_INTERSECTION


def test_unique_isolated_after_subdivision():
    """Subdivide to isolate one crossing — uniqueness may become provable."""
    C1 = np.array([[0.0, 0.0, 0.0, 1.0], [0.5, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]])
    C2 = np.array([[0.0, 1.0, 0.0, 1.0], [0.5, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]])
    F = curve_curve_squared_net_homog(C1, C2, rational=True)
    F_with_dim = F[..., np.newaxis]
    left, right = de_casteljau_split_nd(F_with_dim, axis=0, t=0.5)
    left_sq = np.squeeze(left, axis=-1)
    right_sq = np.squeeze(right, axis=-1)
    Pw, Qw = C1[:, -1], C2[:, -1]
    r_left = classify_sq_dist_net(left_sq, atol=1e-3, Pw=Pw, Qw=Qw)
    r_right = classify_sq_dist_net(right_sq, atol=1e-3, Pw=Pw, Qw=Qw)
    # Both halves have crossings — neither should be NO_INTERSECTION
    assert r_left.kind != NO_INTERSECTION
    assert r_right.kind != NO_INTERSECTION


def test_overlap_identical_curves():
    """Identical curves — F is identically zero, should detect overlap."""
    C1 = np.array([[0.0, 0.0, 0.0, 1.0], [0.5, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]])
    C2 = C1.copy()
    F = curve_curve_squared_net_homog(C1, C2, rational=True)
    Pw, Qw = C1[:, -1], C2[:, -1]
    result = classify_sq_dist_net(F, atol=1e-3, Pw=Pw, Qw=Qw)
    assert result.kind == OVERLAP


def test_overlap_identical_lines():
    """Two identical lines — simplest overlap case."""
    C1 = np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]])
    C2 = C1.copy()
    F = curve_curve_squared_net_homog(C1, C2, rational=True)
    Pw, Qw = C1[:, -1], C2[:, -1]
    result = classify_sq_dist_net(F, atol=1e-3, Pw=Pw, Qw=Qw)
    assert result.kind == OVERLAP
