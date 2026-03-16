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
