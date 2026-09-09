"""Sufficient identity checks for numerical retraces of one exact boundary.

The adapter must first establish that both paths use the same exact
source boundary curve and source parameterization. This module supplies
the target-preimage proof and continuous approximation check.
"""
from fractions import Fraction

import numpy as np

from mmcore.numeric.intersection.csx._planar_overlap import exact_planar_bilinear_overlap
from mmcore.numeric.intersection.ssx._ssx_polyline import polyline_contained


def certified_boundary_retrace(stuv, xyz, keeper_stuv, keeper_xyz,
                               curve, surface, curve_axis, surface_axes,
                               parameter_domain, param_tol, xyz_tol, charge=None):
    """Prove a retrace of a known shared source curve is already represented.

    Exact planar inclusion gives existence and a unique target preimage
    for every source-curve parameter. Source parameters use zero slack;
    tolerance only validates the numerical representation of that unique
    correspondence. Unsupported target charts retain both paths.
    """
    s, k = np.asarray(stuv), np.asarray(keeper_stuv)
    if len(s) < 2 or len(k) < 2:
        return False
    lo, hi = map(float, parameter_domain)
    if hi <= lo:
        return False
    if (s[:, curve_axis].min() < k[:, curve_axis].min()
            or s[:, curve_axis].max() > k[:, curve_axis].max()):
        return False
    if charge is not None and not charge(max(1, len(curve)*surface.shape[0]*surface.shape[1])):
        return None
    proof = exact_planar_bilinear_overlap(curve, surface, rational=True)
    if not proof:
        return False
    # Compare against the exact clipped interval, never rounded boundary
    # roots. This also bounds any source subcurve used by the keeper.
    start, end = proof[0]['exact_t_range']
    lower, upper = Fraction(int(start[0]), int(start[1])), Fraction(int(end[0]), int(end[1]))
    origin, span = Fraction.from_float(lo), Fraction.from_float(hi)-Fraction.from_float(lo)
    for endpoint in (k[:, curve_axis].min(), k[:, curve_axis].max()):
        parameter = (Fraction.from_float(float(endpoint))-origin)/span
        if not lower <= parameter <= upper:
            return False
    tolerances = np.array(param_tol, dtype=float, copy=True)
    for axis in range(4):
        if axis not in surface_axes:
            tolerances[axis] = 0.
    return polyline_contained(s, xyz, k, keeper_xyz, tolerances, xyz_tol, charge=charge)
