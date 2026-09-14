"""Numerical real roots of a scalar Bernstein polynomial on its domain.

Derivative roots divide the domain into monotone intervals. This avoids
enumerating a flat multiple root as hundreds of unrelated small residuals.
The caller still checks the resulting geometric candidate at its own
modeling tolerance.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class BernsteinRoots:
    roots: tuple[float, ...]
    constant_zero: bool
    complete: bool
    work: int


def bernstein_roots(coeffs, *, parameter_tol=1e-12, charge=None):
    """Find numerical roots, including stationary roots, in ``[0, 1]``.

    ``parameter_tol`` controls root location, not a polynomial residual
    band. ``charge`` optionally accepts one work unit and returns whether
    it was available. An incomplete result must not replace a general
    intersection search. Coefficient roundoff can still limit root
    separation; this routine does not return algebraic certificates.
    """
    coefficients = np.asarray(coeffs, dtype=float)
    if coefficients.ndim != 1 or not len(coefficients):
        raise ValueError("coeffs must be a nonempty vector")
    if not np.isfinite(coefficients).all():
        raise ValueError("coeffs must be finite")
    if not np.isfinite(parameter_tol) or parameter_tol <= 0:
        raise ValueError("parameter_tol must be finite and positive")
    work = 0
    machine_eps = np.finfo(float).eps

    class Exhausted(Exception):
        pass

    def spend():
        nonlocal work
        if charge is not None and not charge(1):
            raise Exhausted
        work += 1

    def evaluate(net, parameter):
        spend()
        values = net.copy()
        for length in range(len(values)-1, 0, -1):
            values[:length] = ((1-parameter)*values[:length]
                               + parameter*values[1:length+1])
        return float(values[0])

    def solve(net):
        spend()
        scale = float(np.max(np.abs(net)))
        if scale == 0:
            return []
        net = net / scale
        degree = len(net)-1
        if degree == 0:
            return []
        if degree == 1:
            slope = net[1]-net[0]
            if slope == 0:
                return []
            root = -net[0]/slope
            return [float(root)] if 0 <= root <= 1 else []

        derivative = degree*np.diff(net)
        # Degree elevation introduces small differences in an otherwise
        # constant net. It cannot create a zero far from this constant.
        error = 16*degree*machine_eps
        if float(np.max(np.abs(derivative))) <= error:
            return []
        critical = solve(derivative)
        knots = [0.] + [x for x in critical if 0 < x < 1] + [1.]
        values = [evaluate(net, x) for x in knots]
        roots = []
        bracketed = set()
        for index, (left, right, fleft, fright) in enumerate(
                zip(knots, knots[1:], values, values[1:])):
            if fleft == 0 or fright == 0 or np.signbit(fleft) == np.signbit(fright):
                continue
            roots.append(float(brentq(
                lambda x: evaluate(net, x), left, right,
                xtol=parameter_tol, rtol=4*machine_eps)))
            bracketed.update((index, index+1))
        for index, (x, value) in enumerate(zip(knots, values)):
            # A shallow extremum must not replace its two sign-changing
            # roots. They can be far apart in model space even when the
            # height between them is close to arithmetic roundoff.
            interior_touch = (0 < index < len(knots)-1
                              and index not in bracketed and abs(value) <= error)
            if value == 0 or interior_touch:
                roots.append(x)
        roots.sort()
        return roots

    try:
        roots = solve(coefficients)
    except Exhausted:
        return BernsteinRoots((), False, False, work)
    return BernsteinRoots(tuple(roots), bool(np.all(coefficients == 0)), True, work)
