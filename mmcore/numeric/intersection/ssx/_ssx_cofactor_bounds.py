"""Source-valid interval bounds for the four SSX Jacobian column minors.

Precomputed floating determinant polynomials can lose their signs through
cancellation. Uniform-weight sources use outward Bernstein coefficient
products which preserve polynomial dependency inside each surface normal.
The general fallback encloses the original unsquared residual's Jacobian
before forming interval determinants. A failed strict sign is unknown,
never an exclusion.
"""
from itertools import permutations
from math import prod, comb
from fractions import Fraction

import numpy as np

from mmcore.numeric._bezier_common import restrict_net_axis_v
from mmcore.numeric.intersection._root_box_certificate import residual_roundoff_bound
from mmcore.numeric.intersection.ssx._ssx5_singular import psi_vector_net


def _multiply(first, second):
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        products = np.array([a*b for a in first for b in second])
    if not np.all(np.isfinite(products)):
        return -np.inf, np.inf
    return float(np.nextafter(products.min(), -np.inf)), float(np.nextafter(products.max(), np.inf))


def _determinant(lower, upper):
    lo = hi = 0.
    for order in permutations(range(3)):
        term = (1., 1.)
        for row, column in enumerate(order):
            term = _multiply(term, (lower[row, column], upper[row, column]))
        odd = sum(order[a] > order[b] for a in range(3) for b in range(a+1, 3)) % 2
        if odd:
            term = -term[1], -term[0]
        lo = float(np.nextafter(lo+term[0], -np.inf))
        hi = float(np.nextafter(hi+term[1], np.inf))
    return lo, hi


def _interval_multiply(al, au, bl, bu):
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        values = np.stack(np.broadcast_arrays(al*bl, al*bu, au*bl, au*bu))
        low, high = np.min(values, axis=0), np.max(values, axis=0)
    invalid = np.isnan(low) | np.isnan(high)
    return (np.where(invalid, -np.inf, np.nextafter(low, -np.inf)),
            np.where(invalid, np.inf, np.nextafter(high, np.inf)))


def _interval_subtract(al, au, bl, bu):
    return np.nextafter(al-bu, -np.inf), np.nextafter(au-bl, np.inf)


def _source_polynomial_column(source, axis, opposite_weight, negative=False):
    degree = source.shape[axis]-1
    if degree == 0:
        shape = list(source.shape)
        shape[axis], shape[-1] = 1, 3
        return np.zeros(shape), np.zeros(shape)
    a = np.take(source[..., :3], range(1, degree+1), axis=axis)
    b = np.take(source[..., :3], range(degree), axis=axis)
    low, high = _interval_subtract(a, a, b, b)
    low, high = _interval_multiply(low, high, float(degree), float(degree))
    low, high = _interval_multiply(low, high, opposite_weight, opposite_weight)
    return (-high, -low) if negative else (low, high)


def _bernstein_cross(first, second):
    """2D Bernstein cross product with outward coefficient arithmetic.

    Product weights are exact binomial ratios. Their floating enclosures
    and every accumulation are outward rounded, including subnormals;
    no error estimate depends on the size of the cancelled result.
    """
    al, au = first
    bl, bu = second
    adeg, bdeg = np.array(al.shape[:2])-1, np.array(bl.shape[:2])-1
    shape = tuple(adeg+bdeg+1)+(3,)
    low, high = np.zeros(shape), np.zeros(shape)
    for ia in np.ndindex(al.shape[:2]):
        for ib in np.ndindex(bl.shape[:2]):
            target = tuple(x+y for x, y in zip(ia, ib))
            weight = prod(Fraction(comb(int(a), i)*comb(int(b), j),
                                   comb(int(a+b), i+j))
                          for a, b, i, j in zip(adeg, bdeg, ia, ib))
            rounded = float(weight)
            wl, wu = np.nextafter(rounded, -np.inf), np.nextafter(rounded, np.inf)
            pl, pu = _interval_multiply(al[ia][[1,2,0]], au[ia][[1,2,0]],
                                        bl[ib][[2,0,1]], bu[ib][[2,0,1]])
            ql, qu = _interval_multiply(al[ia][[2,0,1]], au[ia][[2,0,1]],
                                        bl[ib][[1,2,0]], bu[ib][[1,2,0]])
            term = _interval_subtract(pl, pu, ql, qu)
            tl, tu = _interval_multiply(*term, wl, wu)
            low[target] = np.nextafter(low[target]+tl, -np.inf)
            high[target] = np.nextafter(high[target]+tu, np.inf)
    return low, high


def _outer_dot(first, second):
    shape = first[0].shape[:2]+second[0].shape[:2]
    low, high = np.zeros(shape), np.zeros(shape)
    for coordinate in range(3):
        a = tuple(values[..., coordinate, None, None] for values in first)
        b = tuple(values[None, None, ..., coordinate] for values in second)
        tl, tu = _interval_multiply(*a, *b)
        low = np.nextafter(low+tl, -np.inf)
        high = np.nextafter(high+tu, np.inf)
    return low, high


def _coefficient_centers(interval):
    low, high = interval
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        return None
    center = .5*low+.5*high
    radius = np.nextafter(np.maximum(center-low, high-center), np.inf)
    # Exact equality of floating midpoint coefficients proves that this
    # midpoint polynomial is constant along an axis. Removing that
    # redundant degree changes no polynomial; the inherited global
    # coefficient radius still encloses the original exact source net.
    # In particular affine-plane normals need no foreign UV subdivision.
    for axis in range(center.ndim):
        if center.shape[axis] > 1:
            representative = np.take(center, [0], axis=axis)
            if np.all(center == representative):
                center = representative
    # Each exact coefficient lies within this fixed source error. The
    # Bernstein basis is nonnegative and sums to one under restriction.
    error = np.nextafter(float(np.max(radius))
                         + residual_roundoff_bound(center[..., None], depth=2)[0], np.inf)
    if not np.isfinite(error):
        return None
    return center[..., None], error


class SourceCofactorBounds:
    """Cached global-parameter cofactor bounds over source product boxes.

    ``first`` and ``second`` are the ORIGINAL homogeneous Bezier control
    nets, with positive weights. ``bounds(box)`` returns ``(lower, upper)``
    for the four unsigned column minors of dPsi/d(s,t,u,v). Their signed
    tangent direction is ``(+T1,-T2,+T3,-T4)``. Bounds remain in GLOBAL
    parameter coordinates, even for tiny or degenerate requested boxes.

    Work is charged before construction/restriction in the SSX ledger's
    existing 128-coefficient units. A denied charge returns None and sets
    ``exhausted``. Requests for an already cached box do not spend again.
    """
    def __init__(self, first, second, *, charge=None, polynomial=True):
        self.charge = charge
        self.exhausted = False
        self.cache = {}
        self.derivatives = None
        self.errors = None
        self.polynomial_nets = None
        self.residual = None
        self.residual_error = None
        first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
        if (first.ndim != 3 or second.ndim != 3 or first.shape[-1] != 4
                or second.shape[-1] != 4 or not np.all(np.isfinite(first))
                or not np.all(np.isfinite(second)) or not np.all(first[..., 3] > 0.)
                or not np.all(second[..., 3] > 0.)):
            return
        coefficient_count = 3*prod(first.shape[:2]+second.shape[:2])
        if not self._spend(max(1, (5*coefficient_count+127)//128)):
            return
        with np.errstate(over='ignore', invalid='ignore', under='ignore'):
            net = psi_vector_net(first, second)
            source = (np.max(np.abs(first[..., :3]), axis=(0, 1))*np.max(np.abs(second[..., 3]))
                      + np.max(np.abs(second[..., :3]), axis=(0, 1))*np.max(np.abs(first[..., 3])))
        if not np.all(np.isfinite(net)) or not np.all(np.isfinite(source)):
            return
        # This source-bound includes the homogeneous cross products and
        # subtraction. Differentiation precedes restriction, so none of
        # these error floors depends on a shrinking/cancelled child net.
        original_error = residual_roundoff_bound(net, source_scale=source)
        self.residual = net
        self.residual.setflags(write=False)
        self.residual_error = np.nextafter(
            original_error+residual_roundoff_bound(net, depth=2), np.inf)
        magnitude = np.max(np.abs(net), axis=tuple(range(4)))
        eps = np.finfo(float).eps
        gamma2 = 2.*eps/(1.-2.*eps)
        derivatives, errors = [], []
        for axis in range(4):
            degree = net.shape[axis]-1
            if degree == 0:
                shape = list(net.shape)
                shape[axis] = 1
                derivative = np.zeros(shape)
                derivative_error = np.zeros(3)
            else:
                derivative = degree*np.diff(net, axis=axis)
                # Two inherited endpoint errors plus the subtraction and
                # integer-degree multiplication used for each derivative.
                derivative_error = np.nextafter(
                    2.*degree*original_error+2.*gamma2*degree*magnitude
                    + 2.*np.nextafter(0., 1.), np.inf)
            restriction_error = residual_roundoff_bound(derivative, depth=2)
            derivatives.append(derivative)
            errors.append(np.nextafter(derivative_error+restriction_error, np.inf))
        self.derivatives = tuple(derivatives)
        self.errors = tuple(errors)
        self.restriction_units = max(1, (sum(d.size for d in derivatives)+127)//128)
        if (polynomial and np.all(first[..., 3] == first[0, 0, 3])
                and np.all(second[..., 3] == second[0, 0, 3])):
            shapes = [tuple(max(1, n-(axis == k)) for k, n in enumerate(source.shape[:2]))
                      for source in (first, second) for axis in (0, 1)]
            normal_shapes = [tuple(a+b-1 for a, b in zip(shapes[k], shapes[k+1]))
                             for k in (0, 2)]
            pair_work = sum(prod(shapes[k])*prod(shapes[k+1]) for k in (0, 2))
            output_count = sum(prod(shapes[k])*prod(normal_shapes[1-k//2]) for k in range(4))
            if not self._spend(max(1, (24*pair_work+12*output_count+127)//128)):
                return
            with np.errstate(over='ignore', invalid='ignore', under='ignore'):
                columns = [_source_polynomial_column(source, axis, weight, negative)
                           for source, weight, negative in (
                               (first, second[0,0,3], False), (second, first[0,0,3], True))
                           for axis in (0, 1)]
                normals = [_bernstein_cross(columns[k], columns[k+1]) for k in (0, 2)]
                nets = [_coefficient_centers(_outer_dot(columns[1], normals[1])),
                        _coefficient_centers(_outer_dot(columns[0], normals[1])),
                        _coefficient_centers(_outer_dot(normals[0], columns[3])),
                        _coefficient_centers(_outer_dot(normals[0], columns[2]))]
            if all(net is not None for net in nets):
                self.polynomial_nets = tuple(nets)
                self.polynomial_units = max(1, (sum(net.size for net, _ in nets)+127)//128)

    def _spend(self, amount):
        if self.exhausted:
            return False
        if self.charge is None or self.charge(int(amount)):
            return True
        self.exhausted = True
        return False

    def bounds(self, box):
        if self.derivatives is None or self.exhausted:
            return None
        key = tuple((float(lo), float(hi)) for lo, hi in box)
        if (len(key) != 4 or any(not (0. <= lo <= hi <= 1.) for lo, hi in key)):
            return None
        if key in self.cache:
            return self.cache[key]
        plow = phigh = None
        if self.polynomial_nets is not None:
            if not self._spend(self.polynomial_units):
                return None
            polynomial_bounds = []
            for net, error in self.polynomial_nets:
                restricted = net
                for axis, (lo, hi) in enumerate(key):
                    restricted = restrict_net_axis_v(restricted, axis, lo, hi, 0., 1.)
                polynomial_bounds.append((np.nextafter(float(restricted.min())-error, -np.inf),
                                          np.nextafter(float(restricted.max())+error, np.inf)))
            plow, phigh = np.asarray(polynomial_bounds).T
            if np.any((plow > 0.) | (phigh < 0.)):
                plow.setflags(write=False)
                phigh.setflags(write=False)
                self.cache[key] = plow, phigh
                return self.cache[key]
        if not self._spend(self.restriction_units):
            return None
        jacobian_lower, jacobian_upper = [], []
        for derivative, error in zip(self.derivatives, self.errors):
            restricted = derivative
            for axis, (lo, hi) in enumerate(key):
                restricted = restrict_net_axis_v(restricted, axis, lo, hi, 0., 1.)
            jacobian_lower.append(np.nextafter(restricted.min(axis=(0, 1, 2, 3))-error, -np.inf))
            jacobian_upper.append(np.nextafter(restricted.max(axis=(0, 1, 2, 3))+error, np.inf))
        jlo, jhi = np.asarray(jacobian_lower).T, np.asarray(jacobian_upper).T
        bounds = [_determinant(np.delete(jlo, axis, axis=1), np.delete(jhi, axis, axis=1))
                  for axis in range(4)]
        lower, upper = np.asarray(bounds).T
        if plow is not None:
            combined_low, combined_high = np.maximum(lower, plow), np.minimum(upper, phigh)
            if np.all(combined_low <= combined_high):
                lower, upper = combined_low, combined_high
        lower.setflags(write=False)
        upper.setflags(write=False)
        self.cache[key] = lower, upper
        return self.cache[key]


class SourceZeroSetCofactorBounds:
    """Bounds on the source zero set inside a requested parameter box.

    Exact affine coordinate identities contract a necessary box before the
    source cofactor bounds are evaluated. The result encloses cofactors at
    every intersection in the original box, but is deliberately NOT a
    bound away from the zero set. It supports regularity, oriented germs,
    and tangent ratios; it must not be used for a residual Krawczyk test.
    """
    def __init__(self, source_bounds, constraints):
        self.source_bounds = source_bounds
        self.constraints = constraints
        self.contractor = None
        self.cache = {}
        self.empty_boxes = set()

    def bounds(self, box):
        original = tuple((float(lo), float(hi)) for lo, hi in box)
        if original in self.cache:
            return self.cache[original]
        necessary_box = self.constraints.contract(box)
        if necessary_box is None:
            self.empty_boxes.add(original)
            self.cache[original] = None
            return None
        initial = self.source_bounds.bounds(necessary_box)
        if initial is None or np.any((initial[0] > 0.) | (initial[1] < 0.)):
            self.cache[original] = initial
            return initial
        if self.contractor is None:
            from mmcore.numeric.intersection.ssx._ssx_residual_contract import SourceResidualContractor
            self.contractor = SourceResidualContractor(self.source_bounds)
        def regular(candidate):
            bounds = self.source_bounds.bounds(candidate)
            return bool(bounds is not None and np.any((bounds[0] > 0.) | (bounds[1] < 0.)))
        contracted = self.contractor.contract(necessary_box, stop_when=regular)
        if contracted is None:
            self.empty_boxes.add(original)
            self.cache[original] = None
            return None
        result = (initial if contracted == necessary_box
                  else self.source_bounds.bounds(contracted))
        self.cache[original] = result
        return result
