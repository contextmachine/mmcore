"""Exact plane/convex-bilinear overlap certificates, independent of atol.

All predicates below use rational arithmetic on the supplied binary
floating-point coefficients. A small nonzero plane residual is rejected;
it never becomes an identity because it lies within a geometric tolerance.

The supported surface is a regular, planar polynomial bilinear patch
(uniform homogeneous weights are equivalent). Strict convexity of its
four boundary vertices gives an injective map of the parameter square
onto that quadrilateral: its boundary is simple and its planar Jacobian
has the same strict sign at all four corners. The Jacobian is affine in
u and v, so it cannot vanish in the square.

A positive-weight rational Bezier curve lies in the convex hull of its
Cartesian control points. If that hull lies in the quadrilateral and
all its homogeneous coefficients satisfy the plane identity, the entire
curve has a unique surface preimage. For degree-one curves we additionally
compute the exact, maximal clipped span by intersecting four halfspaces.
Unsupported or inconclusive inputs return None and leave general CSX
responsible for the case. Numerical boundary roots are not used as proof.
"""
from fractions import Fraction

import numpy as np


def _sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1],
            a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0])


def _dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def _cross2(a, b):
    return a[0]*b[1]-a[1]*b[0]


def _exact_points(net, rational):
    flat = np.asarray(net, dtype=float).reshape(-1, 4 if rational else 3)
    points, weights = [], []
    for row in flat:
        coefficients = tuple(Fraction.from_float(float(value)) for value in row)
        weight = coefficients[3] if rational else Fraction(1)
        if weight <= 0:
            return None
        points.append(tuple(value/weight for value in coefficients[:3]))
        weights.append(weight)
    return points, weights


def _convex_chart(surface_points):
    # Input flattening order is (00,01,10,11); polygon boundary order
    # follows (00,10,11,01), consistently with the parameter square.
    quad3 = [surface_points[i] for i in (0, 2, 3, 1)]
    origin = quad3[0]
    normal = _cross(_sub(quad3[1], origin), _sub(quad3[3], origin))
    if not any(normal) or any(_dot(normal, _sub(p, origin)) for p in quad3):
        return None
    # The exact proof permits any nonzero component. The largest one
    # gives the numerical inverse the best-conditioned coordinate plane.
    drop = max(range(3), key=lambda i: abs(normal[i]))
    axes = tuple(i for i in range(3) if i != drop)
    quad = [tuple(p[i] for i in axes) for p in quad3]
    edges = [_sub(quad[(i+1) % 4], quad[i]) for i in range(4)]
    turns = [_cross2(edges[i], edges[(i+1) % 4]) for i in range(4)]
    if all(turn > 0 for turn in turns):
        orientation = 1
    elif all(turn < 0 for turn in turns):
        orientation = -1
    else:
        return None
    return origin, normal, axes, quad, edges, orientation


def exact_planar_bilinear_overlap(C, S, boundary_roots=None, rational=False):
    """Return exact overlap dictionaries, or None when no proof is available.

    ``boundary_roots`` is accepted for a CSX integration interface, but
    approximate root samples are intentionally irrelevant to the proof.
    The u/v ranges are conservative [0,1] enclosures, not an assertion
    that a curved inverse parameterization interpolates its endpoints.
    Surface nonuniform weights and partially clipped higher-degree curves
    are currently unsupported; they do not receive a weaker certificate.
    """
    del boundary_roots
    curve = np.asarray(C, dtype=float)
    surface = np.asarray(S, dtype=float)
    dimension = 4 if rational else 3
    if (curve.ndim != 2 or curve.shape[1] != dimension or len(curve) < 2
            or surface.shape != (2, 2, dimension)
            or not np.all(np.isfinite(curve)) or not np.all(np.isfinite(surface))):
        return None
    exact_curve = _exact_points(curve, rational)
    exact_surface = _exact_points(surface, rational)
    if exact_curve is None or exact_surface is None:
        return None
    curve_points, curve_weights = exact_curve
    surface_points, surface_weights = exact_surface
    if any(weight != surface_weights[0] for weight in surface_weights):
        return None
    chart = _convex_chart(surface_points)
    if chart is None:
        return None
    origin, normal, axes, quad, edges, orientation = chart
    if any(_dot(normal, _sub(point, origin)) for point in curve_points):
        return None
    if all(point == curve_points[0] for point in curve_points):
        return None  # A constant curve belongs to the parameter-fiber tier.
    planar_curve = [tuple(point[i] for i in axes) for point in curve_points]

    def halfspace(point, edge_index):
        return orientation * _cross2(edges[edge_index], _sub(point, quad[edge_index]))

    lo, hi = Fraction(0), Fraction(1)
    if len(curve_points) == 2:
        # First clip in the affine Cartesian line parameter lambda.
        for edge_index in range(4):
            a = halfspace(planar_curve[0], edge_index)
            delta = halfspace(planar_curve[1], edge_index) - a
            if delta > 0:
                lo = max(lo, -a/delta)
            elif delta < 0:
                hi = min(hi, -a/delta)
            elif a < 0:
                return None
        if hi <= lo:
            return None
        # Positive weights make lambda(t)=w1*t/(w0*(1-t)+w1*t)
        # strictly increasing. Its inverse is exact rational arithmetic.
        w0, w1 = curve_weights

        def parameter(value):
            return value*w0 / (w1*(1-value)+value*w0)

        lo, hi = parameter(lo), parameter(hi)
    elif any(halfspace(point, edge_index) < 0
             for point in planar_curve for edge_index in range(4)):
        return None

    return [{
        "boundary_zeros": [],
        "overlap_endpoints": [],
        "t_range": (float(lo), float(hi)),
        "u_range": (0.0, 1.0),
        "v_range": (0.0, 1.0),
        "uv_range_is_enclosure": True,
        "certification": "exact",
        "parameterization": "unique_bilinear_inverse",
        "proof": "exact_plane_identity_and_convex_quad_inclusion",
        # Preserve exact endpoint rationals as decimal strings, avoiding
        # any claim that their rounded floats are exact boundary roots.
        "exact_t_range": ((str(lo.numerator), str(lo.denominator)),
                          (str(hi.numerator), str(hi.denominator))),
    }]
