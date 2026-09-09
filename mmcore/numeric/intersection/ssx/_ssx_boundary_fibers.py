"""Original-source collapsed boundary seeds, with no census completeness claim.

Exact constant-image edges own parameter fibers. Almost collapsed edges may
suggest a numerical continuation seed, but never acquire a fiber certificate.
Both leave the entire source face unresolved: enumerating its target boundary
preimages does not enumerate every target interior preimage or incident arc.
"""
from fractions import Fraction

import numpy as np

from mmcore.numeric._bezier_common import eval_surface, eval_surface_d1, geometry_collapsed
from mmcore.numeric.intersection._exact_univariate import (
    coefficient_build_work, isolate_root_intervals, _power, _gcd, _refine,
)
from mmcore.numeric.intersection._root_box_certificate import exact_bernstein_value


def _edge(surface, axis, side):
    return surface[0 if side == 0 else -1] if axis == 0 else surface[:, 0 if side == 0 else -1]


def _point(net, parameters):
    value = exact_bernstein_value(net, parameters)
    return tuple(x/value[3] for x in value[:3])


def _point_work(net):
    return int(net.size)*(1+sum(size-1 for size in net.shape[:-1]))


def _planar_point_preimage(point, target, tick):
    """Exact membership in an injective convex polynomial bilinear chart."""
    if target.shape != (2, 2, 4) or not np.all(target[..., 3] == target[0, 0, 3]):
        return None
    from mmcore.numeric.intersection.csx._planar_overlap import (
        _exact_points, _convex_chart, _dot, _sub, _cross2)
    from mmcore.numeric.intersection.csx._planar_roots import _bilinear_inverse
    tick(8*target.size)
    points, _ = _exact_points(target, True)
    chart = _convex_chart(points)
    if chart is None:
        return None
    origin, normal, axes, quad, edges, orientation = chart
    projected = tuple(point[i] for i in axes)
    if (_dot(normal, _sub(point, origin)) != 0 or any(
            orientation*_cross2(edges[i], _sub(projected, quad[i])) < 0 for i in range(4))):
        return None
    a, c_end, b_end, opposite = points
    b, c = _sub(b_end, a), _sub(c_end, a)
    if all(opposite[i] == a[i]+b[i]+c[i] for i in range(3)):
        i, j = axes
        det = b[i]*c[j]-b[j]*c[i]
        offset = _sub(point, a)
        u = (offset[i]*c[j]-offset[j]*c[i])/det
        v = (b[i]*offset[j]-b[j]*offset[i])/det
        return (float(u), float(v)), ((u, u), (v, v))
    uv = _bilinear_inverse(np.array([float(x) for x in point]),
                           target[..., :3]/target[..., 3:], axes)
    if uv is None:
        return None
    return tuple(uv), ((Fraction(0), Fraction(1)),)*2


def source_boundary_fiber_candidate(first, second, axis, side):
    """Cheap proposal gate only; exact constant-image proof is separate."""
    source = first if axis < 2 else second
    curve = _edge(np.asarray(source), axis % 2, side)
    return (bool(np.all(np.isfinite(curve))) and bool(np.all(curve[:, 3] > 0))
            and geometry_collapsed(curve[:, :3]/curve[:, 3:]))


def source_boundary_fiber_seeds(first, second, axis, side, *,
                                max_cells=2000, max_results=128, atol=1e-3):
    """Discover exact fibers or explicitly numerical seeds on one source face.

    Inputs are the original positive-weight homogeneous nets. The exact path
    uses P=cW on every owner coefficient, then common-polynomial GCD/Sturm
    roots on each target boundary edge. The numerical path is a bounded point
    inversion proposal on the original target. Neither can discharge a face.
    """
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if (axis not in range(4) or side not in (0., 1.)
            or any(net.ndim != 3 or net.shape[-1] != 4
                   or not np.all(np.isfinite(net)) or np.any(net[..., 3] <= 0)
                   for net in (first, second))):
        return None
    if not source_boundary_fiber_candidate(first, second, axis, side):
        return None
    owner, target = (first, second) if axis < 2 else (second, first)
    curve = _edge(owner, axis % 2, side)
    result = dict(isolated=[], overlaps=[], parameter_fibers=[],
                  boundary_seed_proposals=[], cells_processed=0,
                  budget_exhausted=False, boundary_topology_complete=False,
                  truncation_cause='parameter_fiber')
    limit = max(0, int(max_cells))
    output_limit = max(0, int(max_results))

    class Denied(Exception):
        pass

    def tick(amount=1):
        amount = max(1, int(amount))
        if amount > limit-result['cells_processed']:
            raise Denied
        result['cells_processed'] += amount

    def append(key, entry):
        if len(result['parameter_fibers'])+len(result['boundary_seed_proposals']) >= output_limit:
            result['budget_exhausted'] = True
            result['truncation_cause'] = 'results'
            return False
        tick()
        result[key].append(entry)
        return True

    try:
        tick(max(1, curve.size))
        rows = [tuple(Fraction(float(x)) for x in row) for row in curve]
        point = tuple(x/rows[0][3] for x in rows[0][:3])
        exact_owner = all(all(row[k] == point[k]*row[3] for k in range(3)) for row in rows)
        exact_atol2 = Fraction(float(atol))**2
        if exact_owner:
            planar = _planar_point_preimage(point, target, tick)
            if planar is not None:
                uv, uv_box = planar
                tick(_point_work(target))
                representative = _point(target, tuple(Fraction(float(t)) for t in uv))
                xyz = np.array([float(x) for x in point])
                if (sum((a-b)**2 for a, b in zip(point, representative)) <= exact_atol2/16
                        and sum((Fraction(float(a))-b)**2 for a, b in zip(xyz, point)) <= exact_atol2/16):
                    append('parameter_fibers', dict(
                        t_range=(0., 1.), u=uv[0], v=uv[1], point=xyz,
                        certification='exact_source_parameter_fiber',
                        source_fiber_certificate=dict(
                            source_ids=(id(first), id(second)), axis=axis, side=side,
                            point=point, target_parameter_box=uv_box,
                            target_proof='exact_plane_convex_quad_point_inclusion')))
                else:
                    result['truncation_cause'] = 'resolution'
                return result
            seen = set()
            for target_axis in (0, 1):
                for target_side in (0., 1.):
                    edge = _edge(target, target_axis, target_side)
                    tick(3*coefficient_build_work(len(edge))+edge.size)
                    target_rows = [tuple(Fraction(float(x)) for x in row) for row in edge]
                    polynomials = [_power([row[k]-point[k]*row[3] for row in target_rows])
                                   for k in range(3)]
                    nonzero = [p for p in polynomials if any(p)]
                    if not nonzero:
                        polynomial, sequence, intervals = None, None, [(Fraction(1, 2),)*2]
                    else:
                        polynomial = nonzero[0]
                        for component in nonzero[1:]:
                            polynomial = _gcd(polynomial, component, tick)
                            if len(polynomial) == 1:
                                break
                        polynomial, _, sequence, intervals = isolate_root_intervals(polynomial, tick)
                    for interval in intervals:
                        while interval[0] != interval[1] and np.nextafter(float(interval[0]), np.inf) < float(interval[1]):
                            interval = _refine(polynomial, sequence, interval, tick)
                        t = float(sum(interval)/2)
                        uv = (target_side, t) if target_axis == 0 else (t, target_side)
                        if interval[0] == interval[1]:
                            exact_uv = ((Fraction(target_side), interval[0]) if target_axis == 0
                                        else (interval[0], Fraction(target_side)))
                            root_key = ('exact_parameter', exact_uv)
                        else:
                            root_key = ('edge_root', target_axis, target_side,
                                        tuple(polynomial), interval)
                        if root_key in seen:
                            continue
                        tick(_point_work(edge))
                        representative = _point(edge, (Fraction(t),))
                        xyz = np.array([float(x) for x in point])
                        if (sum((a-b)**2 for a, b in zip(point, representative)) > exact_atol2/16
                                or sum((Fraction(float(a))-b)**2 for a, b in zip(xyz, point)) > exact_atol2/16):
                            result['truncation_cause'] = 'resolution'
                            continue
                        entry = dict(t_range=(0., 1.), u=uv[0], v=uv[1], point=xyz,
                                     certification='exact_source_parameter_fiber',
                                     source_fiber_certificate=dict(
                                         source_ids=(id(first), id(second)), axis=axis, side=side,
                                         point=point, target_axis=target_axis, target_side=target_side,
                                         target_interval=interval,
                                         target_interval_open=interval[0] != interval[1],
                                         target_polynomial=None if polynomial is None else tuple(polynomial)))
                        if not nonzero:
                            entry['v_range' if target_axis == 0 else 'u_range'] = (0., 1.)
                        if not append('parameter_fibers', entry):
                            return result
                        seen.add(root_key)
            return result

        # The tiny edge extent is a proposal gate, never an exact identity.
        # A centered, rescaled target improves numerical inversion without
        # changing the original coefficients used to validate the proposal.
        tick(_point_work(curve)+target.size)
        query = _point(curve, (Fraction(1, 2),))
        query_float = np.array([float(x) for x in query])
        local = target/float(np.max(target[..., 3]))
        local = local.copy()
        local[..., :3] -= query_float*local[..., 3:]
        scale = float(np.max(np.abs(local[..., :3])))
        if not np.isfinite(scale):
            return result
        if scale > 0:
            local[..., :3] /= scale
        evaluation_work = max(1, (target.size+31)//32)
        best, distance = None, np.inf
        for u in np.linspace(0., 1., 5):
            for v in np.linspace(0., 1., 5):
                tick(evaluation_work)
                value = eval_surface(local, u, v, rational=True)
                distance2 = float(value@value)
                if distance2 < distance:
                    best, distance = np.array([u, v]), distance2
        if best is None:
            return result
        for _ in range(32):
            tick(3*evaluation_work)
            value, du, dv = eval_surface_d1(local, *best, rational=True)
            try:
                step = np.linalg.lstsq(np.column_stack((du, dv)), -value, rcond=None)[0]
            except np.linalg.LinAlgError:
                break
            updated = np.clip(best+step, 0., 1.)
            if np.array_equal(updated, best):
                break
            best = updated
        tick(_point_work(target)+curve.size)
        target_point = _point(target, tuple(Fraction(float(x)) for x in best))
        # The full source edge lies in its positive-weight Cartesian hull.
        # These are representation bounds only; they cannot prove a zero.
        owner_points = [tuple(row[k]/row[3] for k in range(3)) for row in rows]
        if (sum((a-b)**2 for a, b in zip(query, target_point)) <= exact_atol2/16
                and max(sum((a-b)**2 for a, b in zip(query, p)) for p in owner_points) <= exact_atol2/16
                and sum((Fraction(float(a))-b)**2 for a, b in zip(query_float, query)) <= exact_atol2/16):
            append('boundary_seed_proposals', dict(
                t_range=(0., 1.), u=float(best[0]), v=float(best[1]), point=query_float,
                certification='numerical_seed_proposal',
                owner_edge_exactly_collapsed=False))
        result['truncation_cause'] = 'resolution'
    except Denied:
        result['budget_exhausted'] = True
        result['truncation_cause'] = 'max_cells'
    except (OverflowError, ValueError, ZeroDivisionError):
        # Exact source membership can exist without a finite representable
        # Cartesian point or inverse. Optional seed fitting must not abort
        # the caller or turn that failure into an empty/complete census.
        result['truncation_cause'] = 'resolution'
    return result
