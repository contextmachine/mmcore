"""Original-source cut census for an affine plane and a Bezier chart.

Restriction and the univariate polynomial use exact rational arithmetic on
original binary controls. A returned Sturm certificate therefore describes
that source surface, even if a floating restriction would change root count.
Unsupported charts return ``None``. A curve identically in the plane
returns an explicit unresolved source face, never an isolated-root search.
"""
from fractions import Fraction

import numpy as np

from mmcore.numeric.intersection._exact_univariate import coefficient_build_work, _power
from mmcore.numeric.intersection.csx._planar_overlap import _convex_chart, _sub, _dot
from mmcore.numeric.intersection.csx._planar_roots import (
    _exact_planar_bilinear_roots_data, _outward_interval)
from mmcore.numeric.intersection.ssx._ssx_root_identity import _pure_affine_coordinates


def _fraction(value):
    return value if isinstance(value, Fraction) else Fraction.from_float(float(value))


def _split(points, value):
    rows = [list(points)]
    while len(rows[-1]) > 1:
        rows.append([tuple((1-value)*a+value*b for a, b in zip(left, right))
                     for left, right in zip(rows[-1], rows[-1][1:])])
    return [row[0] for row in rows], [row[-1] for row in rows[::-1]]


def _restrict(points, lo, hi):
    if hi != 1:
        points, _ = _split(points, hi)
    if lo:
        _, points = _split(points, lo/hi)
    return points


def _isocurve(surface, axis, value):
    array = surface if axis == 0 else list(map(list, zip(*surface)))
    return [_split([row[j] for row in array], value)[0][-1]
            for j in range(len(array[0]))]


def _surface_rectangle(surface, box):
    columns = [_restrict(list(column), *box[0]) for column in zip(*surface)]
    return [_restrict(list(row), *box[1]) for row in zip(*columns)]


def _cartesian(points):
    weights = [point[3] for point in points]
    return [tuple(x/w for x in point[:3]) for point, w in zip(points, weights)], weights


def _surface_point(surface, u, v):
    homogeneous = _split(_isocurve(surface, 0, u), v)[0][-1]
    return tuple(value/homogeneous[3] for value in homogeneous[:3])


def _partial(cause, cells=0, **extra):
    return dict(isolated=[], overlaps=[], parameter_fibers=[], cells_processed=cells,
                budget_exhausted=True, boundary_topology_complete=False,
                truncation_cause=cause, **extra)


def _pair(value):
    return str(value.numerator), str(value.denominator)


def supported_source_pair(first_h, second_h):
    """Cheap source-level applicability predicate; cache for immutable controls.

    It proves that at least one original surface is an injective uniform-
    weight affine plane. Individual cuts can still be unsupported (for
    example a plane-owner cut without a matching affine graph coordinate).
    """
    for source in (first_h, second_h):
        source = np.asarray(source, dtype=float)
        if (source.shape != (2, 2, 4) or not np.all(np.isfinite(source))
                or source[0, 0, 3] <= 0
                or not np.all(source[..., 3] == source[0, 0, 3])):
            continue
        points, _ = _cartesian([tuple(_fraction(value) for value in point)
                                for row in source for point in row])
        a, c, b, opposite = points
        if (not any(opposite[k]+a[k]-b[k]-c[k] for k in range(3))
                and _convex_chart(points) is not None):
            return True
    return False


def source_cut_construction_work(first_h, second_h):
    """Prepaid exact-coefficient setup bound, also billable on unsupported cuts."""
    sources = tuple(np.asarray(source) for source in (first_h, second_h))
    candidates = [index for index, source in enumerate(sources)
                  if source.shape == (2, 2, 4)
                  and np.all(source[..., 3] == source[0, 0, 3])]
    return (sum(int(np.prod(source.shape[:2])) for source in sources)
            + max((sum(coefficient_build_work(sources[1-index].shape[k])
                       * sources[1-index].shape[1-k] for k in (0, 1))
                   for index in candidates), default=0))


def exact_source_planar_cut(first_h, second_h, axis, cut, box, *,
                            max_cells=100_000, max_results=4096, atol=1e-3):
    """Enumerate an original-source four-parameter face exactly.

    Inputs are homogeneous controls and global unit-square parameter bounds.
    Each root has ``stuv``, a global ``parameter_root_box``, and a
    ``source_cut_certificate``. The certificate's scalar polynomial is in
    the original varying parameter, not a rounded restricted parameter.
    Its closed interval contains exactly one distinct root; exact affine
    inversion supplies the remaining coordinates. ``pinned`` retains all
    exact fixed-coordinate relations, including reverse plane-owner cuts.

    Construction and Sturm work consume ``max_cells`` before execution.
    ``None`` means the exact planar reduction does not apply, not emptiness.
    """
    sources = tuple(np.asarray(source, dtype=float) for source in (first_h, second_h))
    if (axis not in range(4) or len(box) != 4
            or any(source.ndim != 3 or source.shape[-1] != 4
                   or not np.all(np.isfinite(source)) or not np.all(source[..., 3] > 0)
                   for source in sources)):
        return None
    try:
        bounds = tuple(tuple(_fraction(value) for value in pair) for pair in box)
        pin = _fraction(cut)
    except (ValueError, OverflowError):
        return None
    if (any(len(pair) != 2 or not 0 <= pair[0] < pair[1] <= 1 for pair in bounds)
            or not bounds[axis][0] <= pin <= bounds[axis][1]):
        return None
    owner = axis//2
    # Prefer the opposite chart as plane. The reverse route pins one
    # graph parameter through a proven pure-affine world coordinate.
    possible_planes = [1-owner, owner]
    possible_planes = [index for index in possible_planes
                       if sources[index].shape == (2, 2, 4)
                       and np.all(sources[index][..., 3] == sources[index][0, 0, 3])]
    if not possible_planes:
        return None
    construction_work = source_cut_construction_work(first_h, second_h)
    if construction_work > max_cells:
        return _partial('preflight', required_coefficient_work=construction_work)
    exact_sources = tuple([[tuple(_fraction(value) for value in point) for point in row]
                           for row in source] for source in sources)
    plane_index = None
    for index in possible_planes:
        plane = exact_sources[index]
        plane_points, _ = _cartesian([p for row in plane for p in row])
        a, c, b, opposite = plane_points
        if (not any(opposite[k]+a[k]-b[k]-c[k] for k in range(3))
                and _convex_chart(plane_points) is not None):
            plane_index, original_plane_points = index, plane_points
            break
    if plane_index is None:
        return None
    graph_index = 1-plane_index
    plane, graph = exact_sources[plane_index], exact_sources[graph_index]
    pinned = {axis: pin}
    if owner == graph_index:
        graph_axis, graph_pin = axis % 2, pin
    else:
        graph_axis, graph_pin = None, None
        plane_affine, graph_affine = (_pure_affine_coordinates(source) for source in
                                      (sources[plane_index], sources[graph_index]))
        for coordinate in range(3):
            relation = plane_affine.get((coordinate, axis % 2))
            if relation is None:
                continue
            for candidate_axis in (0, 1):
                inverse = graph_affine.get((coordinate, candidate_axis))
                if inverse is not None:
                    graph_axis = candidate_axis
                    graph_pin = (relation[0]+relation[1]*pin-inverse[0])/inverse[1]
                    break
            if graph_axis is not None:
                break
        if graph_axis is None:
            return None
        pinned[2*graph_index+graph_axis] = graph_pin
    graph_global_axis = 2*graph_index+graph_axis
    if not bounds[graph_global_axis][0] <= graph_pin <= bounds[graph_global_axis][1]:
        return dict(isolated=[], overlaps=[], parameter_fibers=[], cells_processed=construction_work,
                    budget_exhausted=False, boundary_topology_complete=True)
    varying_axis = 2*graph_index+1-graph_axis
    global_curve = _isocurve(graph, graph_axis, graph_pin)
    curve = _restrict(global_curve, *bounds[varying_axis])
    target_bounds = bounds[2*plane_index:2*plane_index+2]
    target = _surface_rectangle(plane, target_bounds)
    points, weights = _cartesian(curve)
    target_points, target_weights = _cartesian([point for row in target for point in row])
    origin, normal, *_ = _convex_chart(original_plane_points)
    if not any(w*_dot(normal, _sub(p, origin)) for p, w in zip(points, weights)):
        # The scalar source equation vanishes identically. A 3D isolated
        # root solver cannot enumerate this parameter curve; sending it
        # there only spends its allowance on an infinite family of roots.
        # Convex trimming/stratum ownership is a separate obligation. The
        # complete closed source face safely covers that remaining work,
        # including possible empty pieces, without assuming an overlap.
        face_box = list(bounds)
        face_box[axis] = pin, pin
        return _partial('resolution', construction_work, unresolved_source_boxes=[{
            'parameter_root_box': tuple(_outward_interval(pair) for pair in face_box),
            'reason': 'positive_dimensional_cut'}])
    result = _exact_planar_bilinear_roots_data(
        points, weights, target_points, target_weights, max_cells=max_cells,
        max_results=max_results, atol=atol, construction_work=construction_work)
    if result is None:
        return None
    if not result['isolated']:
        if result.pop('unresolved_parameter_boxes', None):
            result['unresolved_source_boxes'] = [{
                'parameter_root_box': tuple(_outward_interval(pair) for pair in bounds),
                'reason': 'local_parameter_representation'}]
        return result
    global_polynomial_work = coefficient_build_work(len(global_curve))
    if result['cells_processed']+global_polynomial_work > max_cells:
        result['isolated'] = []
        result.update(budget_exhausted=True, boundary_topology_complete=False,
                      truncation_cause='max_cells', unresolved_source_boxes=[{
                          'parameter_root_box': tuple(_outward_interval(pair) for pair in bounds),
                          'reason': 'source_certificate_work'}])
        return result
    result['cells_processed'] += global_polynomial_work
    global_points, global_weights = _cartesian(global_curve)
    origin, normal, *_ = _convex_chart(original_plane_points)
    polynomial = _power([w*_dot(normal, _sub(p, origin)) for p, w in zip(global_points, global_weights)])
    # Monic normalization makes the same source face independent of target
    # rectangle scaling and isolates the same root across adjacent cells.
    polynomial = tuple(value/polynomial[-1] for value in polynomial)
    parameter_axes = (varying_axis, 2*plane_index, 2*plane_index+1)
    mapped_roots = []
    representation_work = 16+sum(
        source.shape[1]*coefficient_build_work(source.shape[0])
        + coefficient_build_work(source.shape[1]) for source in sources)
    for root in result['isolated']:
        if result['cells_processed']+representation_work > max_cells:
            result.update(budget_exhausted=True, boundary_topology_complete=False,
                          truncation_cause='max_cells')
            result.setdefault('unresolved_source_boxes', []).append({
                'parameter_root_box': tuple(_outward_interval(pair) for pair in bounds),
                'reason': 'source_representation_work'})
            break
        result['cells_processed'] += representation_work
        exact_box = [None]*4
        stuv = [None]*4
        for fixed_axis, fixed_value in pinned.items():
            exact_box[fixed_axis] = (fixed_value, fixed_value)
            stuv[fixed_axis] = float(fixed_value)
        for local_axis, global_axis in enumerate(parameter_axes):
            if global_axis in pinned:
                continue
            lo, hi = bounds[global_axis]
            interval = root['exact_parameter_root_box'][local_axis]
            exact_box[global_axis] = tuple(lo+(hi-lo)*value for value in interval)
            local_value = root[('t', 'u', 'v')[local_axis]]
            stuv[global_axis] = float(lo+(hi-lo)*_fraction(local_value))
        exact_parameter = tuple(_fraction(value) for value in stuv)
        first_point = _surface_point(exact_sources[0], *exact_parameter[:2])
        second_point = _surface_point(exact_sources[1], *exact_parameter[2:])
        try:
            point = np.array(first_point, dtype=float)
            rounded_point = tuple(_fraction(value) for value in point)
            distance = max(sum((x-y)**2 for x, y in zip(first_point, second_point)),
                           sum((x-y)**2 for x, y in zip(rounded_point, first_point)),
                           sum((x-y)**2 for x, y in zip(rounded_point, second_point)))
        except (OverflowError, ValueError):
            distance = None
        global_interval = exact_box[varying_axis]
        certificate = {
            'kind': 'exact_source_planar_cut',
            'source_ids': (id(first_h), id(second_h)),
            'axis': axis, 'cut': _pair(pin), 'varying_axis': varying_axis,
            'pinned': tuple((key, _pair(value)) for key, value in sorted(pinned.items())),
            'polynomial': tuple(_pair(value) for value in polynomial),
            'interval': tuple(_pair(value) for value in global_interval),
            'parameter_root_box': tuple(tuple(_pair(value) for value in pair) for pair in exact_box),
        }
        global_box = tuple(_outward_interval(pair) for pair in exact_box)
        if (distance is None or distance > _fraction(atol)**2
                or any(tuple(existing['stuv']) == tuple(stuv) for existing in mapped_roots)):
            result.update(budget_exhausted=True, boundary_topology_complete=False,
                          truncation_cause='resolution')
            result.setdefault('unresolved_source_boxes', []).append({
                'parameter_root_box': global_box, 'reason': 'parameter_representation',
                'source_cut_certificate': certificate})
            continue
        entry = dict(stuv=np.array(stuv), point=point, parameter_root_box=global_box,
                     parameter_root_certification='exact_source_sturm_isolation',
                     source_cut_certificate=certificate)
        if 'root_multiplicity' in root:
            entry['root_multiplicity'] = root['root_multiplicity']
        mapped_roots.append(entry)
    result['isolated'] = mapped_roots
    # A local unresolved interval is not a complete global proof region;
    # retain the whole queried source box conservatively in that case.
    if result.pop('unresolved_parameter_boxes', None):
        result.setdefault('unresolved_source_boxes', []).append({
            'parameter_root_box': tuple(_outward_interval(pair) for pair in bounds),
            'reason': 'local_parameter_representation'})
    return result
