"""Representability checks when local roots enter a global knot domain.

An affine map is injective over the reals, but its float implementation need
not be. Preserve exact affine provenance until assembly, and report local
solutions explicitly when the public float parameter fields cannot hold them.
"""
from fractions import Fraction

import numpy as np
from mmcore.numeric.intersection._root_box_certificate import exact_bernstein_value


def _fraction(value):
    return Fraction.from_float(float(value))


def affine_parameters(parameters, bounds):
    exact = tuple(_fraction(a) + (_fraction(b)-_fraction(a))*_fraction(t)
                  for t, (a, b) in zip(parameters, bounds))
    return tuple(float(t) for t in exact), exact


def affine_enclosure(interval, bounds):
    """Outward float bounds for the exact affine image of an interval."""
    _, exact = affine_parameters(interval, (bounds, bounds))
    lower, upper = min(exact), max(exact)
    lo, hi = float(lower), float(upper)
    if _fraction(lo) > lower:
        lo = float(np.nextafter(lo, -np.inf))
    if _fraction(hi) < upper:
        hi = float(np.nextafter(hi, np.inf))
    return lo, hi


def _exact_evaluate(control, parameters, rational):
    work = exact_bernstein_value(control, parameters)
    if rational:
        return tuple(x/work[-1] for x in work[:-1])
    return tuple(work)


def mapping_issue(status, context, return_status, payload, reason):
    status['complete'] = False
    status['boundary_topology_complete'] = False
    status['partial_results'] += 1
    status.setdefault('unrepresentable_parameters', []).append(dict(
        context=context, reason=reason, **payload))
    if not return_status:
        raise RuntimeError(f"{context}: global parameter representation {reason}; "
                           "pass return_status=True to receive local solutions")


def map_isolated(entry, keys, bounds, sources, rational, atol,
                 status, context, return_status):
    """Map one root; ``sources`` contains (control net, parameter indices).

    Geometry is checked using exact evaluation of the source binary control
    net at the round-tripped global floats. This is a representation check,
    separate from the underlying solver's root existence certificate.
    """
    local = tuple(float(entry[k]) for k in keys)
    mapped, exact = affine_parameters(local, bounds)
    payload = dict(local_parameters=local, parameter_bounds=tuple(bounds),
                   exact_global_parameters=tuple(str(t) for t in exact),
                   local_result=dict(entry))
    restored = tuple((_fraction(t)-_fraction(a))/(_fraction(b)-_fraction(a))
                     for t, (a, b) in zip(mapped, bounds))
    if any(t != _fraction(x) for t, x in zip(restored, local)):
        try:
            point = tuple(_fraction(x) for x in entry['point'])
            limit = _fraction(atol)**2
            values = [_exact_evaluate(control, tuple(restored[i] for i in axes), rational)
                      for control, axes in sources]
            valid = all(0 <= t <= 1 for t in restored) and all(
                sum((a-b)**2 for a, b in zip(point, value)) <= limit
                for value in values)
            valid &= all(sum((a-b)**2 for a, b in zip(left, right)) <= limit
                         for index, left in enumerate(values) for right in values[index+1:])
        except (OverflowError, ValueError, ZeroDivisionError):
            valid = False
        if not valid:
            mapping_issue(status, context, return_status, payload,
                          'rounded parameters do not represent the local solution')
            return None
    result = {k: entry[k] for k in ('point', 'certification', 'd_min') if k in entry}
    result.update(zip(keys, mapped))
    certificate = {k: v for k, v in entry.items()
                   if k not in keys+('point', 'certification', 'd_min')}
    if certificate:
        result['local_root_certificate'] = dict(
            parameter_bounds=tuple(bounds), **certificate)
    result['_exact_global_parameters'] = exact
    result['_local_parameter_payload'] = payload
    return result


def reject_parameter_aliases(entries, keys, status, context, return_status,
                             identity_keys=()):
    """Never merge different local solutions because global floats collide."""
    groups = {}
    for entry in entries:
        key = tuple(entry[k] for k in identity_keys+keys)
        groups.setdefault(key, []).append(entry)
    result = []
    for group in groups.values():
        # A tolerance contact represents a distance component, not an exact
        # parameter root. Its independent span minimizers may differ by one
        # ulp and remain eligible for the public contact clustering policy.
        identities = {e.get('_exact_global_parameters', tuple(e[k] for k in keys))
                      for e in group if e.get('certification', 'exact') != 'tolerance'}
        if len(identities) > 1:
            mapping_issue(status, context, return_status,
                          dict(local_solutions=[e['_local_parameter_payload'] for e in group]),
                          'distinct local solutions collapse to the same global parameters')
        else:
            result.extend(group)
    return result


def map_overlap(entry, keys, bounds, sources, rational, atol,
                status, context, return_status, enclosure_keys=()):
    """Map paired overlap endpoints, or explicitly designated enclosures."""
    ranges = tuple(tuple(entry.get(k+'_range', (0., 1.))) for k in keys)
    mapped_ranges, exact_ranges = [], []
    payload = dict(local_result=dict(entry), parameter_bounds=tuple(bounds))
    for key, interval, bound in zip(keys, ranges, bounds):
        mapped, exact = affine_parameters(interval, (bound, bound))
        if interval[0] != interval[1] and mapped[0] == mapped[1]:
            mapping_issue(status, context, return_status, payload,
                          'nonzero overlap range collapses to one global parameter')
            return None
        mapped_ranges.append(affine_enclosure(interval, bound)
                             if key in enclosure_keys else mapped)
        exact_ranges.append(exact)
    # Enclosure corners need not lie on the intersection. Validate only
    # sources whose complete parameter tuple is actually paired.
    paired = [(control, axes) for control, axes in sources
              if all(keys[i] not in enclosure_keys for i in axes)]
    for endpoint in (0, 1):
        local = tuple(interval[endpoint] for interval in ranges)
        restored = tuple((_fraction(interval[endpoint])-_fraction(a))/(_fraction(b)-_fraction(a))
                         for interval, (a, b) in zip(mapped_ranges, bounds))
        if all(t == _fraction(x) for t, x in zip(restored, local)):
            continue
        try:
            limit = _fraction(atol)**2
            valid = True
            values = []
            for control, axes in paired:
                original = _exact_evaluate(control, tuple(_fraction(local[i]) for i in axes), rational)
                represented = _exact_evaluate(control, tuple(restored[i] for i in axes), rational)
                values.append(represented)
                valid &= (all(0 <= restored[i] <= 1 for i in axes)
                          and sum((a-b)**2 for a, b in zip(original, represented)) <= limit)
            valid &= all(sum((a-b)**2 for a, b in zip(left, right)) <= limit
                         for index, left in enumerate(values) for right in values[index+1:])
        except (OverflowError, ValueError, ZeroDivisionError):
            valid = False
        if not valid:
            mapping_issue(status, context, return_status, payload,
                          'rounded overlap endpoint does not represent the local solution')
            return None
    mapped = dict(zip((k+'_range' for k in keys), mapped_ranges))
    mapped['_exact_global_ranges'] = tuple(exact_ranges)
    mapped['local_overlap_certificate'] = payload
    return mapped


def strip_mapping_metadata(entries):
    for entry in entries:
        entry.pop('_exact_global_parameters', None)
        entry.pop('_local_parameter_payload', None)
    return entries
