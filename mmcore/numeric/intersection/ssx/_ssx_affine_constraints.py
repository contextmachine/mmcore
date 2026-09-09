"""Exact necessary parameter relations from certified source coordinates.

A source coordinate ``offset+slope*parameter`` gives an exact relation
between one parameter on each surface. These constraints only describe
where intersections may occur: a contracted box is not an existence proof
and need not enclose the residual or its derivatives away from the zero set.
"""
from fractions import Fraction

import numpy as np

from mmcore.numeric.intersection.csx._planar_roots import _outward_interval


def _fraction(value):
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value))
    return Fraction.from_float(float(value))


def augment_constant_coordinates(sources, affine_maps, *, charge=None):
    """Add exact P_k=c*W identities to a constraints-only map copy.

    A constant source coordinate can fix a parameter of the opposite
    chart. Zero slopes must not enter the original maps used by inverse
    and immersion predicates, which require a nonzero chart derivative.
    Return None when the prepaid source algebra is denied.
    """
    result = tuple(dict(mapping) for mapping in affine_maps)
    for source,mapping in zip(sources,result):
        source = np.asarray(source)
        if (source.ndim != 3 or source.shape[-1] != 4
                or not np.all(np.isfinite(source)) or np.any(source[...,3] <= 0)):
            continue
        coordinates = [coordinate for coordinate in range(3)
                       if not any((coordinate,axis) in mapping for axis in (0,1))]
        if not coordinates:
            continue
        count = source.shape[0]*source.shape[1]
        # Convert W once; each tested coordinate converts P, divides one
        # anchor, and checks every exact product/equality against c*W.
        operations = count+len(coordinates)*(3*count+1)
        if charge is not None and not charge(max(1,(operations+127)//128)):
            return None
        weights = tuple(_fraction(value) for value in source[...,3].flat)
        for coordinate in coordinates:
            values = tuple(_fraction(value) for value in source[...,coordinate].flat)
            constant = values[0]/weights[0]
            if all(value == constant*weight for value,weight in zip(values,weights)):
                mapping[coordinate,0] = constant,Fraction(0)
    return result


def augment_projected_coordinates(sources, affine_maps, *, charge=None):
    """Find exact common affine coordinates after a world-space projection.

    For each pair of source axes, a row nullspace contains precisely the
    vectors q for which both projected control nets q.P are affine in the
    respective single parameter. Exact Bernstein coefficient identities
    prove each relation globally. The inputs must have uniform positive
    weights; nonuniform rational charts retain their existing maps.

    New integer coordinate keys are constraints-only and do not modify the
    pure world-coordinate maps used by inverse or immersion predicates.
    Denied exact work returns None, never a truncated set of new proofs.
    """
    result = tuple(dict(mapping) for mapping in affine_maps)
    if len(sources) != 2 or len(result) != 2:
        raise ValueError('Expected two source surfaces and coordinate maps')
    nets = tuple(np.asarray(source) for source in sources)
    if any(net.ndim != 3 or net.shape[-1] != 4
           or min(net.shape[:2]) < 2 or not np.all(np.isfinite(net))
           or np.any(net[...,3] <= 0.) or not np.all(net[...,3] == net[0,0,3])
           for net in nets):
        return result

    class Stopped(Exception):
        pass

    def spend(amount):
        if charge is not None and not charge(max(1,int(amount))):
            raise Stopped

    def nullspace(rows):
        # Incremental exact echelon form has at most three pivot rows.
        # Rank three rejects this axis pair immediately; no numeric SVD
        # or scale-dependent rank threshold contributes to the proof.
        basis = {}
        for source_row in rows:
            spend(3)
            row = list(source_row)
            for pivot in sorted(basis):
                factor = row[pivot]
                if factor:
                    spend(2*(3-pivot))
                    for k in range(pivot,3):
                        row[k] -= factor*basis[pivot][k]
            pivot = next((k for k,value in enumerate(row) if value),None)
            if pivot is None:
                continue
            spend(3-pivot)
            divisor = row[pivot]
            row[pivot:] = [value/divisor for value in row[pivot:]]
            basis[pivot] = tuple(row)
            if len(basis) == 3:
                return []
        vectors = []
        for free in (k for k in range(3) if k not in basis):
            vector = [Fraction(0)]*3
            vector[free] = Fraction(1)
            for pivot in sorted(basis,reverse=True):
                spend(2*(2-pivot)+1)
                vector[pivot] = -sum(basis[pivot][k]*vector[k]
                                     for k in range(pivot+1,3))
            spend(3)
            anchor = next(value for value in vector if value)
            vectors.append(tuple(value/anchor for value in vector))
        return vectors

    try:
        points, residuals, ends = [], [], []
        for net in nets:
            # Original source conversion and exact Cartesian division.
            spend(1+6*net.shape[0]*net.shape[1])
            weight = _fraction(net[0,0,3])
            values = np.asarray([_fraction(value)/weight for value in net[...,:3].flat],
                                dtype=object).reshape(net.shape[:2]+(3,))
            points.append(values)
            by_axis, endpoint_axes = [], []
            for axis in (0,1):
                spend(3)
                endpoint = values[-1,0] if axis == 0 else values[0,-1]
                delta = tuple(b-a for a,b in zip(values[0,0],endpoint))
                endpoint_axes.append(delta)
                rows = []
                for i in range(net.shape[0]):
                    for j in range(net.shape[1]):
                        spend(10)
                        parameter = Fraction((i,j)[axis],net.shape[axis]-1)
                        row = tuple(value-origin-parameter*difference
                                    for value,origin,difference in zip(values[i,j],values[0,0],delta))
                        if any(row):
                            rows.append(row)
                by_axis.append(rows)
            residuals.append(by_axis)
            ends.append(endpoint_axes)
        keys = [coordinate for mapping in result for coordinate,axis in mapping]
        next_key = max([2]+keys)+1
        projection_keys = {}
        for first_axis in (0,1):
            for second_axis in (0,1):
                for projection in nullspace(residuals[0][first_axis]+residuals[1][second_axis]):
                    if projection not in projection_keys:
                        projection_keys[projection] = next_key
                        next_key += 1
                    key = projection_keys[projection]
                    for owner,axis in enumerate((first_axis,second_axis)):
                        spend(12)
                        offset = sum(a*b for a,b in zip(projection,points[owner][0,0]))
                        slope = sum(a*b for a,b in zip(projection,ends[owner][axis]))
                        result[owner][key,axis] = offset,slope
    except Stopped:
        return None
    return result


class AffineParameterConstraints:
    """Cached exact contraction of four global parameter intervals.

    ``affine_maps`` contains the two dictionaries returned by
    ``_pure_affine_coordinates``: keys are (coordinate_key, local_axis),
    values are exact (offset, slope). The caller owns that identity proof.
    Keys may also identify exact common world projections proven by the
    constraints-only augmentation; the same key must mean the same q.P.

    ``contract(box)`` returns a necessary box containing every root in the
    input box. ``None`` means the affine constraints prove it empty.
    Insufficient work returns the input box unchanged and sets ``exhausted``;
    denied work must never masquerade as an empty-set certificate.
    """
    def __init__(self, affine_maps, *, charge=None):
        self.charge = charge
        self.exhausted = False
        self.empty = False
        self.identity = False
        self.components = None
        self.cache = {}
        if len(affine_maps) != 2:
            raise ValueError('Expected two certified source coordinate maps')
        relations = []
        coordinates = sorted({coordinate for mapping in affine_maps
                              for coordinate,axis in mapping})
        for coordinate in coordinates:
            for first_axis in (0, 1):
                a = affine_maps[0].get((coordinate, first_axis))
                if a is None:
                    continue
                for second_axis in (0, 1):
                    b = affine_maps[1].get((coordinate, second_axis))
                    if b is not None:
                        relations.append((first_axis, 2+second_axis, a, b))
        if not self._spend(max(1, 4+4*len(relations))):
            return
        edges = [[] for _ in range(4)]
        fixed_variables = []
        for x, y, left, right in relations:
            a, b = map(_fraction, left)
            c, d = map(_fraction, right)
            if not b and not d:
                if a != c:
                    self.empty = True
                    return
            elif not b:
                fixed_variables.append((y, (a-c)/d))
            elif not d:
                fixed_variables.append((x, (c-a)/b))
            else:
                alpha, beta = (a-c)/d, b/d
                edges[x].append((y, alpha, beta))
                edges[y].append((x, -alpha/beta, 1/beta))
        components = []
        visited = set()
        for root in range(4):
            if root in visited:
                continue
            expressions = {root: (Fraction(0), Fraction(1))}
            stack = [root]
            fixed_root = None
            while stack:
                current = stack.pop()
                visited.add(current)
                alpha, beta = expressions[current]
                for target, shift, scale in edges[current]:
                    proposed = shift+scale*alpha, scale*beta
                    if target not in expressions:
                        expressions[target] = proposed
                        stack.append(target)
                        continue
                    prior_alpha, prior_beta = expressions[target]
                    delta = prior_beta-proposed[1]
                    if not delta:
                        if prior_alpha != proposed[0]:
                            self.empty = True
                            return
                    else:
                        value = (proposed[0]-prior_alpha)/delta
                        if fixed_root is not None and fixed_root != value:
                            self.empty = True
                            return
                        fixed_root = value
            for variable, value in fixed_variables:
                if variable in expressions:
                    alpha, beta = expressions[variable]
                    value = (value-alpha)/beta
                    if fixed_root is not None and fixed_root != value:
                        self.empty = True
                        return
                    fixed_root = value
            components.append((tuple((variable, *values) for variable, values
                                     in sorted(expressions.items())), fixed_root))
        self.components = tuple(components)
        self.identity = all(len(expressions) == 1 and fixed_root is None
                            for expressions, fixed_root in self.components)

    def _spend(self, amount=1):
        if self.exhausted:
            return False
        if self.charge is None or self.charge(int(amount)):
            return True
        self.exhausted = True
        return False

    def contract(self, box):
        key = tuple(tuple(float(value) for value in pair) for pair in box)
        if len(key) != 4 or any(len(pair) != 2 or not np.all(np.isfinite(pair)) for pair in key):
            raise ValueError('Expected four finite parameter intervals')
        if any(lo > hi for lo, hi in key) or self.empty:
            return None
        if self.identity:
            # The paid constructor found no relation among parameters and
            # no fixed coordinate. Every box is unchanged, so there is no
            # interval algebra to allocate, cache, or pay for on each call.
            return key
        if key in self.cache:
            return self.cache[key]
        if self.components is None or not self._spend(4+len(self.components)):
            return key
        exact_box = tuple(tuple(_fraction(value) for value in pair) for pair in key)
        output = list(key)
        for expressions, fixed_root in self.components:
            root_low = root_high = fixed_root
            for variable, alpha, beta in expressions:
                lo, hi = exact_box[variable]
                interval = sorted(((lo-alpha)/beta, (hi-alpha)/beta))
                root_low = interval[0] if root_low is None else max(root_low, interval[0])
                root_high = interval[1] if root_high is None else min(root_high, interval[1])
            if root_low > root_high:
                self.cache[key] = None
                return None
            for variable, alpha, beta in expressions:
                exact = sorted((alpha+beta*root_low, alpha+beta*root_high))
                lo, hi = _outward_interval(exact)
                output[variable] = max(key[variable][0], lo), min(key[variable][1], hi)
        self.cache[key] = tuple(output)
        return self.cache[key]

    def canonical_candidate(self, point):
        """Propose a NEW exactly affine-consistent floating parameter tuple.

        This is not root identity or existence. One lowest-axis coordinate
        per free component supplies a proposed root parameter; the caller
        must independently validate the resulting source residual and its
        intended domain. Nonrepresentable exact images return None.
        """
        if self.empty or self.components is None or self.exhausted:
            return None
        if not self._spend(4+len(self.components)):
            return None
        try:
            values = tuple(_fraction(value) for value in point)
        except (ValueError, OverflowError):
            return None
        if len(values) != 4:
            raise ValueError('Expected four candidate parameters')
        result = np.empty(4)
        for expressions, fixed_root in self.components:
            variable, alpha, beta = expressions[0]
            root = ((values[variable]-alpha)/beta if fixed_root is None else fixed_root)
            for variable, alpha, beta in expressions:
                exact = alpha+beta*root
                try:
                    rounded = float(exact)
                except OverflowError:
                    return None
                if not np.isfinite(rounded) or _fraction(rounded) != exact:
                    return None
                result[variable] = rounded
        return result

    def face_key(self, axis, value):
        """Canonical exact key for an equivalent affine slice of the zero set.

        None means no paid key is available. ('empty',) means the source
        constraints themselves rule out that face. Keep the original
        requested axis/value when computing a missing source census:
        rounding the canonical Fraction value would define another face.
        """
        if self.exhausted or not self._spend():
            return None
        if self.empty:
            return ('empty',)
        if self.components is None:
            return None
        if axis not in range(4):
            raise ValueError('Expected a global parameter axis')
        exact_value = _fraction(value)
        for component, (expressions, fixed_root) in enumerate(self.components):
            for variable, alpha, beta in expressions:
                if variable == axis:
                    root = (exact_value-alpha)/beta
                    if fixed_root is not None and root != fixed_root:
                        return ('empty',)
                    return component, root
        raise ValueError('Missing affine component')

    def rebind_source_certificate(self, certificate, axis, value):
        """Describe one proved source root on an exactly equivalent face.

        The certificate must come from the SAME source pair as the affine
        maps. This returns only new exact metadata. A caller reusing an
        entire floating root record must separately require that its
        published coordinate already equals the new pin, or revalidate a
        changed representative against both original sources.
        """
        if certificate.get('kind') != 'exact_source_planar_cut':
            return None
        def decode(pair):
            return Fraction(int(pair[0]), int(pair[1]))
        old_key = self.face_key(certificate['axis'], decode(certificate['cut']))
        new_key = self.face_key(axis, value)
        if old_key is None or old_key == ('empty',) or old_key != new_key:
            return None
        exact_value = _fraction(value)
        box = [tuple(decode(endpoint) for endpoint in pair)
               for pair in certificate['parameter_root_box']]
        # A valid root enclosure must contain the implied coordinate.
        if not box[axis][0] <= exact_value <= box[axis][1]:
            return None
        box[axis] = (exact_value, exact_value)
        pins = {key: decode(pair) for key, pair in certificate['pinned']}
        if axis in pins and pins[axis] != exact_value:
            return None
        pins[axis] = exact_value
        pair = lambda number: (str(number.numerator), str(number.denominator))
        rebound = dict(certificate)
        rebound.update(axis=axis, cut=pair(exact_value),
                       pinned=tuple((key,pair(number)) for key,number in sorted(pins.items())),
                       parameter_root_box=tuple(tuple(pair(number) for number in interval)
                                                for interval in box))
        return rebound
