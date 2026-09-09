"""Common-face isolation for independently discovered boundary events."""
from fractions import Fraction

import numpy as np

from mmcore.numeric.bern import bernstein_partial_derivative_coeffs, de_casteljau_split_nd
from mmcore.numeric._bezier_common import restrict_net_axis_v
from mmcore.numeric.intersection._root_box_certificate import (
    residual_roundoff_bound, jacobian_is_injective)
from mmcore.numeric.intersection.ssx._ssx5_singular import psi_vector_net


def _pure_affine_coordinates(surface):
    """Exact world coordinates depending affinely on one chart parameter."""
    weights = surface[..., 3]
    if not np.all(weights > 0):
        return {}
    if not np.all(weights == weights.flat[0]):
        # Verify P_coordinate=(a+b*t_axis)*W in Bernstein form. Pointwise
        # dehomogenized coefficients do not prove a rational identity;
        # elevate P/W one degree and multiply W by the affine polynomial.
        weight_rows = [[Fraction(float(x)) for x in row] for row in weights]
        result = {}
        for coordinate in range(3):
            points = [[Fraction(float(x)) for x in row]
                      for row in surface[...,coordinate]]
            for axis in (0,1):
                degree = surface.shape[axis]-1
                if not degree:
                    continue
                def coefficient(rows, i, j):
                    return rows[i][j] if axis == 0 else rows[j][i]
                start = coefficient(points,0,0)/coefficient(weight_rows,0,0)
                stop = coefficient(points,degree,0)/coefficient(weight_rows,degree,0)
                slope = stop-start
                if not slope:
                    continue
                valid = True
                for j in range(surface.shape[1-axis]):
                    for i in range(degree+2):
                        alpha = Fraction(i,degree+1)
                        pl = coefficient(points,i-1,j) if i else 0
                        pr = coefficient(points,i,j) if i <= degree else 0
                        wl = coefficient(weight_rows,i-1,j) if i else 0
                        wr = coefficient(weight_rows,i,j) if i <= degree else 0
                        if alpha*pl+(1-alpha)*pr != start*(alpha*wl+(1-alpha)*wr)+slope*alpha*wl:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    result[coordinate,axis] = start,slope
        return result
    weight = Fraction.from_float(float(weights.flat[0]))
    result = {}
    for coordinate in range(3):
        values = [[Fraction.from_float(float(v))/weight for v in row]
                  for row in surface[..., coordinate]]
        for axis in (0, 1):
            degree = surface.shape[axis]-1
            if not degree:
                continue
            start = values[0][0]
            end = values[-1][0] if axis == 0 else values[0][-1]
            slope = end-start
            if not slope:
                continue
            if all(values[i][j] == start+slope*Fraction((i, j)[axis], degree)
                   for i in range(len(values)) for j in range(len(values[0]))):
                result[coordinate, axis] = (start, slope)
    return result


def _restrict(net, box):
    for axis, (lo, hi) in enumerate(box):
        net = restrict_net_axis_v(net, axis, lo, hi, 0., 1.)
    return net


def _affine_interval(lo, hi, lower, upper):
    """Enclose the exact affine image of binary floating point endpoints."""
    start, stop = Fraction.from_float(float(lo)), Fraction.from_float(float(hi))
    result = []
    for endpoint, direction in ((lower, -np.inf), (upper, np.inf)):
        exact = start + (stop-start)*Fraction.from_float(float(endpoint))
        rounded = float(exact)
        if ((direction < 0 and Fraction.from_float(rounded) > exact)
                or (direction > 0 and Fraction.from_float(rounded) < exact)):
            rounded = np.nextafter(rounded, direction)
        result.append(rounded)
    return result


def _source_precision_refusal(net, restricted, source_error):
    """Refuse a regular inclusion proof below its fixed source uncertainty.

    Bound the exact restricted polynomial's complete value range, including
    restriction roundoff. If one component varies by less than twice the
    inherited error, an admissible constant error shift removes its zero.
    A strict existence certificate robust to that error is impossible.
    Every smaller box has an even narrower exact range. This is a refusal
    of this certificate, never an assertion that the actual source has no
    root; callers retain the unresolved event and any other proof route.
    """
    axes = tuple(range(net.ndim-1))
    rounding = residual_roundoff_bound(net, depth=2*len(axes))
    width = np.nextafter(restricted.max(axis=axes)-restricted.min(axis=axes), np.inf)
    width = np.nextafter(width+np.nextafter(2.*rounding, np.inf), np.inf)
    # Strictly below twice the fixed error: even an endpoint zero can be
    # removed by one of the admissible constant shifts.
    allowance = np.nextafter(2.*np.asarray(source_error), -np.inf)
    return bool(np.any(np.isfinite(width) & (width <= allowance)))


def _krawczyk_enclosure(net, candidate, radii, source_error, precision_refused=None):
    """A strict inclusion box for a square, regular face-root problem."""
    box = tuple((max(0., float(x-r)), min(1., float(x+r)))
                for x, r in zip(candidate, radii))
    if any(hi <= lo for lo, hi in box):
        return None
    restricted = _restrict(net, box)
    if _source_precision_refusal(net, restricted, source_error):
        if precision_refused is not None:
            precision_refused.append(True)
        return None
    dimensions = net.ndim-1
    axes = tuple(range(dimensions))
    if not jacobian_is_injective(restricted, axes, source_error):
        return None
    eps = np.finfo(float).eps
    magnitude = np.max(np.abs(restricted), axis=axes)
    low, high = [], []
    for axis in axes:
        degree = restricted.shape[axis]-1
        derivative = bernstein_partial_derivative_coeffs(restricted, axis=axis)
        error = degree*(2*source_error+4*eps*magnitude)
        low.append(np.nextafter(derivative.min(axis=axes)-error, -np.inf))
        high.append(np.nextafter(derivative.max(axis=axes)+error, np.inf))
    lower, upper = np.array(low).T, np.array(high).T
    midpoint, radius = .5*(lower+upper), .5*(upper-lower)
    try:
        inverse = np.linalg.inv(midpoint)
    except np.linalg.LinAlgError:
        return None
    value = restricted
    for _ in axes:
        degree = value.shape[0]-1
        work = value.copy()
        for length in range(degree, 0, -1):
            work[:length] = .5*(work[:length]+work[1:length+1])
        value = work[0]
    center = .5-inverse@value
    uncertainty = (np.abs(inverse)@source_error
                   + .5*np.sum(np.abs(np.eye(dimensions)-inverse@midpoint)
                               + np.abs(inverse)@radius, axis=1))
    uncertainty += 64*eps*(1+np.abs(center)+np.abs(inverse)@(np.abs(value)+source_error))
    klow = np.nextafter(center-uncertainty, -np.inf)
    khigh = np.nextafter(center+uncertainty, np.inf)
    if not (np.all(klow > 0.) and np.all(khigh < 1.)):
        return None
    return np.array([_affine_interval(lo, hi, lower, upper)
                     for (lo, hi), lower, upper in zip(box, klow, khigh)])


class BoundaryRootIdentity:
    """Identify roots only on a proved common exact parameter slice.

    Candidate proximity is a broadphase. Each nonidentical representative
    needs an inclusion enclosure, and their common face must be injective
    over the union of those enclosures. Nearby distinct roots fail that
    latter certificate even if their distance is below modeling tolerance.
    """
    def __init__(self, first, second, *, charge=None):
        self.charge = charge
        self.exhausted = False
        self.net = psi_vector_net(first, second)
        source = (np.abs(first[..., :3]).max(axis=(0, 1))*np.abs(second[..., 3]).max()
                  + np.abs(second[..., :3]).max(axis=(0, 1))*np.abs(first[..., 3]).max())
        self.error = residual_roundoff_bound(self.net, depth=2, source_scale=source)
        self.affine = (_pure_affine_coordinates(first), _pure_affine_coordinates(second))
        self.faces = {}
        self.reduced_faces = {}
        self.enclosures = {}
        self.unions = {}
        self.source_ids = (id(first), id(second))
        self.source_census = None
        self.source_certificates = {}
        self.source_queries = {}
        self.source_sequences = {}

    def register_source_root(self, point, certificate):
        """Register an exact original-source cut certificate, not a child net."""
        if (certificate.get('kind') != 'exact_source_planar_cut'
                or tuple(certificate.get('source_ids', ())) != self.source_ids):
            return False
        def exact(pair):
            return Fraction(int(pair[0]), int(pair[1]))
        stored = dict(certificate)
        # Temporary trace-exit proposals can otherwise be collected and
        # their Python IDs reused for an unrelated boundary event.
        stored['owner'] = point
        stored['exact_box'] = tuple(tuple(exact(value) for value in pair)
                                    for pair in certificate['parameter_root_box'])
        stored['exact_interval'] = tuple(exact(value) for value in certificate['interval'])
        stored['exact_polynomial'] = tuple(exact(value) for value in certificate['polynomial'])
        stored['exact_pinned'] = tuple((axis, exact(value))
                                       for axis, value in certificate['pinned'])
        self.source_certificates[id(point)] = stored
        return True

    def _source_enclosure(self, point, radii, common):
        certificate = self.source_certificates.get(id(point))
        if certificate is None:
            return None
        axis, value = common
        box = certificate['exact_box']
        if box[axis] != (Fraction(value), Fraction(value)):
            return None
        from mmcore.numeric.intersection.csx._planar_roots import _outward_interval
        result = np.array([_outward_interval(pair) for pair in box])
        # The caller uses the enclosure for its own cell/arc bounds. Do
        # not return a certificate outside the requested neighborhood.
        if np.any(result[:, 0] < point.stuv-radii) or np.any(result[:, 1] > point.stuv+radii):
            return None
        return result

    def _same_source_root(self, first, second):
        a = self.source_certificates.get(id(first))
        b = self.source_certificates.get(id(second))
        if a is None or b is None:
            return None
        # Two independent original-source existence certificates contained
        # in the same exact singleton identify one lifted root, even when
        # their incident faces use different scalar slice polynomials.
        # Equal rounded representatives or overlapping boxes do not suffice.
        if (a['exact_box'] == b['exact_box']
                and all(lo == hi for lo, hi in a['exact_box'])):
            return True
        if a['varying_axis'] != b['varying_axis']:
            return None
        varying = a['varying_axis']
        # The other graph coordinate pins the scalar problem. A reverse
        # plane-owner cut can carry an additional redundant plane pin;
        # it describes the same source root through the unique inverse.
        graph_pin = varying ^ 1
        pa, pb = dict(a['exact_pinned']), dict(b['exact_pinned'])
        polynomial = a['exact_polynomial']
        if pa.get(graph_pin) != pb.get(graph_pin):
            return False
        if polynomial != b['exact_polynomial']:
            return None
        left, right = a['exact_interval'], b['exact_interval']
        if left[1] < right[0] or right[1] < left[0]:
            return False
        from mmcore.numeric.intersection._exact_univariate import (
            _gcd, _derivative, _divide, _sturm, _open_count, _value)
        class Exhausted(Exception):
            pass
        def tick():
            if not self._spend():
                raise Exhausted
        try:
            if polynomial not in self.source_sequences:
                repeated = _gcd(list(polynomial), _derivative(polynomial), tick)
                squarefree, _ = _divide(list(polynomial), repeated)
                self.source_sequences[polynomial] = squarefree, _sturm(squarefree, tick)
            squarefree, sequence = self.source_sequences[polynomial]
            tick()
            lo, hi = min(left[0], right[0]), max(left[1], right[1])
            count = int(_value(squarefree, lo) == 0)
            if hi > lo:
                count += _open_count(squarefree, sequence, lo, hi)
                count += int(_value(squarefree, hi) == 0)
            return count == 1
        except Exhausted:
            return False

    def _spend(self, amount=1):
        if self.exhausted:
            return False
        if self.charge is None or self.charge(int(amount)):
            return True
        self.exhausted = True
        return False

    def refine_source_box(self, point, box, owner, *, strict=False):
        """Resolve owner-face comparisons by exact source-root counting.

        A root can lie inside an owner while its machine-width isolation
        interval straddles a face. Cut that interval at the exact face
        value, including affine-linked coordinates, and keep the side
        containing its one source root. This is root isolation, not an
        unproved intersection of an existence box with the owner.
        """
        certificate = self.source_certificates.get(id(point))
        if certificate is None or certificate.get('owner') is not point:
            return box
        varying = certificate['varying_axis']
        constraints = getattr(self, 'affine_constraints', None)
        components = getattr(constraints, 'components', None) or ()
        if not any(lo < hi and (lo <= bound <= hi if strict else lo < bound < hi)
                   for (lo, hi), bounds in zip(box, owner) for bound in bounds):
            return box
        key = tuple(map(tuple, box)), tuple(map(tuple, owner)), bool(strict)
        cache = certificate.setdefault('_owner_refinements', {})
        if key in cache:
            return cache[key]
        if not self._spend(1+sum(len(expressions) for expressions, fixed in components)):
            return box
        relations = {varying: (Fraction(0), Fraction(1))}
        for expressions, fixed in components:
            source = next(((a,b) for axis,a,b in expressions if axis == varying), None)
            if source is None:
                continue
            origin, slope = source
            for axis, a, b in expressions:
                relations[axis] = origin-slope*a/b, slope/b
            break
        lo, hi = box[varying]
        def exact(value):
            return value if isinstance(value, Fraction) else Fraction(float(value))
        cuts = sorted({a+b*exact(bound)
                       for axis,(a,b) in relations.items() for bound in owner[axis]
                       if (lo <= a+b*exact(bound) <= hi if strict
                           else lo < a+b*exact(bound) < hi)})
        if not cuts:
            cache[key] = box
            return box
        from mmcore.numeric.intersection._exact_univariate import (
            _gcd, _derivative, _divide, _sturm, _open_count, _value)
        class Stopped(Exception):
            pass
        def tick():
            if not self._spend():
                raise Stopped
        def count(a,b):
            tick()
            return (int(_value(polynomial,a) == 0)
                    + (_open_count(polynomial,sequence,a,b)
                       + int(_value(polynomial,b) == 0) if a < b else 0))
        try:
            original = certificate['exact_polynomial']
            if original not in self.source_sequences:
                repeated = _gcd(list(original),_derivative(original),tick)
                polynomial,_ = _divide(list(original),repeated)
                self.source_sequences[original] = polynomial,_sturm(polynomial,tick)
            polynomial,sequence = self.source_sequences[original]
            if count(lo,hi) != 1:
                # A broad interval containing additional endpoint roots
                # does not identify which one owns this source event.
                cache[key] = box
                return box
            for cut in cuts:
                if not lo <= cut <= hi or lo == hi:
                    continue
                tick()
                if _value(polynomial,cut) == 0:
                    lo = hi = cut
                    break
                if lo < cut < hi:
                    if count(lo,cut):
                        hi = cut
                    else:
                        lo = cut
                # A non-root separator at the interval endpoint is an
                # open bound on this root. Refine to a strictly disjoint
                # *closed* interval when requested for event ordering;
                # rounding this separator to float can erase the gap.
                while strict and lo < hi and lo <= cut <= hi:
                    middle = (lo+hi)/2
                    tick()
                    if _value(polynomial,middle) == 0:
                        lo = hi = middle
                    elif count(lo,middle):
                        hi = middle
                    else:
                        lo = middle
            refined = list(box)
            for axis,(a,b) in relations.items():
                limits = sorted(((lo-a)/b,(hi-a)/b))
                refined[axis] = max(box[axis][0],limits[0]), min(box[axis][1],limits[1])
            result = tuple(refined)
            cache[key] = result if all(a <= b for a,b in result) else box
            return cache[key]
        except Stopped:
            return box

    def separate_source_boxes(self, points, boxes):
        """Refine exact source enclosures against other events' exact pins.

        This proposes neither identity nor a merge. Each existing root is
        retained by its own Sturm count; non-root separating endpoints are
        moved outside the resulting closed interval using exact bisection.
        Unknown certificates and denied work preserve the supplied boxes.
        Original source certificates remain immutable.
        """
        refined = [tuple(map(tuple, box)) for box in boxes]
        if len(points) != len(refined):
            raise ValueError('points and boxes must have equal length')
        if len(refined) < 2:
            return tuple(refined)
        if not self._spend(len(points)*(len(points)-1)):
            return tuple(refined)
        for index, point in enumerate(points):
            certificate = self.source_certificates.get(id(point))
            if certificate is None or certificate.get('owner') is not point:
                continue
            for other, separator in enumerate(refined):
                if index == other:
                    continue
                refined[index] = self.refine_source_box(
                    point, refined[index], separator, strict=True)
        return tuple(refined)

    def _common_slice(self, first, second):
        a, b = first.face[0], second.face[0]
        va, vb = float(first.stuv[a]), float(second.stuv[b])
        if a == b:
            return (a, va) if va == vb else None
        constraints = getattr(self, 'affine_constraints', None)
        if constraints is not None:
            first_key = constraints.face_key(a, va)
            second_key = constraints.face_key(b, vb)
            if (first_key is not None and first_key != ('empty',)
                    and first_key == second_key):
                # Common projected source coordinates can establish an
                # equivalent slice even when no individual world axis is
                # affine. This selects a shared EXISTENCE/UNIQUENESS test;
                # equivalent faces alone never identify their roots.
                return a, va
        if a//2 == b//2:
            return None
        for coordinate in range(3):
            fa = self.affine[a//2].get((coordinate, a%2))
            fb = self.affine[b//2].get((coordinate, b%2))
            if (fa is not None and fb is not None
                    and fa[0]+fa[1]*Fraction.from_float(va)
                    == fb[0]+fb[1]*Fraction.from_float(vb)):
                return a, va
        return None

    def enclose(self, point, radii, common=None):
        box = self._enclose(point, radii, common=common)
        constraints = getattr(self, 'affine_constraints', None)
        if box is not None and constraints is not None:
            necessary = constraints.contract(box)
            if necessary is not None:
                return np.asarray(necessary)
        return box

    def refine_enclosure(self, point, radii):
        """Refine an already isolated source root without changing identity.

        The caller's source box must contain exactly one established root.
        A temporary proposal earns its own existence enclosure; containment
        in that old unique box proves it is the same root. Failure leaves
        the original box and source certificate untouched. On success the
        caller commits the returned box; a new exact certificate is also
        transferred with the original point as its retained owner.
        """
        if not getattr(point,'_source_root_box',False) or point.root_box is None:
            return None
        old = np.asarray(point.root_box,dtype=float)
        requested = np.broadcast_to(np.asarray(radii,dtype=float),(4,))
        if (old.shape != (4,2) or not np.all(np.isfinite(old))
                or np.any(old[:,0] > old[:,1])
                or not np.all(np.isfinite(requested)) or np.any(requested < 0.)):
            return None
        if not self._spend():
            return None
        from copy import copy
        trial = copy(point)
        trial.stuv = np.asarray(point.stuv,dtype=float).copy()
        trial.root_box = None
        trial._source_root_box = False
        # A known scalar source certificate can answer without a new
        # census. Copying it preserves the proof, not numerical identity.
        certificate = self.source_certificates.get(id(point))
        if certificate is not None and certificate.get('owner') is point:
            self.source_certificates[id(trial)] = dict(certificate,owner=trial)
        widths = old[:,1]-old[:,0]
        local_radii = np.minimum(requested,np.where(widths > 0.,widths/2,np.inf))
        try:
            proposed = self.enclose(trial,local_radii)
            if proposed is None:
                return None
            proposed = np.asarray(proposed,dtype=float)
            if (proposed.shape != (4,2) or not np.all(np.isfinite(proposed))
                    or np.any(proposed[:,0] > proposed[:,1])
                    or np.any(proposed[:,0] < old[:,0])
                    or np.any(proposed[:,1] > old[:,1])):
                return None
            refined_certificate = self.source_certificates.get(id(trial))
            if refined_certificate is not None and refined_certificate.get('owner') is trial:
                self.source_certificates[id(point)] = dict(refined_certificate,owner=point)
            return proposed.copy()
        finally:
            self.source_certificates.pop(id(trial),None)

    def _enclose(self, point, radii, common=None):
        """Certify a tight root enclosure on this point's original face."""
        radii = np.asarray(radii, dtype=float)
        axis, value = (common if common is not None else
                       (point.face[0], float(point.stuv[point.face[0]])))
        if axis not in range(4):
            return None
        common = axis, value
        source_enclosure = self._source_enclosure(point, radii, common)
        if source_enclosure is not None:
            return source_enclosure
        if self.source_census is not None:
            trial_radii = np.broadcast_to(radii, (4,)).copy()
            floor = 4*np.spacing(np.maximum(np.abs(point.stuv), np.finfo(float).tiny))
            while np.any(trial_radii > floor):
                query = tuple((max(0., float(x-r)), min(1., float(x+r)))
                              if k != axis else (0., 1.)
                              for k, (x, r) in enumerate(zip(point.stuv, trial_radii)))
                key = common, query
                if key not in self.source_queries:
                    self.source_queries[key] = self.source_census(axis, value, query)
                result = self.source_queries[key]
                if result is None:
                    break
                if (not result.get('boundary_topology_complete', False)
                        or result.get('budget_exhausted', False)):
                    break
                roots = result.get('isolated', [])
                if len(roots) == 1:
                    if self.register_source_root(point, roots[0]['source_cut_certificate']):
                        source_enclosure = self._source_enclosure(point, radii, common)
                        if source_enclosure is not None:
                            return source_enclosure
                    break
                if not roots:
                    break
                # Multiple exact roots in a proposal neighborhood are
                # separate events. Refine the neighborhood of this
                # representative, never merge their isolating intervals.
                trial_radii *= .5
        pinned = {axis: value}
        eliminated = set()
        for target_axis in range(2*(1-axis//2), 2*(1-axis//2)+2):
            for coordinate in range(3):
                if coordinate in eliminated:
                    continue
                source_affine = self.affine[axis//2].get((coordinate, axis%2))
                target_affine = self.affine[target_axis//2].get((coordinate, target_axis%2))
                if source_affine is None or target_affine is None:
                    continue
                exact = ((source_affine[0]+source_affine[1]*Fraction(value)
                          - target_affine[0])/target_affine[1])
                rounded = float(exact)
                # Restriction at a rounded nonbinary rational would change
                # the problem. Such a relation stays in the unreduced solve.
                if Fraction(rounded) != exact or not 0. <= rounded <= 1.:
                    continue
                pinned[target_axis] = rounded
                eliminated.add(coordinate)
                break
        other = [k for k in range(4) if k not in pinned]
        coordinates = [k for k in range(3) if k not in eliminated]
        face_key = (tuple(sorted(pinned.items())), tuple(coordinates))
        if face_key not in self.reduced_faces:
            if not self._spend(max(1, (self.net.size+127)//128)):
                return None
            face = self.net
            for fixed_axis, fixed_value in sorted(pinned.items(), reverse=True):
                left, _ = de_casteljau_split_nd(face, axis=fixed_axis, t=fixed_value)
                face = np.take(left, -1, axis=fixed_axis)
            self.reduced_faces[face_key] = face[..., coordinates]
        face = self.reduced_faces[face_key]
        key = (tuple(point.stuv), common, tuple(radii))
        if key not in self.enclosures:
            trial_radii = radii[other].copy()
            candidate = point.stuv[other]
            floor = 4*np.spacing(np.maximum(np.abs(candidate), np.finfo(float).tiny))
            enclosure = None
            while np.all(trial_radii > floor):
                if not self._spend(max(1, (4*face.size+127)//128)):
                    return None
                precision_refused = []
                enclosure = _krawczyk_enclosure(
                    face, candidate, trial_radii, self.error[coordinates],
                    precision_refused=precision_refused)
                if enclosure is not None:
                    break
                if precision_refused:
                    break
                # A failed wide inclusion test does not disprove a root.
                # Refine the candidate neighborhood until it isolates the
                # regular face root, reaches representable precision, or
                # exhausts the shared work allowance.
                trial_radii *= .5
            # A first Krawczyk image may be much wider than the actual
            # root uncertainty. Iterate from an expanded enclosure while
            # it contracts geometrically, retaining the last proved box
            # if finite precision or the work allowance stops refinement.
            while enclosure is not None:
                widths = enclosure[:,1]-enclosure[:,0]
                if not np.all(widths > 0.):
                    break
                if not self._spend(max(1, (4*face.size+127)//128)):
                    break
                refined = _krawczyk_enclosure(
                    face, enclosure.mean(axis=1), .75*widths,
                    self.error[coordinates])
                if refined is None or np.any(refined[:,0] < enclosure[:,0]) or np.any(refined[:,1] > enclosure[:,1]):
                    break
                new_widths = refined[:,1]-refined[:,0]
                enclosure = refined
                if np.max(new_widths) >= .5*np.max(widths):
                    break
            self.enclosures[key] = enclosure
        result = self.enclosures[key]
        if result is None:
            return None
        box = np.empty((4, 2))
        for fixed_axis, fixed_value in pinned.items():
            box[fixed_axis] = fixed_value
        box[other] = result
        return box

    def __call__(self, first, second, param_tol, xyz_tol):
        # Pair scans are work too, including broadphase failures. A denied
        # certificate retains separate events and lets the caller report
        # the exhausted allowance through its existing budget object.
        if not self._spend():
            return False
        if first is second:
            return True
        exact_a = self.source_certificates.get(id(first))
        exact_b = self.source_certificates.get(id(second))
        if exact_a is not None and exact_b is not None:
            if any(a[0] > b[1] or b[0] > a[1]
                   for a, b in zip(exact_a['exact_box'], exact_b['exact_box'])):
                return False
            same_source = self._same_source_root(first, second)
            if same_source is not None:
                return bool(same_source and np.linalg.norm(first.xyz-second.xyz) <= xyz_tol)
        first_box, second_box = getattr(first, 'root_box', None), getattr(second, 'root_box', None)
        separate = False
        if first_box is not None and second_box is not None:
            a, b = np.asarray(first_box), np.asarray(second_box)
            separate = bool(np.any((a[:, 0] > b[:, 1]) | (b[:, 0] > a[:, 1])))
            if separate and getattr(first, '_source_root_box', False) and getattr(second, '_source_root_box', False):
                return False
        radii = np.asarray(param_tol, dtype=float)
        if (not np.all(np.abs(first.stuv-second.stuv) <= radii)
                or np.linalg.norm(first.xyz-second.xyz) > xyz_tol):
            return False
        same_source = self._same_source_root(first, second)
        if same_source is not None:
            return same_source
        common = self._common_slice(first, second)
        if common is None:
            # Incident faces of the same chart need not be equivalent
            # slices. Obtain existence on each original face separately;
            # only a subsequent exact source-root identity may unify them.
            # This also handles outer corner proposals before either face
            # has acquired its lazy source census certificate.
            if self.source_census is not None:
                for point in (first, second):
                    if id(point) not in self.source_certificates:
                        if not self._spend():
                            return False
                        enclosure = (self.refine_enclosure(point, radii)
                                     if getattr(point, '_source_root_box', False)
                                     else self.enclose(point, radii))
                        if enclosure is None:
                            return False
                same_source = self._same_source_root(first, second)
                if same_source is not None:
                    return same_source
            return False
        axis, value = common
        other = [i for i in range(4) if i != axis]
        if common not in self.faces:
            if not self._spend(max(1, (self.net.size+127)//128)):
                return False
            left, _ = de_casteljau_split_nd(self.net, axis=axis, t=value)
            self.faces[common] = np.take(left, -1, axis=axis)
        face = self.faces[common]
        boxes = []
        for point in (first, second):
            certificate = self.source_certificates.get(id(point))
            if certificate is not None or getattr(point, '_source_root_box', False):
                if not self._spend():
                    return False
                original_axis = point.face[0]
                original_value = float(point.stuv[original_axis])
                if certificate is not None:
                    exact_box = certificate['exact_box']
                    if exact_box[original_axis] != (Fraction(original_value),)*2:
                        return False
                    from mmcore.numeric.intersection.csx._planar_roots import _outward_interval
                    box = np.array([_outward_interval(pair) for pair in exact_box])
                else:
                    owned = getattr(point, 'root_box', None)
                    if owned is None:
                        return False
                    box = np.asarray(owned, dtype=float).copy()
                if (box.shape != (4, 2) or not np.all(np.isfinite(box))
                        or np.any(box[:, 0] < 0.) or np.any(box[:, 1] > 1.)
                        or np.any(box[:, 0] > box[:, 1])
                        or not np.all(box[original_axis] == original_value)
                        or not box[axis, 0] <= value <= box[axis, 1]):
                    return False
                # This already owned root lies on its original face.
                # Exact slice equivalence proves the common pin without
                # changing which root the enclosure owns. A fresh solve
                # around the displayed proposal could select another root.
                box[axis] = value, value
            else:
                # Rounded child-net boxes are only hints. Fresh proposals
                # still need original-source existence on the common face.
                box = self.enclose(point, radii, common=common)
            if box is None:
                return False
            boxes.append(box[other])
        same_source = self._same_source_root(first, second)
        if same_source is not None:
            return same_source
        core = np.array([(min(a[0], b[0]), max(a[1], b[1]))
                         for a, b in zip(*boxes)])
        padding = radii[other].copy()
        floor = np.maximum(core[:,1]-core[:,0],
                           4*np.spacing(np.maximum(np.abs(core).max(axis=1),
                                                   np.finfo(float).tiny)))
        while True:
            union = tuple((max(0., lo-pad), min(1., hi+pad))
                          for (lo, hi), pad in zip(core, padding))
            key = (common, union)
            if key not in self.unions:
                if not self._spend(max(1, (4*face.size+127)//128)):
                    return False
                self.unions[key] = jacobian_is_injective(
                    _restrict(face, union), (0, 1, 2), self.error)
            if self.unions[key]:
                return True
            if np.all(padding <= floor):
                return False
            # The complete existence enclosures remain in every union.
            # Shrinking only optional padding can prove a common nearby
            # root without ever erasing a second independently enclosed root.
            padding *= .5

    def resolve_region(self, point, radii, region):
        """Resolve a complete pending face region on the supplied source.

        The child solver's approximate root is only a proposal. Source
        inclusion establishes existence inside the whole pending region;
        source injectivity excludes additional roots anywhere in it.
        """
        enclosure = self.enclose(point, radii)
        if enclosure is None:
            return None
        region = np.asarray(region, dtype=float)
        if np.any(enclosure[:, 0] < region[:, 0]) or np.any(enclosure[:, 1] > region[:, 1]):
            return None
        axis, value = point.face[0], float(point.stuv[point.face[0]])
        common = axis, value
        if common not in self.faces:
            if not self._spend(max(1, (self.net.size+127)//128)):
                return None
            left, _ = de_casteljau_split_nd(self.net, axis=axis, t=value)
            self.faces[common] = np.take(left, -1, axis=axis)
        other = [k for k in range(4) if k != axis]
        intervals = tuple((float(lo), float(hi)) if hi > lo else
                          (max(0., float(lo)-radii[k]), min(1., float(hi)+radii[k]))
                          for k, (lo, hi) in zip(other, region[other]))
        key = (common, intervals)
        if key not in self.unions:
            face = self.faces[common]
            if not self._spend(max(1, (4*face.size+127)//128)):
                return None
            self.unions[key] = jacobian_is_injective(
                _restrict(face, intervals), (0, 1, 2), self.error)
        return enclosure if self.unions[key] else None
