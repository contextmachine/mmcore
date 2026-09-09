"""Reuse complete exact source-face censuses under affine slice identity."""
from fractions import Fraction

from mmcore.numeric.intersection.csx._planar_roots import _outward_interval


def _decode(pair):
    return Fraction(int(pair[0]), int(pair[1]))


class SourceFaceCensusCache:
    """Cache topology while preserving each requested floating representative.

    ``uncached`` solves the requested ORIGINAL source face and owns its
    computation ledger. ``constraints`` proves equivalent parameter slices;
    no canonical Fraction pin is rounded and passed to a different solve.
    """
    def __init__(self, uncached, constraints, source_ids, *, charge,
                 can_retry=lambda: True):
        self.uncached = uncached
        self.constraints = constraints
        self.source_ids = tuple(source_ids)
        self.charge = charge
        self.can_retry = can_retry
        self.censuses = {}
        self.census_faces = {}

    @staticmethod
    def _partial(box, reason='work_budget'):
        return {'isolated': [], 'boundary_topology_complete': False,
                'budget_exhausted': True, 'truncation_cause': 'max_cells',
                'unresolved_source_boxes': [{'parameter_root_box': box, 'reason': reason}]}

    def _source_valid(self, result):
        return all(tuple(root.get('source_cut_certificate', {}).get('source_ids', ()))
                   == self.source_ids for root in result.get('isolated', []))

    def _local(self, axis, value, box):
        result = self.uncached(axis, value, box)
        if result is not None and not self._source_valid(result):
            return self._partial(box, 'source_certificate_identity')
        return result

    def __call__(self, axis, value, box):
        key = self.constraints.face_key(axis, value)
        if key is None:
            return self._partial(box) if self.constraints.exhausted else self._local(axis, value, box)
        if key == ('empty',):
            return {'isolated': [], 'overlaps': [], 'parameter_fibers': [],
                    'cells_processed': 0, 'boundary_topology_complete': True,
                    'budget_exhausted': False}
        full = ((0., 1.),)*4
        exact_value = value if isinstance(value, Fraction) else Fraction(float(value))
        if key not in self.censuses:
            self.censuses[key] = self.uncached(axis, value, full)
            self.census_faces[key] = axis, exact_value
        census = self.censuses[key]
        if census is None:
            return None
        if not self._source_valid(census):
            return self._partial(box, 'source_certificate_identity')
        if (census.get('budget_exhausted', False)
                or not census.get('boundary_topology_complete', False)):
            # A partial full-face result never supplies unfiltered roots to
            # a smaller owner. A local query can still succeed independently.
            if (tuple(tuple(pair) for pair in box) == full
                    and self.census_faces[key] == (axis, exact_value)):
                return census
            # Even a full-domain query on an equivalent face needs new
            # representative validation and exact metadata rebinding.
            # Incomplete censuses bypass that complete-root filtering path.
            return self._local(axis, value, box) if self.can_retry() else self._partial(box)
        if not self.charge(max(1, len(census.get('isolated', [])))):
            return self._partial(box)
        exact_box = tuple((Fraction(float(lo)), Fraction(float(hi))) for lo, hi in box)
        roots = []
        for original in census.get('isolated', []):
            certificate = original['source_cut_certificate']
            root = original
            if certificate['axis'] != axis or _decode(certificate['cut']) != exact_value:
                # Topological equivalence does not certify an altered float
                # representative: reuse only an already matching coordinate.
                if original['stuv'][axis] != float(value):
                    return self._local(axis, value, box) if self.can_retry() else self._partial(box)
                certificate = self.constraints.rebind_source_certificate(certificate, axis, value)
                if certificate is None:
                    return self._local(axis, value, box) if self.can_retry() else self._partial(box)
                root = dict(original)
                root['source_cut_certificate'] = certificate
                root['parameter_root_box'] = tuple(_outward_interval(tuple(_decode(x) for x in pair))
                                                  for pair in certificate['parameter_root_box'])
            enclosure = tuple(tuple(_decode(x) for x in pair)
                              for pair in certificate['parameter_root_box'])
            if any(lo > b or hi < a for (lo, hi), (a, b) in zip(enclosure, exact_box)):
                continue
            if any(lo < a or hi > b for (lo, hi), (a, b) in zip(enclosure, exact_box)):
                return self._local(axis, value, box) if self.can_retry() else self._partial(box)
            roots.append(root)
        result = dict(census)
        result['isolated'] = roots
        result['cells_processed'] = 0  # the original full census was paid once
        return result
