"""Transport the original SSX residual to a closed CSX face product.

Restriction changes the Bernstein basis, never the source equations. The
absolute coefficient envelope survives every restriction and permutation;
the rounded curve and surface supplied to Newton are only proposals.
"""
from fractions import Fraction

import numpy as np

from mmcore.numeric._bezier_common import restrict_net_axis_v
from mmcore.numeric.intersection._root_box_certificate import (
    exact_bernstein_value, residual_roundoff_bound)


class SourceFaceResiduals:
    def __init__(self, source_bounds, charge=None, sources=None):
        self.source = source_bounds
        self.charge = charge
        self.cache = {}
        self.exhausted = False
        self.sources = sources
        self.exact_values = {}

    def exact_root(self, face, parameters, *, prepaid=False):
        """Prove existence at the exact source affine image of local floats.

        This also handles closed-face roots where strict interior Krawczyk
        inclusion is unavailable. A rounded global representative is never
        substituted for the exact affine image.
        """
        if self.sources is None or len(parameters) != 3:
            return False
        axis, value, box = face
        if any(not 0. <= float(x) <= 1. for x in parameters):
            return False
        order = [axis ^ 1]+([2, 3] if axis < 2 else [0, 1])
        source_parameters = [Fraction(0)]*4
        source_parameters[axis] = Fraction(float(value))
        for k, local in zip(order, parameters):
            lo, hi = (Fraction(float(x)) for x in box[k])
            source_parameters[k] = lo+(hi-lo)*Fraction(float(local))
        key = tuple(source_parameters)
        if key in self.exact_values:
            return self.exact_values[key]
        if self.exhausted:
            return False
        units = max(1, (sum(net.size for net in self.sources)+31)//32)
        if not prepaid and self.charge is not None and not self.charge(units):
            self.exhausted = True
            return False
        first = exact_bernstein_value(self.sources[0], key[:2])
        second = exact_bernstein_value(self.sources[1], key[2:])
        answer = all(first[k]*second[3] == second[k]*first[3] for k in range(3))
        self.exact_values[key] = answer
        return answer

    def __call__(self, axis, value, box):
        axis, value = int(axis), float(value)
        box = tuple((float(lo), float(hi)) for lo, hi in box)
        if (axis not in range(4) or not 0. <= value <= 1. or len(box) != 4
                or any(not 0. <= lo <= hi <= 1. for lo, hi in box)
                or not box[axis][0] <= value <= box[axis][1]):
            return None
        key = axis, value, box
        if key in self.cache:
            return self.cache[key]
        net, inherited = self.source.residual, self.source.residual_error
        if net is None or inherited is None or self.exhausted:
            return None
        units = max(1, (8*net.size+127)//128)
        if self.charge is not None and not self.charge(units):
            self.exhausted = True
            return None
        restricted = net
        with np.errstate(over='ignore', invalid='ignore', under='ignore'):
            for k, (lo, hi) in enumerate(box):
                if k == axis:
                    lo = hi = value
                restricted = restrict_net_axis_v(restricted, k, lo, hi, 0., 1.)
            # Every axis uses at most two de Casteljau splits. The error
            # floor is based on the original operands, not a cancelled or
            # shrinking child. Bernstein restriction is a convex map, so
            # inherited absolute coefficient errors do not grow.
            error = np.nextafter(inherited+residual_roundoff_bound(net, depth=8), np.inf)
        if not np.all(np.isfinite(restricted)) or not np.all(np.isfinite(error)):
            return None
        restricted = np.take(restricted, 0, axis=axis)
        remaining = [k for k in range(4) if k != axis]
        order = [axis ^ 1]+([2, 3] if axis < 2 else [0, 1])
        residual = np.transpose(restricted, [remaining.index(k) for k in order]+[3]).copy()
        if axis >= 2:
            residual = -residual
        residual.setflags(write=False)
        error.setflags(write=False)
        self.cache[key] = residual, error
        return self.cache[key]
