"""Necessary source-domain contraction for the underdetermined Psi system.

An interval mean-value equation bounds each pivot coordinate in terms of
the other three. Gaussian preconditioning is a numerical proposal only:
every resulting row is evaluated with outward intervals. A contraction
preserves all zeros; it supplies neither existence nor uniqueness.
"""
import numpy as np

from mmcore.numeric._bezier_common import restrict_net_axis_v
from mmcore.numeric.intersection.ssx._ssx_cofactor_bounds import _interval_multiply


def _project(matrix, low, high):
    lower = np.zeros((len(matrix),)+low.shape[1:])
    upper = lower.copy()
    for coordinate in range(3):
        scale = matrix[:, coordinate].reshape((-1,)+(1,)*(low.ndim-1))
        lo, hi = _interval_multiply(scale, scale, low[coordinate], high[coordinate])
        lower = np.nextafter(lower+lo, -np.inf)
        upper = np.nextafter(upper+hi, np.inf)
    return lower, upper


def _divide(low, high, denominator_low, denominator_high):
    if denominator_low <= 0. <= denominator_high:
        return None
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        values = np.array([low/denominator_low, low/denominator_high,
                           high/denominator_low, high/denominator_high])
    if np.any(np.isnan(values)):
        return None
    return np.nextafter(values.min(), -np.inf), np.nextafter(values.max(), np.inf)


class SourceResidualContractor:
    """Cached interval Gauss-Seidel steps over original-source controls.

    The supplied SourceCofactorBounds owns immutable source residual and
    derivative nets, their arithmetic errors, and the shared work ledger.
    Denied/unsupported work preserves the current necessary domain.
    None is returned only after a certified empty interval intersection.
    """
    def __init__(self, source_bounds):
        self.source = source_bounds
        self.cache = {}

    def contract(self, box, *, stop_when=None):
        original = tuple((float(lo), float(hi)) for lo, hi in box)
        if original in self.cache:
            return self.cache[original]
        source = self.source
        if (source.residual is None or source.derivatives is None
                or len(original) != 4
                or any(not 0. <= lo <= hi <= 1. for lo, hi in original)):
            return original
        current = np.asarray(original)
        while True:
            units = max(1, (source.residual.size
                            + 10*sum(net.size for net in source.derivatives)+127)//128)+4
            if not source._spend(units):
                break
            center = np.clip(.5*current[:, 0]+.5*current[:, 1],
                             current[:, 0], current[:, 1])
            jacobian_low, jacobian_high, restricted_jets = [], [], []
            for net, error in zip(source.derivatives, source.errors):
                restricted = net
                for axis, (lo, hi) in enumerate(current):
                    restricted = restrict_net_axis_v(restricted, axis, lo, hi, 0., 1.)
                jacobian_low.append(np.nextafter(restricted.min(axis=(0, 1, 2, 3))-error, -np.inf))
                jacobian_high.append(np.nextafter(restricted.max(axis=(0, 1, 2, 3))+error, np.inf))
                restricted_jets.append((np.nextafter(np.moveaxis(restricted, -1, 0)
                                                     -error[:, None, None, None, None], -np.inf),
                                        np.nextafter(np.moveaxis(restricted, -1, 0)
                                                     +error[:, None, None, None, None], np.inf)))
            jlo, jhi = np.asarray(jacobian_low).T, np.asarray(jacobian_high).T
            if not np.all(np.isfinite(jlo)) or not np.all(np.isfinite(jhi)):
                break
            midpoint = .5*jlo+.5*jhi
            scale = float(np.max(np.abs(midpoint)))
            if not np.isfinite(scale) or scale == 0.:
                break
            scaled = midpoint/scale
            value = source.residual
            for axis, coordinate in enumerate(center):
                value = restrict_net_axis_v(value, axis, coordinate, coordinate, 0., 1.)
            vlo = np.nextafter(value.min(axis=(0, 1, 2, 3))-source.residual_error, -np.inf)
            vhi = np.nextafter(value.max(axis=(0, 1, 2, 3))+source.residual_error, np.inf)
            next_box = current.copy()
            proposals = []
            try:
                left, _, _ = np.linalg.svd(scaled, full_matrices=False)
                proposals.append((left.T, tuple((row, axis) for row in range(3) for axis in range(4))))
            except np.linalg.LinAlgError:
                pass
            for inverse, targets in proposals:
                projected = [_project(inverse, low, high) for low, high in restricted_jets]
                plow = np.column_stack([low.min(axis=(1, 2, 3, 4)) for low, high in projected])
                phigh = np.column_stack([high.max(axis=(1, 2, 3, 4)) for low, high in projected])
                flow, fhigh = _project(inverse, vlo, vhi)
                if not all(np.all(np.isfinite(a)) for a in (plow, phigh, flow, fhigh)):
                    continue
                for row, axis in targets:
                    lo, hi = -fhigh[row], -flow[row]
                    for other in range(4):
                        if other == axis:
                            continue
                        dlow = np.nextafter(next_box[other, 0]-center[other], -np.inf)
                        dhigh = np.nextafter(next_box[other, 1]-center[other], np.inf)
                        a, b = _interval_multiply(plow[row, other], phigh[row, other], dlow, dhigh)
                        lo, hi = np.nextafter(lo-b, -np.inf), np.nextafter(hi-a, np.inf)
                    displacement = _divide(lo, hi, plow[row, axis], phigh[row, axis])
                    if displacement is None:
                        continue
                    lower = np.nextafter(center[axis]+displacement[0], -np.inf)
                    upper = np.nextafter(center[axis]+displacement[1], np.inf)
                    next_box[axis] = max(next_box[axis, 0], lower), min(next_box[axis, 1], upper)
                    if next_box[axis, 0] > next_box[axis, 1]:
                        self.cache[original] = None
                        return None
            if np.array_equal(current, next_box):
                break
            current = next_box
            if stop_when is not None and stop_when(tuple(map(tuple, current))):
                break
            # One complete Gauss-Seidel sweep is a necessary-domain
            # certificate. Global source subdivision supplies subsequent
            # refinements; iterating to a floating fixed point can spend
            # the search ledger on an otherwise inconclusive proposal.
            break
        result = tuple(map(tuple, current))
        self.cache[original] = result
        return result
