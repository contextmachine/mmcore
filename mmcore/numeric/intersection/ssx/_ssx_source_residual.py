"""Cached original-source residual hull exclusions for complete SSX cells."""
import numpy as np

from mmcore.numeric._bezier_common import restrict_net_axis_v
from mmcore.numeric.intersection._root_box_certificate import residual_roundoff_bound
from mmcore.numeric.intersection.ssx._ssx_cofactor_bounds import _interval_multiply


class SourceResidualExclusions:
    """Prove emptiness from original residual coefficient intervals only.

    A proposed separating direction is arbitrary. Its interval dot product
    must exclude zero over the entire coefficient hull. Unknown or unpaid
    work returns False, preserving the caller's discovery obligation.
    """
    def __init__(self, source_bounds, charge=None):
        self.source = source_bounds
        self.charge = charge
        self.cache = {}
        self.exhausted = False

    def _spend(self, units):
        if self.exhausted:
            return False
        if self.charge is None or self.charge(max(1,int(units))):
            return True
        self.exhausted = True
        return False

    def __call__(self, box):
        key = tuple((float(lo),float(hi)) for lo,hi in box)
        if len(key) != 4 or any(not 0. <= lo <= hi <= 1. for lo,hi in key):
            return False
        if key in self.cache:
            return self.cache[key]
        net,error = self.source.residual,self.source.residual_error
        if net is None or error is None or not self._spend((8*net.size+127)//128):
            return False
        restricted = net
        with np.errstate(over='ignore',invalid='ignore',under='ignore'):
            for axis,(lo,hi) in enumerate(key):
                restricted = restrict_net_axis_v(restricted,axis,lo,hi,0.,1.)
            error = np.nextafter(error+residual_roundoff_bound(net,depth=8),np.inf)
            low = np.nextafter(restricted-error,-np.inf).reshape(-1,3)
            high = np.nextafter(restricted+error,np.inf).reshape(-1,3)
        if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
            return False
        if np.any(low.min(axis=0) > 0.) or np.any(high.max(axis=0) < 0.):
            self.cache[key] = True
            return True
        if not self._spend((16*net.size+127)//128):
            return False

        flat = restricted.reshape(-1,3)
        scale = float(np.abs(flat).max())
        if not np.isfinite(scale) or scale == 0.:
            self.cache[key] = False
            return False
        scaled = flat/scale
        mean = scaled.mean(axis=0)

        def separated(direction):
            if not np.all(np.isfinite(direction)) or not np.any(direction):
                return False
            lower = np.zeros(len(flat))
            upper = lower.copy()
            with np.errstate(over='ignore',invalid='ignore',under='ignore'):
                for axis in range(3):
                    a,b = _interval_multiply(low[:,axis],high[:,axis],
                                             direction[axis],direction[axis])
                    lower = np.nextafter(lower+a,-np.inf)
                    upper = np.nextafter(upper+b,np.inf)
            return bool(np.all(lower > 0.) or np.all(upper < 0.))

        result = separated(mean)
        if not result:
            try:
                centered = scaled-mean
                _,directions = np.linalg.eigh(centered.T@centered)
                result = separated(directions[:,0])
            except np.linalg.LinAlgError:
                pass
        self.cache[key] = result
        return result
