"""An unidentified exit cannot discharge an already proved two-port arc."""
from fractions import Fraction

import numpy as np


def test_declined_equivalent_face_identity_keeps_known_arc_owner_partial(monkeypatch):
    from mmcore.numeric.intersection.ssx import _bez_ssx5 as ssx
    from mmcore.numeric.intersection.ssx._ssx_root_identity import BoundaryRootIdentity

    first = np.array([[[i/2,j/2,i*j/4] for j in range(3)] for i in range(3)])
    second = first.copy()
    z = (1.,-1.,1.)
    second[...,2] += np.array([[z[i]+z[j]-.5 for j in range(3)] for i in range(3)])
    matrix = np.array([[1.,1.,0.],[0.,1.,1.],[1.,0.,1.]])

    def original_common_slice(self,first,second):
        # Model an unavailable projected-coordinate proof. Same-face and
        # original pure-coordinate proofs remain admissible.
        a,b = first.face[0],second.face[0]
        va,vb = float(first.stuv[a]),float(second.stuv[b])
        if a == b:
            return (a,va) if va == vb else None
        if a//2 == b//2:
            return None
        for coordinate in range(3):
            left = self.affine[a//2].get((coordinate,a%2))
            right = self.affine[b//2].get((coordinate,b%2))
            if (left is not None and right is not None
                    and left[0]+left[1]*Fraction(va) == right[0]+right[1]*Fraction(vb)):
                return a,va
        return None

    monkeypatch.setattr(BoundaryRootIdentity,'_common_slice',original_common_slice)
    real_trace = ssx._trace_cell_by_registrations
    calls = []
    def recording(cell,*args,**kwargs):
        result = real_trace(cell,*args,**kwargs)
        if len(cell.crossings) == 2:
            calls.append((len(result[0]),cell.trace_incomplete))
        return result
    monkeypatch.setattr(ssx,'_trace_cell_by_registrations',recording)
    result = ssx.bez_ssx(first@matrix.T,second@matrix.T,rational=False,
                         atol=1e-3,max_cells=60000,max_depth=4)
    assert calls
    assert any(incomplete for count,incomplete in calls)
    assert all(incomplete or count <= 1 for count,incomplete in calls)
    assert not result['complete']
    assert result['unresolved_regions']
