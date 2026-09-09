"""C3 retirement needs a global proof on both original source charts."""
import numpy as np

from mmcore.numeric.intersection.ssx import _bez_ssx5 as ssx
from mmcore.numeric.intersection.ssx import _ssx5_singular as singular
from mmcore.numeric.intersection.ssx._ssx5_overlap import _exact_surface_injective
from mmcore.numeric.intersection.ssx._ssx_root_identity import _pure_affine_coordinates


def _curved_pair():
    # x/y are small nonlinear perturbations of (s,t), so no world
    # coordinate directly recovers either parameter. A constant left
    # projection nevertheless proves global chart injectivity.
    first = np.array([[[s+s*t/8,t+s*t/8,t-.25-.25*square,1.] for t in (0.,1.)]
                      for s,square in zip((0.,.5,1.),(0.,0.,1.))])
    second = np.array([[[u,v,0.,1.] for v in (-.25,1.5)] for u in (-.25,1.5)])
    return first,second


def test_two_original_global_chart_proofs_skip_c3_search(monkeypatch):
    first,second = _curved_pair()
    assert not _pure_affine_coordinates(first)
    assert _exact_surface_injective(first)
    calls = []
    original = singular.c3_pass
    def tracked(*args,**kwargs):
        calls.append(True)
        return original(*args,**kwargs)
    monkeypatch.setattr(singular,'c3_pass',tracked)
    result = ssx.bez_ssx(first,second,rational=True)
    assert result['complete'],result['status']
    assert any(len(branch.curve[0]) >= 9 for branch in result['branches'])
    assert not calls
    assert result['status']['work']['cell_counts'].get('global_injectivity',0) > 0


def test_folded_and_nonuniform_charts_have_no_polynomial_injectivity_proof():
    folded = np.array([[[x,t,0.,1.] for t in (0.,1.)] for x in (.25,-.25,.25)])
    assert not _exact_surface_injective(folded)
    weighted = _curved_pair()[0]
    weighted[1] *= 2.
    assert not _exact_surface_injective(weighted)
