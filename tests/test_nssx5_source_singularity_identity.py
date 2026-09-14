"""Singularity metadata distinguishes the two owning surface charts."""

import numpy as np

from mmcore.numeric.intersection.ssx import _nssx5 as ssx
from mmcore.numeric.intersection.ssx._bez_ssx5 import SSXSingularity


def _context(closed=False):
    return ssx._DomainCtx(np.zeros(4),np.ones(4),np.ones(4),np.full(4,.001),(closed,False,False,False))






def test_cusps_on_different_source_charts_keep_both_surface_owners():
    q = np.full(4,.5)
    values = [SSXSingularity('cusp',q.copy(),np.zeros(3),surface=owner) for owner in (1,2)]
    result = ssx._assemble_singularities(values,[],_context(),.001,ssx._make_aggregate({},1))
    assert len(result) == 2 and {value.surface for value in result} == {1,2}
