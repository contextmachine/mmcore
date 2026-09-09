"""Main SSX exclusions belong to original source residuals."""
from types import SimpleNamespace

import numpy as np

from mmcore.numeric.intersection.ssx._ssx_cofactor_bounds import SourceCofactorBounds
from mmcore.numeric.intersection.ssx._ssx_source_residual import SourceResidualExclusions


def _surface(height):
    return np.array([[[s,t,height,1.] for t in (0.,1.)] for s in (0.,1.)])


def test_original_source_root_is_never_excluded_by_cancelled_child_values():
    source = SourceCofactorBounds(_surface(0.),_surface(0.),polynomial=False)
    exclude = SourceResidualExclusions(source)
    assert not exclude(((.2,.7),(.3,.8),(.4,.9),(.1,.6)))
    assert not exclude(((.5,.5),)*4)


def test_direct_source_separation_and_disjoint_parameter_owners_are_excluded():
    assert SourceResidualExclusions(SourceCofactorBounds(
        _surface(0.),_surface(1.),polynomial=False))(((0.,1.),)*4)
    exclude = SourceResidualExclusions(SourceCofactorBounds(
        _surface(0.),_surface(0.),polynomial=False))
    assert exclude(((0.,.2),(0.,1.),(.8,1.),(0.,1.)))


def test_aligned_projection_can_separate_a_hull_crossing_every_coordinate_axis():
    # Every coefficient has x+y=1, but each coordinate interval contains0.
    net = np.array([[[-1.,2.,0.],[2.,-1.,0.]]]).reshape(2,1,1,1,3)
    source = SimpleNamespace(residual=net,residual_error=np.full(3,1e-12))
    assert SourceResidualExclusions(source)(((0.,1.),)*4)


def test_inherited_source_error_prevents_false_empty_from_positive_rounded_net():
    eta = np.nextafter(0.,1.)
    source = SimpleNamespace(residual=np.full((2,2,2,2,3),eta),
                             residual_error=np.full(3,4*eta))
    assert not SourceResidualExclusions(source)(((.25,.75),)*4)


def test_denial_is_prepaid_and_cached_certificates_do_not_repeat_work(monkeypatch):
    from mmcore.numeric.intersection.ssx import _ssx_source_residual as module
    source = SourceCofactorBounds(_surface(0.),_surface(1.),polynomial=False)
    calls = []
    exclude = SourceResidualExclusions(source,charge=lambda units:calls.append(units) or True)
    box = ((0.,1.),)*4
    assert exclude(box)
    charged = list(calls)
    assert exclude(box) and calls == charged
    denied = SourceResidualExclusions(source,charge=lambda units:False)
    def forbidden(*args,**kwargs):
        raise AssertionError('unpaid source restriction')
    monkeypatch.setattr(module,'restrict_net_axis_v',forbidden)
    assert not denied(box) and denied.exhausted
