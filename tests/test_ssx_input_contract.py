"""Control hull exclusions require finite, strictly positive weights."""
import numpy as np
import pytest

from mmcore.nurbs._nurbs_eval import NURBSSurfaceTuple
from mmcore.numeric.intersection.ssx import _bez_ssx5 as bez
from mmcore.numeric.intersection.ssx import _nssx5 as nurbs


def surface():
    return np.array([[[u,v,0.,1.] for v in (0.,1.)] for u in (0.,1.)])


def forbidden(*args, **kwargs):
    raise AssertionError('unsupported source reached geometry processing')


@pytest.mark.parametrize('weight', [0., -1., np.nan, np.inf])
@pytest.mark.parametrize('side', [0, 1])
def test_bezier_rejects_invalid_weights_before_normalization_or_aabb(monkeypatch, weight, side):
    pair = [surface(), surface()]
    pair[side][0,0,3] = weight
    monkeypatch.setattr(bez, '_ssx_normalization_context', forbidden)
    with pytest.raises(ValueError, match='finite.*positive|positive.*finite'):
        bez.bez_ssx(*pair)


@pytest.mark.parametrize('rational', [False, True])
def test_bezier_rejects_nonfinite_coordinates_even_with_zero_work(rational):
    bad = surface() if rational else surface()[...,:3]
    bad[0,0,0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        bez.bez_ssx(bad,bad,rational=rational,max_cells=0)


def tuple_surface(net):
    return NURBSSurfaceTuple(
        order_u=2,order_v=2,knot_u=np.array([0.,0.,1.,1.]),
        knot_v=np.array([0.,0.,1.,1.]),
        control_points=net[...,:3],weights=net[...,3])


@pytest.mark.parametrize('weight', [0., -1., np.nan, np.inf])
def test_nurbs_rejects_invalid_weights_before_domain_or_decomposition(monkeypatch, weight):
    net = surface()
    net[0,0,3] = weight
    monkeypatch.setattr(nurbs,'_domain_ctx',forbidden)
    with pytest.raises(ValueError, match='finite.*positive|positive.*finite'):
        nurbs.nurbs_ssx(tuple_surface(net),tuple_surface(surface()))


def test_nurbs_rejects_nonfinite_coordinates_before_decomposition(monkeypatch):
    net = surface()
    net[0,0,0] = np.inf
    monkeypatch.setattr(nurbs,'decompose_surface',forbidden)
    with pytest.raises(ValueError, match='finite'):
        nurbs.nurbs_ssx(tuple_surface(net),tuple_surface(surface()))


def test_nonuniform_positive_weights_are_supported_without_mutation():
    net = surface()
    net[0,0] *= 2.
    original = net.copy()
    # Validation must preserve the caller's homogeneous representation.
    result = bez.bez_ssx(net,net,max_cells=0)
    assert not result['complete']
    np.testing.assert_array_equal(net,original)
