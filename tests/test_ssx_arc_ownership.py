from fractions import Fraction as F

from mmcore.numeric.intersection.ssx._ssx_arc_ownership import map_source_path, source_path_covered


def _line(a,b):
    return ((F(a),)*4,(F(b),)*4)


def test_proved_source_path_union_covers_varied_sampling_and_reversal():
    assert source_path_covered(_line(0,1),[_line(F(1,2),0),_line(F(1,2),1)])


def test_exact_source_interval_union_keeps_an_unrepresentable_gap():
    e = F(1,2**60)
    assert not source_path_covered(_line(0,1),[_line(0,F(1,2)),_line(F(1,2)+e,1)])


def test_equal_float_approximations_are_not_source_provenance():
    path = ((0.,)*4,(1.,)*4)
    assert not source_path_covered(path,[path])


def test_global_source_path_mapping_preserves_fractional_preimages():
    bounds = ((1e16,1e16+4.),)*4
    first = map_source_path(_line(0,F(1,8)),bounds)
    second = map_source_path(_line(F(1,8)+F(1,2**60),F(1,2)),bounds)
    assert float(first[-1][0]) == float(second[0][0])
    assert first[-1][0] < second[0][0]
    assert not source_path_covered(map_source_path(_line(0,F(1,2)),bounds),[first,second])


def test_denied_source_path_proof_returns_unknown_before_clipping(monkeypatch):
    from mmcore.numeric.intersection.ssx import _ssx_arc_ownership as module
    monkeypatch.setattr(module,'_covered_fraction',lambda *a,**k: (_ for _ in ()).throw(AssertionError('unpaid proof')))
    assert source_path_covered(_line(0,1),[_line(0,1)],charge=lambda _:False) is None
