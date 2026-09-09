import numpy as np

from mmcore.numeric.intersection.ssx._ssx_affine_constraints import AffineParameterConstraints
from mmcore.numeric.intersection.ssx._ssx_root_identity import _pure_affine_coordinates
from mmcore.numeric.intersection.ssx._ssx_singular_candidate import SourceSingularCandidates


def _sources():
    a = np.array([[[s, t, x-y, 1.] for t, y in zip((0., .5, 1.), (1., -1., 1.))]
                  for s, x in zip((0., .5, 1.), (1., -1., 1.))])
    b = np.array([[[u, v, 0., 1.] for v in (-.5, 1.5)] for u in (-.5, 1.5)])
    return a, b


def _generator(a, b, charge=None):
    constraints = AffineParameterConstraints(tuple(_pure_affine_coordinates(s) for s in (a, b)))
    return SourceSingularCandidates(a, b, constraints, charge)


def test_exact_affine_correction_generates_a_source_saddle_root():
    source = _generator(*_sources())
    candidate = np.array([.5, .5, np.nextafter(.5, 1.), np.nextafter(.5, 0.)])
    saved = candidate.copy()
    np.testing.assert_array_equal(source(candidate), np.full(4, .5))
    np.testing.assert_array_equal(candidate, saved)


def test_exact_saddle_has_isolated_regular_chart_delta_root():
    source = _generator(*_sources())
    np.testing.assert_array_equal(source.isolated_regular(np.full(4, .5)), np.full(4, .5))


def test_tangential_curve_samples_are_not_isolated_delta_roots():
    _, plane = _sources()
    source = np.array([[[s, t, height, 1.] for t, height in zip((0., .5, 1.), (1., -1., 1.))]
                       for s in (0., 1.)])
    generator = _generator(source, plane)
    assert generator(np.full(4, .5)) is not None
    assert generator.isolated_regular(np.full(4, .5)) is None


def test_rank_deficient_source_is_not_a_regular_chart_tangent_point():
    source, plane = _sources()
    source[..., 0] = 0.5
    generator = _generator(source, plane)
    assert generator(np.full(4, .5)) is not None
    assert generator.isolated_regular(np.full(4, .5)) is None


def test_degree_zero_chart_rejects_regular_classification_without_empty_jet():
    source, plane = _sources()
    source = np.array([[[.5, .5, 0., 1.], [.5, .5, 0., 1.]]])
    generator = _generator(source, plane)
    assert generator(np.full(4, .5)) is not None
    assert generator.isolated_regular(np.full(4, .5)) is None


def test_affine_relations_do_not_promote_nonzero_source_residual():
    a, b = _sources()
    a[..., 2] += 2.**-40
    assert _generator(a, b)(np.full(4, .5)) is None


def test_a_regular_source_root_is_not_a_delta_root():
    a, b = _sources()
    a = np.array([[[s, t, s-.5, 1.] for t in (0., 1.)] for s in (0., 1.)])
    assert _generator(a, b)(np.full(4, .5)) is None


def test_nonrepresentable_affine_inverse_stays_a_candidate_obligation():
    a, b = _sources()
    b = b.copy()
    b[0, :, 0] = 0.
    b[1, :, 0] = 3.
    assert _generator(a, b)(np.full(4, .5)) is None


def test_denied_source_proof_does_not_publish_a_candidate():
    charges = []
    source = _generator(*_sources(), charge=lambda n: charges.append(n) or False)
    assert source(np.full(4, .5)) is None
    assert len(charges) == 1 and charges[0] > 0
