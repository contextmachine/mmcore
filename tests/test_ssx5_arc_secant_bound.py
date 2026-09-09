"""Whole-arc approximation from a validated velocity interval."""
import numpy as np

from mmcore.numeric.intersection.ssx._ssx_affine_path import (
    SourceArcChordBounds, source_arc_chord_error_bounded, source_box_image_diameter_bounded)


def _parabola():
    plane = np.array([[[s, t, 0.] for t in (0., 1.)] for s in (0., 1.)])
    graph = np.array([[[s, t, t-squared] for t in (0., 1.)]
                      for s, squared in ((0., 0.), (.5, 0.), (1., 1.))])
    endpoints = np.array([[.25, .0625, .25, .0625], [.5, .25, .5, .25]])
    root_boxes = np.stack((endpoints, endpoints), axis=-1)
    box = np.stack((endpoints[0], endpoints[1]), axis=-1)
    xyz = np.column_stack((endpoints[:, :2], np.zeros(2)))
    minors = (np.array([-1., .5, -1., .5]), np.array([-1., 1., -1., 1.]))
    return plane, graph, box, root_boxes, xyz, minors


def test_secant_bound_certifies_arc_much_longer_than_tolerance():
    a, b, box, roots, xyz, minors = _parabola()
    assert not source_box_image_diameter_bounded(a, b, box, xyz, 1/32, rational=False)
    assert source_arc_chord_error_bounded(a, b, box, roots, xyz, 0, minors, 1/32, rational=False)
    # The entire parabola's actual midpoint deviation is 1/64, while
    # the derivative-range certificate conservatively bounds it by 1/32.
    assert not source_arc_chord_error_bounded(a, b, box, roots, xyz, 0, minors, 1/128, rational=False)


def test_published_endpoint_error_is_included_in_secant_bound():
    a, b, box, roots, xyz, minors = _parabola()
    xyz = xyz.copy()
    xyz[:, 2] += .125
    assert not source_arc_chord_error_bounded(a, b, box, roots, xyz, 0, minors, .04, rational=False)


def test_unknown_or_reversed_monotone_parameter_cannot_certify_arc():
    a, b, box, roots, xyz, minors = _parabola()
    unknown = tuple(value.copy() for value in minors)
    unknown[1][0] = 1.
    assert not source_arc_chord_error_bounded(a, b, box, roots, xyz, 0, unknown, 1., rational=False)
    assert not source_arc_chord_error_bounded(a, b, box, roots[::-1], xyz[::-1], 0, minors, 1., rational=False)


def test_secant_source_setup_is_prepaid():
    a, b, box, roots, xyz, minors = _parabola()
    charges = []
    assert not source_arc_chord_error_bounded(a, b, box, roots, xyz, 0, minors, 1., rational=False,
        charge=lambda amount: charges.append(amount) or False)
    assert len(charges) == 1 and charges[0] > 0


def test_original_fraction_jets_are_constructed_once_and_query_work_is_repaid(monkeypatch):
    import mmcore.numeric.intersection.ssx._ssx_affine_path as module

    a, b, box, roots, xyz, minors = _parabola()
    built, charges = [], []
    build = module._exact_source_jet
    def counted(net, rational):
        built.append(net.shape)
        return build(net, rational)
    monkeypatch.setattr(module, '_exact_source_jet', counted)
    source = SourceArcChordBounds(a, b, rational=False, charge=lambda amount: charges.append(amount) or True)
    construction = list(charges)
    assert len(built) == 2 and len(construction) == 1
    assert source.bounded(box, roots, xyz, 0, minors, 1/32)
    first_query = charges[len(construction):]
    assert len(first_query) > 1 and sum(first_query) > 0
    before = len(charges)
    assert source.bounded(box, roots, xyz, 0, minors, 1/32)
    assert charges[before:] == first_query
    assert len(built) == 2


def test_denied_construction_stops_before_fraction_conversion(monkeypatch):
    import mmcore.numeric.intersection.ssx._ssx_affine_path as module

    a, b, box, roots, xyz, minors = _parabola()
    def forbidden(*_):
        raise AssertionError('Unpaid exact source construction')
    monkeypatch.setattr(module, '_exact_source_jet', forbidden)
    source = SourceArcChordBounds(a, b, rational=False, charge=lambda _: False)
    assert source.jets is None and source.exhausted
    assert not source.bounded(box, roots, xyz, 0, minors, 1.)


def test_denied_restriction_stops_before_kernel_and_remains_exhausted(monkeypatch):
    import mmcore.numeric.intersection.ssx._ssx_affine_path as module

    a, b, box, roots, xyz, minors = _parabola()
    charges = []
    def allowance(amount):
        charges.append(amount)
        return len(charges) < 3  # Source construction and query inputs only.
    source = SourceArcChordBounds(a, b, rational=False, charge=allowance)
    def forbidden(*_):
        raise AssertionError('Unpaid exact restriction')
    monkeypatch.setattr(module, '_restrict_homogeneous_rectangle', forbidden)
    assert not source.bounded(box, roots, xyz, 0, minors, 1.)
    assert source.exhausted and len(charges) == 3
    assert not source.bounded(box, roots, xyz, 0, minors, 1.)
    assert len(charges) == 3


def test_cached_source_jets_are_an_immutable_input_snapshot():
    a, b, box, roots, xyz, minors = _parabola()
    source = SourceArcChordBounds(a, b, rational=False)
    a[..., 2] += 1.
    b[..., 2] += 1.
    assert source.bounded(box, roots, xyz, 0, minors, 1/32)
    assert not source_arc_chord_error_bounded(a, b, box, roots, xyz, 0, minors, 1/32, rational=False)
    assert all(not net.flags.writeable for exact, derivatives in source.jets
               for net in (exact, *derivatives) if net is not None)


def test_cached_rational_jet_centering_commutes_with_exact_restriction():
    from fractions import Fraction
    import mmcore.numeric.intersection.ssx._ssx_affine_path as module

    rng = np.random.default_rng(718)
    net = rng.integers(-8, 9, (4, 3, 4)).astype(float)/8.
    net[..., 3] = rng.integers(8, 17, (4, 3))/8.
    origin = (Fraction(2**20), Fraction(-3, 8), Fraction(5, 16))
    start, end = (Fraction(1, 4), Fraction(3, 8)), (Fraction(3, 4), Fraction(7, 8))
    source = SourceArcChordBounds(net, net, rational=True)
    exact, derivatives = source.jets[0]
    centered = exact.copy()
    for k in range(3):
        centered[..., k] -= origin[k]*centered[..., 3]
    for axis, derivative in enumerate(derivatives):
        before = module._restrict_homogeneous_rectangle(
            (centered.shape[axis]-1)*np.diff(centered, axis=axis), start, end)
        after = module._restrict_homogeneous_rectangle(derivative, start, end)
        for k in range(3):
            after[..., k] -= origin[k]*after[..., 3]
        np.testing.assert_array_equal(before, after)
