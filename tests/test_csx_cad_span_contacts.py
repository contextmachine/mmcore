"""CAD coincidence retains the numerical contact seeds needed by SSX."""
from math import comb

import numpy as np
import pytest

import mmcore.numeric.intersection.csx._bez_csx4 as csx
from mmcore.numeric._bezier_common import eval_curve, eval_surface
from mmcore.numeric.intersection.ssx._bez_ssx5 import _cut_face_contacts


def _three_crossings(amplitude):
    power = amplitude * np.polynomial.polynomial.polyfromroots([.2, .5, .8])
    coefficients = np.array([sum(power[k]*comb(i, k)/comb(3, k)
                                 for k in range(i+1)) for i in range(4)])
    curve = np.column_stack((np.linspace(.1, .9, 4), np.full(4, .5), coefficients))
    plane = np.array([[[s, t, 0.] for t in (0., 1.)] for s in (0., 1.)])
    return curve, plane


def _validate_contacts(curve, surface, contacts):
    for point in contacts:
        assert all(0. <= point[key] <= 1. for key in ('t', 'u', 'v'))
        curve_point = eval_curve(curve, point['t'], rational=False)
        surface_point = eval_surface(surface, point['u'], point['v'], rational=False)
        assert np.linalg.norm(curve_point-surface_point) <= 1e-3
        assert np.linalg.norm(point['point']-curve_point) <= 1e-3


def test_full_cad_span_keeps_three_distinct_interior_contact_proposals():
    curve, surface = _three_crossings(.005)
    result = csx.bez_csx(curve, surface, atol=1e-3, rational=False)
    assert not result['budget_exhausted']
    assert len(result['overlaps']) == 1
    assert result['overlaps'][0]['t_range'] == (0., 1.)
    assert result['overlaps'][0]['certification'] == 'tolerance'
    contacts = list(_cut_face_contacts(result))
    for expected in (.2, .5, .8):
        matches = [point for point in contacts if abs(point['t']-expected) <= 1e-3]
        assert matches, (expected, contacts)
        assert all(point['certification'] == 'tolerance' for point in matches)
    _validate_contacts(curve, surface, contacts)


@pytest.mark.parametrize('amplitude, exact_mode', [(1., False), (.005, True)])
def test_isolated_crossings_remain_when_no_cad_span_is_admitted(amplitude, exact_mode):
    curve, surface = _three_crossings(amplitude)
    result = csx.bez_csx(curve, surface, atol=1e-3, rational=False,
                         tolerance_tier=not exact_mode)
    assert result['overlaps'] == []
    assert not result['budget_exhausted']
    assert sorted(point['t'] for point in result['isolated']) == pytest.approx([.2, .5, .8], abs=1e-3)
    _validate_contacts(curve, surface, result['isolated'])


def test_contact_refinement_denial_keeps_the_valid_cad_span(monkeypatch):
    curve, surface = _three_crossings(.005)
    def forbidden(*args, **kwargs):
        raise AssertionError('contact polishing ran after work denial')
    monkeypatch.setattr(csx, '_polish_csx_root', forbidden)
    spans = csx._tolerance_csx_overlap_certificate(
        curve, surface, 1e-3, False, 1e-4, None,
        on_contact_work=lambda amount: False)
    assert len(spans) == 1 and spans[0]['t_range'] == (0., 1.)
    assert spans[0].get('boundary_contacts', []) == []


def test_contact_result_cap_is_reported_without_losing_the_span():
    curve, surface = _three_crossings(.005)
    result = csx.bez_csx(curve, surface, atol=1e-3, rational=False,
                         max_results=3)
    assert len(result['overlaps']) == 1
    assert result['overlaps'][0]['t_range'] == (0., 1.)
    assert result['budget_exhausted']
    assert result['truncation_cause'] == 'results'
    contacts = list(_cut_face_contacts(result))
    assert len(contacts) <= 3
    _validate_contacts(curve, surface, contacts)


@pytest.mark.parametrize('max_results', [3, 10])
def test_overlap_contacts_share_the_cap_with_separate_isolated_roots(max_results):
    # The curve lies within CAD tolerance over its first in-domain span
    # [0,.3], crosses the plane at .15 there, then reenters the patch and
    # has a separate isolated root at .8. All three span contacts remain
    # charged when they move from the isolated list onto the overlap.
    degree = 8
    def coefficients(power):
        power = np.pad(power, (0, degree+1-len(power)))
        return [sum(power[k]*comb(i, k)/comb(degree, k)
                    for k in range(i+1)) for i in range(degree+1)]
    z = np.polynomial.polynomial.polymul(
        [0.]*6+[1.], np.polynomial.polynomial.polyfromroots([.15, .8]))
    curve = np.column_stack((coefficients([.04, 1.8, -2.]),
                             coefficients([.5]), coefficients(z)))
    surface = np.array([[[.4*u, v, 0.] for v in (0., 1.)]
                        for u in (0., 1.)])
    result = csx.bez_csx(curve, surface, atol=1e-3, rational=False,
                         max_results=max_results)
    assert len(result['overlaps']) == 1
    contacts = list(_cut_face_contacts(result))
    assert len(contacts) <= max_results
    _validate_contacts(curve, surface, contacts)
    assert any(abs(point['t']-.15) <= 1e-3 for point in contacts)
    if max_results == 3:
        assert result['budget_exhausted']
        assert result['truncation_cause'] == 'results'
    else:
        assert not result['budget_exhausted']
        assert any(abs(point['t']-.8) <= 1e-3
                   for point in result['isolated'])


def test_denied_cad_span_classification_does_not_report_complete():
    curve = np.array([[0., .5, .0005], [1., .5, .0005]])
    surface = np.array([[[u, v, 0.] for v in (0., 1.)]
                        for u in (0., 1.)])
    result = csx.bez_csx(curve, surface, atol=1e-3, rational=False,
                         max_cells=200)
    # Exact zero exclusion cannot finish a CAD span-classification task
    # which was skipped for lack of work allowance.
    assert result['budget_exhausted']
    assert result['truncation_cause'] == 'cells'
    assert result['cells_processed'] <= 200
    _validate_contacts(curve, surface, list(_cut_face_contacts(result)))


def _narrow_nonaffine_strip(start, reverse=False, bulge=False):
    end = start + .05
    curve = np.array([[0., 0., 0.], [100., 0., 0.]])
    if reverse:
        curve = curve[::-1].copy()
    surface = np.array([[[start+(end-start)*u, v-.5+y, z]
                         for v in (0., 1.)]
                        for u, y, z in zip((0., .5, 1.), (0., 0., .2),
                                           (0., .025 if bulge else 0., 0.))])
    return curve, surface, end


@pytest.mark.parametrize('start,reverse', [(71.23, False), (13.71, False), (71.23, True)])
def test_boundary_contacts_resolve_a_narrow_nonaffine_clipped_span(start, reverse):
    curve, surface, end = _narrow_nonaffine_strip(start, reverse)
    result = csx.bez_csx(curve, surface, atol=1e-3, rational=False)
    assert not result['budget_exhausted']
    assert result['overlaps'], result
    # The true interval is .0005 in curve parameters but 50*atol long
    # in model space. Neither fixed global sampling grid visits it.
    for u in np.linspace(0., 1., 101):
        x = start + (end-start)*u
        t = (100.-x)/100. if reverse else x/100.
        assert any(low-1e-3/100. <= t <= high+1e-3/100.
                   for low, high in (span['t_range'] for span in result['overlaps']))
    for span in result['overlaps']:
        for t in np.linspace(*span['t_range'], 101):
            point = eval_curve(curve, t, rational=False)
            u = float(np.clip((point[0]-start)/(end-start), 0., 1.))
            target = eval_surface(surface, u, .5-.2*u*u, rational=False)
            assert np.linalg.norm(point-target) <= 1e-3


def test_two_boundary_contacts_do_not_promote_an_out_of_tolerance_bulge():
    curve, surface, end = _narrow_nonaffine_strip(71.23, bulge=True)
    result = csx.bez_csx(curve, surface, atol=1e-3, rational=False)
    assert result['overlaps'] == []
    assert not result['budget_exhausted']
    assert len(result['isolated']) == 2
    _validate_contacts(curve, surface, result['isolated'])


@pytest.mark.parametrize('side', [0., 1.])
def test_following_a_surface_edge_does_not_pin_an_interior_tolerance_window(side):
    # These two roots are close in curve parameters but eight CAD atols
    # apart in model space. The entire owner curve follows a target UV
    # edge, which does not delimit the interval between the two roots.
    a, b = .49, .4905
    z = 32.*np.array([a*b, a*b-(a+b)/2., (1.-a)*(1.-b)])
    curve = np.column_stack((np.array([0., 8., 16.]), np.full(3, side), z))
    surface = np.array([[[16.*u, v, 0.] for v in (0., 1.)]
                        for u in (0., 1.)])
    result = csx.bez_csx(curve, surface, atol=1e-3, rational=False)
    assert not result['budget_exhausted']
    assert result['overlaps'] == []
    _validate_contacts(curve, surface, result['isolated'])
    for x in (7.84, 7.848):
        assert any(np.linalg.norm(point['point']-[x, side, 0.]) <= 1e-3
                   for point in result['isolated'])


@pytest.mark.parametrize('side', [0., 1.])
def test_touching_a_domain_edge_without_exiting_does_not_pin_a_span(side):
    # The UV path touches the same edge at .4 and .6 but stays inside on
    # BOTH sides. Its outward probes are only 4e-7 from that edge; treating
    # "UV near zero" as an active clamp incorrectly promoted this window.
    degree = 4
    product = np.polynomial.polynomial.polyfromroots([.4, .6])
    y = np.polynomial.polynomial.polymul(product, product)
    if side:
        y = -y
        y[0] += 1.
    def coefficients(power):
        power = np.pad(power, (0, degree+1-len(power)))
        return [sum(power[k]*comb(i, k)/comb(degree, k)
                    for k in range(i+1)) for i in range(degree+1)]
    curve = np.column_stack((coefficients([0., 16.]), coefficients(y),
                             coefficients(.05*product)))
    surface = np.array([[[16.*u, v, 0.] for v in (0., 1.)]
                        for u in (0., 1.)])
    assert csx._tolerance_csx_overlap_certificate(
        curve, surface, 1e-3, False, 1e-3/16., None,
        parameter_range=(.4, .6)) is None
    result = csx.bez_csx(curve, surface, atol=1e-3, rational=False)
    assert result['overlaps'] == []
    assert not result['budget_exhausted']
    _validate_contacts(curve, surface, result['isolated'])
    for x in (6.4, 9.6):
        assert any(np.linalg.norm(point['point']-[x, side, 0.]) <= 1e-3
                   for point in result['isolated'])


@pytest.mark.parametrize('owner,axis,side', [
    (owner, axis, side) for owner in (0, 1) for axis in (0, 1) for side in (0, 1)])
def test_rounded_spherical_boundary_spans_cover_the_independent_common_rim(owner, axis, side):
    from examples.ssx.case_17 import s1, s2
    from test_ssx_spherical_overlap import _fit_octant_frame, _reference, _homogeneous
    from examples.ssx.ssx5_analytic_audit import distances_to_polylines

    pair = (s1, s2)
    fits = [_fit_octant_frame(surface) for surface in pair]
    center = np.mean([fit[0] for fit in fits], axis=0)
    radius = np.mean([fit[1] for fit in fits])
    reference = _reference(center, radius, [fit[2] for fit in fits])
    sources = [_homogeneous(surface) for surface in pair]
    curve = sources[owner][2*side] if axis == 0 else sources[owner][:, 2*side]
    result = csx.bez_csx(curve, sources[1-owner], atol=1e-3, rational=True)
    assert not result['budget_exhausted']
    normal_index = (1-side) if axis == 0 else (2 if side == 0 else None)
    wanted = []
    if normal_index is not None:
        normal = fits[owner][2][:, normal_index]
        wanted = [arc for arc in reference['arcs']
                  if np.max(abs((arc-center) @ normal)) <= .01*1e-3]
    if not wanted:
        assert result['overlaps'] == [] and result['isolated'] == []
        return
    assert len(result['overlaps']) == 1
    paths = [np.array([eval_curve(curve, t, rational=True)
                       for t in np.linspace(*span['t_range'], 1025)])
             for span in result['overlaps']]
    for arc in wanted:
        assert distances_to_polylines(arc, paths).max() <= 1e-3
    for point in _cut_face_contacts(result):
        a = eval_curve(curve, point['t'], rational=True)
        b = eval_surface(sources[1-owner], point['u'], point['v'], rational=True)
        assert np.linalg.norm(a-b) <= 1e-3
