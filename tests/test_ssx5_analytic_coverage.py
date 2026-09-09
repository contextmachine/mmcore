"""Generated SSX tests with an oracle independent of intersection machinery.

The complete zero set follows from the prescribed polynomial factors.
The audit checks continuous coverage and whole segments, then component
ownership, travel and closure; branch counts alone are insufficient.
The larger circle-product family is run by the companion CLI, whose
per-case subprocess watchdog can report unresolved or expensive cases.
"""
from contextlib import contextmanager
import signal
from types import SimpleNamespace

import numpy as np
import pytest

from examples.ssx.ssx5_analytic_audit import (
    Component, VARIANTS, audit_result, case_components, graph_pair,
    power_to_bernstein, solve_case,
)
from mmcore.numeric._bezier_common import eval_surface


@contextmanager
def _watchdog(seconds=30.):
    """Bound generated integration cases without changing the SSX budgets."""
    if not hasattr(signal, "setitimer"):
        yield
        return
    old_handler = signal.getsignal(signal.SIGALRM)

    def expired(signum, frame):
        raise TimeoutError("independent analytic SSX case exceeded watchdog")

    signal.signal(signal.SIGALRM, expired)
    old_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *old_timer)
        signal.signal(signal.SIGALRM, old_handler)


def _result(polylines, closed=True):
    return {"complete": True, "status": {"reasons": []}, "points": [],
            "singularities": [], "overlap_regions": [],
            "branches": [SimpleNamespace(curve=(np.zeros((len(p), 4)), p),
                                          closed=closed) for p in polylines]}


def _circle(theta):
    return np.column_stack((.5+.25*np.cos(theta), .5+.25*np.sin(theta),
                            np.zeros(len(theta))))


@pytest.mark.parametrize("case", ["one_line", "four_lines", "two_circles", "nested_circles"])
def test_generated_graph_has_the_prescribed_zero_set(case):
    components = case_components(case)
    surface, _ = graph_pair(components)
    for component in components:
        xyz, _ = component.reference(.08)
        for point in xyz:
            evaluated = eval_surface(surface, point[0], point[1], rational=False)
            np.testing.assert_allclose(evaluated, point, atol=3e-14, rtol=0.)
    # Check arbitrary deterministic off-set points as well, against the
    # product formula rather than a second power-to-Bernstein conversion.
    for u, v in ((.13, .19), (.41, .61), (.73, .83)):
        z = 1.
        for component in components:
            if component.kind == "line":
                z *= u-component.data[0]
            else:
                cx, cy, radius = component.data
                z *= (u-cx)**2 + (v-cy)**2-radius**2
        np.testing.assert_allclose(eval_surface(surface, u, v, rational=False),
                                   [u, v, z], atol=3e-14, rtol=0.)


def test_power_conversion_reproduces_known_quadratic():
    np.testing.assert_array_equal(power_to_bernstein(np.array([[1.], [-4.], [4.]])),
                                  np.array([[1.], [-1.], [1.]]))


def test_audit_certifies_full_circle_continuously():
    polyline = _circle(np.linspace(0., 2.*np.pi, 257))
    report = audit_result("one_circle", _result([polyline]))
    assert report["passed"], report
    assert report["components"][0]["coverage_upper"] <= 4e-3


def test_audit_rejects_missing_arc_despite_correct_branch_count():
    polyline = _circle(np.linspace(.3, 2.*np.pi-.3, 257))
    report = audit_result("one_circle", _result([polyline], closed=False))
    assert report["branch_count"] == 1
    assert report["silent_failure"]
    assert report["components"][0]["coverage_lower"] > .07


def test_audit_rejects_full_retrace_despite_full_coverage_and_one_branch():
    polyline = _circle(np.linspace(0., 4.*np.pi, 513))
    report = audit_result("one_circle", _result([polyline]))
    assert report["branch_count"] == 1
    assert report["components"][0]["coverage_upper"] <= 4e-3
    assert report["silent_failure"]
    assert report["components"][0]["angular_travel"] == pytest.approx(4.*np.pi)


def test_audit_rejects_duplicate_component_and_missing_component():
    a = np.array([[.25, 0., 0.], [.25, 1., 0.]])
    report = audit_result("two_lines", _result([a, a.copy()], closed=False))
    assert report["branch_count"] == 2
    assert report["silent_failure"]
    assert [part["branches"] for part in report["components"]] == [2, 0]


def test_soundness_checks_the_chord_interior():
    component = Component("circle", (.5, .5, .25))
    chord = _circle(np.array([0., np.pi]))
    assert component.distance(chord).max() < 1e-14
    assert component.segment_error_bound(chord) == pytest.approx(.25)


@pytest.mark.parametrize("variant", VARIANTS)
def test_two_analytic_lines_survive_parameter_and_knot_changes(variant):
    with _watchdog():
        result = solve_case("two_lines", variant)
        report = audit_result("two_lines", result)
    assert report["passed"], report


@pytest.mark.parametrize("variant", VARIANTS)
def test_analytic_circle_preserves_coverage_closure_and_travel(variant):
    with _watchdog():
        result = solve_case("one_circle", variant)
        report = audit_result("one_circle", result)
    assert report["passed"], report


@pytest.mark.parametrize("variant", VARIANTS)
def test_two_circles_do_not_leave_an_offcurve_stationary_corner_unsearched(variant):
    """Port-directed cuts must not retain the same unresolved interior.

    The gradient vanishes between the two regular components. Repeated
    cuts between their nearby boundary events can approach an extremum
    forever while leaving that unrelated stationary point in every owner.
    Whole-domain subdivision must still find both complete closed loops.
    """
    with _watchdog():
        result = solve_case("two_circles", variant)
        report = audit_result("two_circles", result)
    assert report["passed"], report


@pytest.mark.parametrize("variant", ["identity", "split_u", "split_uv"])
def test_nearby_transverse_lines_survive_exact_knot_insertion(variant):
    """A low-residual nonzero span boundary is not an intersection component.

    Both true roots are separated by 1/128, while the inserted u=.5
    boundary has z=-1/65536.  Treating that boundary as an overlap
    wrongly replaces two full transversal lines with one phantom line.
    """
    with _watchdog():
        result = solve_case("tight_lines", variant)
        report = audit_result("tight_lines", result)
    assert report["passed"], report
