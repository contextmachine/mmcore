"""Regression: the single closed SSI loop of two toroidal NURBS surfaces.

Two rational toroidal surfaces meeting in ONE closed transversal loop of
length ~18.6146. Both defects this fixture caught were assembly-side
predicates that priced a chord as if it were the curve:

  * at atol=1e-3 the valley-fiction filter deleted a genuine arc (the
    branch came back OPEN, short by exactly the deleted 0.8124), because
    it measured `res / sin_ang` at a chord's parametric midpoint — a
    sagitta-scale quantity — against the same 2*atol the marcher is
    allowed to spend on sagitta. Every chord of the deleted arc crossed at
    sin_ang = 0.9996 (~87 deg): maximally transversal, no valley at all.
    The loss was non-monotonic (2e-3 and 5e-4 both fine) and flipped when
    the two surfaces were swapped.

  * at atol=1e-4 the fragment containment dedup missed a true duplicate,
    because the keeper's polyline opened with a chord the step controller
    never sized: the displaced-seed recovery splices the registered
    crossing onto a march begun `alpha` of the parameter box away, and
    `alpha` is a bare fraction, so that chord stays 0.15099 long with a
    9.620e-4 sagitta at EVERY atol (0.48x sag_tol at 1e-3, 4.81x at 1e-4,
    48.1x at 1e-5). The shared junction became a degree-3 node and the
    chain walker returned along the duplicate reversed, yielding an
    out-and-back branch LONGER than the loop (19.6153 against 18.6143).

Both symptoms are invisible to a "did we get one branch" check: the
branch count is 1 in every case. The invariants that discriminate are
CLOSURE, ARC LENGTH, and the absence of a self-retrace.
"""
import numpy as np
import pytest

from mmcore.nurbs._nurbs_eval import NURBSSurfaceTuple, evaluate_nurbs_surface
from mmcore.numeric.intersection.ssx import nurbs_ssx

# Converged reference (atol=1e-5, both orderings agree to 1e-8).
LOOP_ARCLEN = 18.61455


def _toroid_1():
    return NURBSSurfaceTuple(
        order_u=3, order_v=3,
        knot_u=np.array([0., 0., 0., 7.85398163, 7.85398163,
                         15.70796327, 15.70796327, 23.5619449, 23.5619449,
                         31.41592654, 31.41592654, 31.41592654]),
        knot_v=np.array([-7.85398163, -7.85398163, -7.85398163, 0., 0.,
                         7.85398163, 7.85398163, 7.85398163]),
        control_points=np.array([
            [[-15.72556499, 7.22471728, -4.82962913],
             [-13.56640397, 11.66714207, -4.053172],
             [-14.84083898, 11.44242479, 0.77645714],
             [-16.11527399, 11.21770751, 5.60608627],
             [-18.27443501, 6.77528272, 4.82962913]],
            [[-15.72556499, 7.22471728, -4.82962913],
             [-17.89235348, 13.95064029, -5.08844818],
             [-19.16678849, 13.72592301, -0.25881905],
             [-20.4412235, 13.50120573, 4.57081009],
             [-18.27443501, 6.77528272, 4.82962913]],
            [[-15.72556499, 7.22471728, -4.82962913],
             [-20.05151449, 9.5082155, -5.86490531],
             [-21.3259495, 9.28349822, -1.03527618],
             [-22.60038451, 9.05878094, 3.79435295],
             [-18.27443501, 6.77528272, 4.82962913]],
            [[-15.72556499, 7.22471728, -4.82962913],
             [-22.21067551, 5.06579071, -6.64136245],
             [-23.48511052, 4.84107343, -1.81173332],
             [-24.75954553, 4.61635615, 3.01789582],
             [-18.27443501, 6.77528272, 4.82962913]],
            [[-15.72556499, 7.22471728, -4.82962913],
             [-17.88472601, 2.78229249, -5.60608627],
             [-19.15916102, 2.55757521, -0.77645714],
             [-20.43359603, 2.33285793, 4.053172],
             [-18.27443501, 6.77528272, 4.82962913]],
            [[-15.72556499, 7.22471728, -4.82962913],
             [-13.5587765, 0.49879427, -4.57081009],
             [-14.83321151, 0.27407699, 0.25881905],
             [-16.10764652, 0.04935971, 5.08844818],
             [-18.27443501, 6.77528272, 4.82962913]],
            [[-15.72556499, 7.22471728, -4.82962913],
             [-11.39961549, 4.94121906, -3.79435295],
             [-12.6740505, 4.71650178, 1.03527618],
             [-13.94848551, 4.4917845, 5.86490531],
             [-18.27443501, 6.77528272, 4.82962913]],
            [[-15.72556499, 7.22471728, -4.82962913],
             [-9.24045447, 9.38364385, -3.01789582],
             [-10.51488948, 9.15892657, 1.81173332],
             [-11.78932449, 8.93420929, 6.64136245],
             [-18.27443501, 6.77528272, 4.82962913]],
            [[-15.72556499, 7.22471728, -4.82962913],
             [-13.56640397, 11.66714207, -4.053172],
             [-14.84083898, 11.44242479, 0.77645714],
             [-16.11527399, 11.21770751, 5.60608627],
             [-18.27443501, 6.77528272, 4.82962913]]]),
        weights=np.array([[1., 0.70710678, 1., 0.70710678, 1.],
                          [0.70710678, 0.5, 0.70710678, 0.5, 0.70710678]]
                         * 4 + [[1., 0.70710678, 1., 0.70710678, 1.]]))


def _toroid_2():
    return NURBSSurfaceTuple(
        order_u=3, order_v=3,
        knot_u=np.array([0., 0., 0., 5.6635867, 5.6635867,
                         11.3271734, 11.3271734, 16.9907601, 16.9907601,
                         22.6543468, 22.6543468, 22.6543468]),
        knot_v=np.array([-5.6635867, -5.6635867, -5.6635867, 0., 0.,
                         5.6635867, 5.6635867, 5.6635867]),
        control_points=np.array([
            [[-9.47622819, 6., -3.2677392], [-6.75730483, 8., -1.99988441],
             [-8.28107664, 8., 1.26785479], [-9.80484845, 8., 4.53559398],
             [-12.52377181, 6., 3.2677392]],
            [[-9.47622819, 6., -3.2677392], [-8.5699204, 11., -2.84512094],
             [-10.09369221, 11., 0.42261826], [-11.61746403, 11., 3.69035746],
             [-12.52377181, 6., 3.2677392]],
            [[-9.47622819, 6., -3.2677392], [-11.28884376, 9., -4.11297572],
             [-12.81261557, 9., -0.84523652], [-14.33638739, 9., 2.42250267],
             [-12.52377181, 6., 3.2677392]],
            [[-9.47622819, 6., -3.2677392], [-14.00776712, 7., -5.38083051],
             [-15.53153894, 7., -2.11309131], [-17.05531075, 7., 1.15464789],
             [-12.52377181, 6., 3.2677392]],
            [[-9.47622819, 6., -3.2677392], [-12.19515155, 4., -4.53559398],
             [-13.71892336, 4., -1.26785479], [-15.24269517, 4., 1.99988441],
             [-12.52377181, 6., 3.2677392]],
            [[-9.47622819, 6., -3.2677392], [-10.38253597, 1., -3.69035746],
             [-11.90630779, 1., -0.42261826], [-13.4300796, 1., 2.84512094],
             [-12.52377181, 6., 3.2677392]],
            [[-9.47622819, 6., -3.2677392], [-7.66361261, 3., -2.42250267],
             [-9.18738443, 3., 0.84523652], [-10.71115624, 3., 4.11297572],
             [-12.52377181, 6., 3.2677392]],
            [[-9.47622819, 6., -3.2677392], [-4.94468925, 5., -1.15464789],
             [-6.46846106, 5., 2.11309131], [-7.99223288, 5., 5.38083051],
             [-12.52377181, 6., 3.2677392]],
            [[-9.47622819, 6., -3.2677392], [-6.75730483, 8., -1.99988441],
             [-8.28107664, 8., 1.26785479], [-9.80484845, 8., 4.53559398],
             [-12.52377181, 6., 3.2677392]]]),
        weights=np.array([[1., 0.70710678, 1., 0.70710678, 1.],
                          [0.70710678, 0.5, 0.70710678, 0.5, 0.70710678]]
                         * 4 + [[1., 0.70710678, 1., 0.70710678, 1.]]))


def _sole_branch(a, b, atol):
    res = nurbs_ssx(a, b, atol=atol)
    branches = res['branches']
    assert len(branches) == 1, (
        f"expected the single closed loop, got {len(branches)} branches")
    return branches[0], np.asarray(branches[0].curve[1], dtype=np.float64)


def _arclen(xyz):
    return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())


@pytest.mark.parametrize("atol", [1e-2, 1e-3, 1e-4])
@pytest.mark.parametrize("swap", [False, True], ids=["s1s2", "s2s1"])
def test_toroidal_loop_is_closed(atol, swap):
    """One CLOSED loop, at every atol and in either argument order.

    Before the fix this returned closed=False at atol=1e-3 (s1,s2 only)
    and at atol=1e-4 (both orders).
    """
    s1, s2 = _toroid_1(), _toroid_2()
    a, b = (s2, s1) if swap else (s1, s2)
    branch, xyz = _sole_branch(a, b, atol)

    assert branch.closed, "the loop must close"
    assert float(np.linalg.norm(xyz[0] - xyz[-1])) <= 2.0 * atol


@pytest.mark.parametrize("atol", [1e-2, 1e-3, 1e-4])
@pytest.mark.parametrize("swap", [False, True], ids=["s1s2", "s2s1"])
def test_toroidal_loop_arclength(atol, swap):
    """Neither short (deleted arc) nor long (retraced duplicate).

    A chord polyline underestimates arc length, so only the short side is
    tolerance-dependent; any excess is a retrace. The two historical
    failures sat at 17.7980 (-0.82) and 19.6153 (+1.00).
    """
    s1, s2 = _toroid_1(), _toroid_2()
    a, b = (s2, s1) if swap else (s1, s2)
    _, xyz = _sole_branch(a, b, atol)
    length = _arclen(xyz)

    assert length <= LOOP_ARCLEN + 1e-3, (
        f"branch is LONGER than the loop ({length:.5f} > {LOOP_ARCLEN:.5f}): "
        f"a duplicate arc was traversed twice")
    assert length >= LOOP_ARCLEN - 0.05, (
        f"branch is SHORTER than the loop ({length:.5f}): an arc was lost")


def _retraced_span_pairs(xyz, atol, closed):
    """Find nearby parallel spans separated by actual curve travel.

    Index distance changes when an ordinary forward step is subdivided.
    Cumulative arc distance and segment projection do not. Midpoint probes
    also detect a two-edge out-and-back whose only vertices are its turns.
    Closed paths use cyclic travel, so the closure seam is locally adjacent.
    """
    points = np.asarray(xyz, dtype=float)
    if len(points) < 2:
        return []
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 0.]
    points = points[keep]
    if len(points) < 2:
        return []
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    directions = segments/lengths[:, None]
    distance = np.r_[0., np.cumsum(lengths)]
    probes = np.concatenate((points[:-1], .5*(points[:-1]+points[1:]), points[-1:]))
    positions = np.r_[distance[:-1], .5*(distance[:-1]+distance[1:]), distance[-1]]
    tangents = np.concatenate((directions, directions, directions[-1:]))
    hits = []
    for probe, position, tangent in zip(probes, positions, tangents):
        fraction = np.clip(np.sum((probe-points[:-1])*segments, axis=1)/lengths**2, 0., 1.)
        projected = points[:-1]+fraction[:, None]*segments
        separation = np.abs(position-(distance[:-1]+fraction*lengths))
        if closed:
            separation = np.minimum(separation, distance[-1]-separation)
        nearby = np.linalg.norm(projected-probe, axis=1) < 4.*atol
        nonlocal_span = separation > 8.*atol
        parallel = np.abs(directions@tangent) > .95
        for segment in np.flatnonzero(nearby & nonlocal_span & parallel):
            hits.append((float(position), int(segment)))
            if len(hits) >= 8:
                return hits
    return hits


def _subdivide_polyline(points, count):
    points = np.asarray(points)
    factors = np.arange(count)/count
    return np.vstack([*(a+factors[:, None]*(b-a) for a, b in zip(points[:-1], points[1:])),
                       points[-1]])


@pytest.mark.parametrize('closed', [False, True])
def test_retrace_detector_ignores_forward_densification(closed):
    angle = np.linspace(0., 2.*np.pi, 17)
    points = (np.column_stack((np.cos(angle), np.sin(angle), np.zeros(17))) if closed
              else np.array([[0., 0., 0.], [.3, 0., 0.], [1., 0., 0.]]))
    if closed:
        points[-1] = points[0]
    assert _retraced_span_pairs(points, .02, closed) == []
    assert _retraced_span_pairs(_subdivide_polyline(points, 32), .02, closed) == []


@pytest.mark.parametrize('backtracking', [False, True])
def test_retrace_detector_rejects_repeated_traversal(backtracking):
    if backtracking:
        points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 0., 0.]])
    else:
        angle = np.linspace(0., 4.*np.pi, 33)
        points = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(33)))
        points[-1] = points[0]
    assert _retraced_span_pairs(points, .02, True)
    assert _retraced_span_pairs(_subdivide_polyline(points, 8), .02, True)


@pytest.mark.parametrize("atol", [1e-3, 1e-4])
def test_toroidal_loop_has_no_retrace(atol):
    """No spatially repeated span after accounting for adaptive sampling."""
    branch, xyz = _sole_branch(_toroid_1(), _toroid_2(), atol)
    hits = _retraced_span_pairs(xyz, atol, branch.closed)
    assert not hits, f"retraced spans (arc position, segment): {hits}"


def _surface_point(surf, a, b):
    return np.asarray(
        evaluate_nurbs_surface(surf, float(a), float(b), d_order=0)['S'],
        dtype=np.float64)


def test_sagitta_oracle_measures_intersection_not_surface_midpoint_average():
    # The SSI is (s-.5, .5, (s-.5)**2). Its endpoint chord has z=.25,
    # so its exact middle sagitta is .25. Averaging the surface images
    # at the parameter midpoint would incorrectly return only .125.
    ku = np.array([0., 0., 0., 1., 1., 1.])
    kv = np.array([0., 0., 1., 1.])
    controls = np.array([[[x, t, z] for t in (0., 1.)]
                         for x, z in zip((-.5, 0., .5), (.25, -.25, .25))])
    a = NURBSSurfaceTuple(3, 2, ku, kv, controls, np.ones((3, 2)))
    plane = np.array([[[u-.5, .5, v] for v in (0., 1.)] for u in (0., 1.)])
    b = NURBSSurfaceTuple(2, 2, kv, kv, plane, np.ones((2, 2)))
    stuv = np.array([[0., .5, 0., .25], [1., .5, 1., .25]])
    xyz = np.array([[-.5, .5, .25], [.5, .5, .25]])
    sag, chord = _worst_chord_sagitta(a, b, stuv, xyz)
    assert np.isclose(sag, .25, atol=1e-12, rtol=0.)
    assert chord == 1.


def _worst_chord_sagitta(surf1, surf2, stuv, xyz, atol=1e-3):
    """Sample actual SSI points in three normal planes through each chord.

    This oracle solves the global NURBS equations independently of the SSX
    marcher. The fourth equation pins the physical point to a chord-normal
    plane, so averaging unrelated points on the two surfaces cannot hide
    or exaggerate the curve's deviation.
    """
    from scipy.optimize import root
    bounds = np.array((*surf1.interval(), *surf2.interval()))
    domain_error = 64*np.finfo(float).eps*np.maximum(1., np.max(np.abs(bounds), axis=1))
    residual_limit = max(atol*1e-6,
                         256*np.finfo(float).eps*max(1., float(np.max(np.abs(xyz)))))
    worst, worst_chord = 0.0, 0.0
    for k in range(len(xyz) - 1):
        a, b = xyz[k], xyz[k + 1]
        ab = b - a
        length = float(np.linalg.norm(ab))
        if length <= residual_limit:
            continue
        normal = ab/length
        for fraction in (.25, .5, .75):
            origin = a+fraction*ab

            def equations(parameters):
                first = _surface_point(surf1, *parameters[:2])
                second = _surface_point(surf2, *parameters[2:])
                return np.r_[first-second, np.dot(first-origin, normal)]

            def jacobian(parameters):
                first = evaluate_nurbs_surface(surf1, *parameters[:2], d_order=1)
                second = evaluate_nurbs_surface(surf2, *parameters[2:], d_order=1)
                ds, dt = np.asarray(first['Su']), np.asarray(first['Sv'])
                du, dv = np.asarray(second['Su']), np.asarray(second['Sv'])
                return np.vstack((np.column_stack((ds, dt, -du, -dv)),
                                  [np.dot(ds, normal), np.dot(dt, normal), 0., 0.]))

            guess = stuv[k]+fraction*(stuv[k+1]-stuv[k])
            solution = root(equations, guess, jac=jacobian,
                            options={'xtol': 1e-10, 'maxfev': 80})
            assert np.all(np.isfinite(solution.x)), (k, fraction, solution.message)
            residual = float(np.linalg.norm(equations(solution.x)))
            assert residual <= residual_limit, (k, fraction, residual, solution.message)
            assert np.all(solution.x >= bounds[:, 0]-domain_error)
            assert np.all(solution.x <= bounds[:, 1]+domain_error)
            singular = np.linalg.svd(jacobian(solution.x), compute_uv=False)
            assert singular[-1] > 256*np.finfo(float).eps*singular[0], (k, fraction)
            curve_pt = _surface_point(surf1, *solution.x[:2])
            sag = float(np.linalg.norm(curve_pt-origin))
            # Exclude a jump to another distant root of the same normal
            # plane. Nearby continuation is required before judging sagitta.
            assert sag <= max(length, 4*residual_limit), (k, fraction, sag, length)
            if sag > worst:
                worst, worst_chord = sag, length
    return worst, worst_chord


@pytest.mark.parametrize("atol", [1e-3, 1e-4, 1e-5])
def test_branch_chords_honour_advertised_sagitta(atol):
    """Every delivered chord must sit within `sag_tol = 2*atol` of the curve.

    This is the marcher's own contract, and every downstream geometric
    predicate in units of atol depends on it. The historical displaced-seed
    recovery broke it by splicing the registered crossing onto a march begun
    `alpha` of the PARAMETER BOX away, and `alpha` is a bare fraction, so
    the chord stays 0.15099 long at every tolerance while `sag_tol`
    shrank.

    The oracle evaluates actual intersection points, allowing them to
    cross internal patch cuts while remaining in the original knot domains.
    The historical 1.5 multiplier on the advertised allowance is retained.
    """
    branch, xyz = _sole_branch(_toroid_1(), _toroid_2(), atol)
    stuv = np.asarray(branch.curve[0], dtype=np.float64)
    sag, chord = _worst_chord_sagitta(_toroid_1(), _toroid_2(), stuv, xyz, atol)

    assert sag <= 1.5 * (2.0 * atol), (
        f"chord of length {chord:.5f} deviates {sag:.3e} from the curve, "
        f"{sag / (2.0 * atol):.2f}x the advertised sag_tol={2.0 * atol:.1e}")


@pytest.mark.parametrize("atol", [1e-2, 1e-3, 1e-4])
def test_toroidal_loop_is_order_symmetric(atol):
    """ssx(s1, s2) and ssx(s2, s1) must describe the same loop.

    The reported asymmetry was total at atol=1e-3: open (17.7980) one way,
    closed (18.6130) the other.
    """
    s1, s2 = _toroid_1(), _toroid_2()
    _, fwd = _sole_branch(s1, s2, atol)
    _, rev = _sole_branch(s2, s1, atol)

    assert abs(_arclen(fwd) - _arclen(rev)) <= 4.0 * atol
