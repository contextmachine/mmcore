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


@pytest.mark.parametrize("atol", [1e-3, 1e-4])
def test_toroidal_loop_has_no_retrace(atol):
    """No non-adjacent self-coincidence beyond the closure vertex.

    The atol=1e-4 defect appended a reversed copy of an already-traced
    span, so interior samples coincided pairwise with earlier ones.
    """
    branch, xyz = _sole_branch(_toroid_1(), _toroid_2(), atol)
    n = len(xyz)
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    idx = np.arange(n)
    non_adjacent = np.abs(idx[:, None] - idx[None, :]) > 3
    coincident = np.triu(non_adjacent & (d < 4.0 * atol))
    # A closed branch repeats its first vertex last; that is the only
    # legitimate non-adjacent coincidence.
    coincident[0, n - 1] = False

    assert not coincident.any(), (
        f"retraced span: coincident sample pairs "
        f"{list(zip(*np.where(coincident)))[:8]}")


def _surface_point(surf, a, b):
    return np.asarray(
        evaluate_nurbs_surface(surf, float(a), float(b), d_order=0)['S'],
        dtype=np.float64)


def _worst_chord_sagitta(surf1, surf2, stuv, xyz):
    """Largest distance from the curve to a chord, over all chords.

    The curve point at a chord's parametric midpoint is estimated as the
    midpoint of the two surfaces' images there; each image lies within a
    sagitta of the curve, so their average is correct to second order.
    """
    worst, worst_chord = 0.0, 0.0
    for k in range(len(xyz) - 1):
        mid = 0.5 * (stuv[k] + stuv[k + 1])
        curve_pt = 0.5 * (_surface_point(surf1, mid[0], mid[1])
                          + _surface_point(surf2, mid[2], mid[3]))
        a, b = xyz[k], xyz[k + 1]
        ab = b - a
        den = float(np.dot(ab, ab))
        tt = float(np.clip(np.dot(curve_pt - a, ab) / den, 0.0, 1.0)) \
            if den > 1e-30 else 0.0
        sag = float(np.linalg.norm(a + tt * ab - curve_pt))
        if sag > worst:
            worst, worst_chord = sag, float(np.linalg.norm(ab))
    return worst, worst_chord


@pytest.mark.parametrize("atol", [
    1e-3,
    1e-4,
    pytest.param(1e-5, marks=pytest.mark.xfail(
        strict=True,
        reason="KNOWN: the displaced-seed splice short-circuits an arc that "
               "leaves the cell (v < 0), so no in-cell polyline can follow "
               "it — measured 4.33x sag_tol. Not a sampling-density defect; "
               "see the note below. Flip to a plain param when the "
               "cell-ownership fix lands.")),
])
def test_branch_chords_honour_advertised_sagitta(atol):
    """Every delivered chord must sit within `sag_tol = 2*atol` of the curve.

    This is the marcher's own contract, and every downstream geometric
    predicate in units of atol depends on it. The displaced-seed recovery
    breaks it: it splices the registered crossing onto a march begun
    `alpha` of the PARAMETER BOX away, and `alpha` is a bare fraction, so
    the chord stays 0.15099 long at every tolerance while `sag_tol`
    shrinks.

    Refining that splice is NOT possible from inside the cell, and this is
    why the obvious fix does not work: the recovery only fires on a GRAZE,
    and at a graze the arc between the crossing and the seed leaves the
    cell box (measured here: the true curve sits at v ~ -0.0039, outside).
    `_ssx_correct` clamps to [0,1]^4 by construction, so every interpolated
    interior vertex clamps to the face and stalls at a ~1e-4 residual, far
    above `strict_root_tol`. A curvature-sized subdivision was implemented
    and measured across the SSX suite: 49 prepend calls, 0 subdivided, 49
    fell back. The dip belongs to the NEIGHBOURING cell; the splice trades
    geometric accuracy for connectivity, and only a cell-ownership change
    can retire it.

    The allowance above 1.0 covers the second-order curve-point estimate
    and the marcher's right to spend the budget exactly.
    """
    _, xyz = _sole_branch(_toroid_1(), _toroid_2(), atol)
    stuv = np.asarray(_sole_branch(_toroid_1(), _toroid_2(), atol)[0].curve[0],
                      dtype=np.float64)
    sag, chord = _worst_chord_sagitta(_toroid_1(), _toroid_2(), stuv, xyz)

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
