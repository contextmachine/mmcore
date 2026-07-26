"""GJK soundness contract (cluster 3, 2026-07-26).

The engine prunes cells on `if not gjk(hullA, hullB): skip`.  A surface lies
inside its control hull, so hull-disjoint => surface-disjoint: a CORRECT
"separated" can never delete an intersection.  The only dangerous verdict is
a FALSE "separated" on hulls that actually overlap, and these tests pin that
direction, plus the prune strength that makes the primitive worth calling.

Two defects fixed here, both measured before the fix:

  1. `handleSimplex` compared dot products (LENGTH^2) against the caller's
     `tol` (a LENGTH), so the verdict depended on the model's size: for
     geometry of extent s the tests are ~s^2, and once s^2 fell under tol
     every Voronoi test took the wrong branch and the driver exhausted into
     "separated".  Measured cliff: extent <= ~10*tol.  (The tetrahedron case
     always used the scale-invariant `> 0`, which is why it was partial.)
  2. Iteration exhaustion returned "separated" -- unknown reported as a
     definite negative.  Measured 200/200 false negatives at max_iter 1-2.
"""
import numpy as np
import pytest

from mmcore.numeric.algorithms.cygjk import gjk

BOX = np.array([[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
TET = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])


def _gjk(a, b, tol=1e-6, max_iter=25):
    return bool(gjk(np.ascontiguousarray(np.asarray(a, dtype=float)),
                    np.ascontiguousarray(np.asarray(b, dtype=float)),
                    tol, max_iter))


# ---------------------------------------------------------------------------
# SOUNDNESS: overlapping hulls must never be reported separated
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("s", [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8])
def test_overlap_detected_independently_of_geometry_scale(s):
    """The verdict must not depend on the model's size relative to `tol`.

    Regression pin for defect 1: at tol=1e-3 every s <= 1e-2 used to report
    these deeply overlapping boxes as separated.
    """
    assert _gjk(BOX * s, BOX * s + 0.5 * s, tol=1e-3, max_iter=15)


@pytest.mark.parametrize("max_iter", [1, 2, 3, 5, 15, 25])
def test_exhaustion_never_reports_separated(max_iter):
    """Regression pin for defect 2: unknown is not a negative.

    Boxes overlapping by half their extent, with the iteration budget
    starved. Exhaustion must fall back to "not separated" so a caller's
    `if not gjk(...): prune` cannot delete the cell.
    """
    assert _gjk(BOX, BOX + 0.5, max_iter=max_iter)


@pytest.mark.parametrize("T", [0.0, 1.0, 1e3, 1e6, 1e9])
@pytest.mark.parametrize("k", [1e-6, 1e-3, 1.0, 1e3, 1e6])
def test_overlap_verdict_is_similarity_invariant(T, k):
    """Same geometry, moved and scaled, must give the same answer."""
    assert _gjk(BOX * k + T, BOX * k + T + 0.5 * k)


def test_degenerate_overlaps_are_detected():
    """Zero-volume and near-zero-volume overlaps still count."""
    quad = np.array([[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    seg = np.array([[0., 0, 0], [1, 0, 0]])
    needle = np.array([[0., 0, 0], [1, 0, 0], [1, 1e-9, 0], [0, 1e-9, 0]])
    assert _gjk(quad, quad + np.array([0.5, 0.5, 0.0]))      # coplanar
    assert _gjk(seg, seg + np.array([0.5, 0.0, 0.0]))        # collinear
    assert _gjk(BOX, needle + np.array([0.2, 0.5, 0.5]))     # needle in box
    assert _gjk(BOX, BOX)                                    # identical
    assert _gjk(TET, TET + 0.25)


def test_exact_contact_is_not_separated():
    """Touching hulls share a point, so they are not separated."""
    rng = np.random.default_rng(11)
    for _ in range(200):
        t = rng.normal(size=3) * 10.0
        assert _gjk(BOX + t, BOX + t + np.array([1.0, 0.0, 0.0]))   # face
        assert _gjk(BOX + t, BOX + t + np.array([1.0, 1.0, 1.0]))   # vertex


# ---------------------------------------------------------------------------
# STRENGTH: the fix must not turn the prune off
# ---------------------------------------------------------------------------

def test_disjoint_hulls_are_still_separated():
    """Anti-loosening guard.

    A conservative primitive that never separates is sound and useless: the
    engine calls this to prune. Genuinely disjoint hulls must still prune at
    every world position.
    """
    rng = np.random.default_rng(5)
    for _ in range(300):
        t = rng.normal(size=3) * 10.0
        assert not _gjk(BOX + t, BOX + t + np.array([5.0, 0.0, 0.0]))


@pytest.mark.parametrize("gap", [1e-3, 1e-1, 1.0, 10.0])
@pytest.mark.parametrize("T", [0.0, 1e3, 1e6])
def test_separation_detected_across_gaps_and_positions(gap, T):
    a = BOX + T
    b = BOX + T + np.array([1.0 + gap, 0.0, 0.0])
    assert not _gjk(a, b)
