"""The user's three bilinear L-junction cases, VERBATIM (2026-07-12).

Ledger L57/L61. Preserved in the exact NURBSSurfaceTuple form the user
posted — including the non-unit KNOT VECTORS, which the normalized test
fixtures (tests/test_csx_overlap_tier.py) deliberately drop because
single-span nets are Bezier-identical. If the user's still-failing run
(L61) goes through a NURBS-level driver, the knots and the stuv->knot
parameter mapping are the prime suspects, and this file is the ground
truth for them.

History: cases 1 and 2 lost most of one branch (37%/11% survived, ledger
L57 — root-caused to the missing curve-on-surface overlap certification,
fixed by L59/L60); case 3 always worked (its planar quad is a
PARALLELOGRAM: twist = P11-P10-P01+P00 = 0, so the exact-affine
certificate applied). The twist discriminator is what identified the
parameterization as the problem.

TRUTH for cases 1-2: val2's z(u,v) = 2.89525681*u*(1-v); zero set = the
u=0 edge UNION the v=1 edge of val2, meeting at val2's (0,1) corner where
grad z = 0 -> TWO full branches (xyz length ~9.763 each) joined at ONE
tangent_point, complete=True.
"""
import numpy as np

from mmcore.nurbs._nurbs_eval import NURBSSurfaceTuple

# --- Case 1 (user: "the second branch is not completely found") ---------
case1_val1 = NURBSSurfaceTuple(
    order_u=2, order_v=2,
    knot_u=np.array([0.0, 0.0, 17.82890302, 17.82890302]),
    knot_v=np.array([0.0, 0.0, 17.82890302, 17.82890302]),
    control_points=np.array([
        [[28.73565361, -57.3828431, 0.0], [41.34259183, -50.11361956, 0.0]],
        [[41.34259183, -75.32749601, 0.0], [53.84239759, -62.72055778, 0.0]]]),
    weights=np.array([[1.0, 1.0], [1.0, 1.0]]))

case1_val2 = NURBSSurfaceTuple(
    order_u=2, order_v=2,
    knot_u=np.array([0.0, 0.0, 9.76292344, 9.76292344]),
    knot_v=np.array([0.0, 0.0, 9.76292344, 9.76292344]),
    control_points=np.array([
        [[35.58090097, -65.90568734, 0.0], [38.10773149, -56.47542745, 0.0]],
        [[45.01116086, -68.43251786, 2.89525681], [47.53799138, -59.00225797, 0.0]]]),
    weights=np.array([[1.0, 1.0], [1.0, 1.0]]))

# --- Case 2 (user: "the second branch is simply lost; an isolated point
# with an enormous |S1(s,t)-S2(u,v)| was returned instead" — the invalid
# point did NOT reproduce at HEAD 1efe8c2; keep checking residual validity)
case2_val1 = NURBSSurfaceTuple(
    order_u=2, order_v=2,
    knot_u=np.array([0.0, 0.0, 17.82890302, 17.82890302]),
    knot_v=np.array([0.0, 0.0, 17.82890302, 17.82890302]),
    control_points=np.array([
        [[28.73565361, -62.66611014, 0.0], [41.34259183, -50.11361956, 0.0]],
        [[41.34259183, -75.32749601, 0.0], [53.84239759, -62.72055778, 0.0]]]),
    weights=np.array([[1.0, 1.0], [1.0, 1.0]]))

case2_val2 = case1_val2

# --- Case 3 (user: "works perfectly fine, even though it looks the same")
case3_val1 = case1_val2      # the lifted bilinear, passed FIRST here

case3_val2 = NURBSSurfaceTuple(
    order_u=2, order_v=2,
    knot_u=np.array([0.0, 0.0, 24.46650685, 24.46650685]),
    knot_v=np.array([0.0, 0.0, 23.97376664, 23.97376664]),
    control_points=np.array([
        [[46.82101, -80.30032742, 0.0], [26.05911906, -68.31344409, 0.0]],
        [[59.05426342, -59.11171094, 0.0], [38.29237249, -47.12482762, 0.0]]]),
    weights=np.array([[1.0, 1.0], [1.0, 1.0]]))

CASES = {
    1: (case1_val1, case1_val2),
    2: (case2_val1, case2_val2),
    3: (case3_val1, case3_val2),
}


def run_bez_level(case):
    """Bezier-level run (normalized [0,1] parameters; single-span nets)."""
    from mmcore.numeric.intersection.ssx._bez_ssx5 import bez_ssx
    v1, v2 = CASES[case]
    return bez_ssx(np.asarray(v1.control_points, dtype=float),
                   np.asarray(v2.control_points, dtype=float),
                   1e-3, rational=False)


if __name__ == "__main__":
    for case in (1, 2, 3):
        r = run_bez_level(case)
        print(f"--- case {case}: branches={len(r['branches'])} "
              f"sing={[g.kind for g in r['singularities']]} "
              f"complete={r['complete']} reasons={r['status']['reasons']}")
        for bi, b in enumerate(r["branches"]):
            xyz = np.asarray(b.curve[1])
            seg = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
            print(f"    branch {bi}: {len(xyz)} pts len={seg:.3f}")
