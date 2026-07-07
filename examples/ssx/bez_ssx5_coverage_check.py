"""Objective coverage check for bez_ssx5 cases.

Builds an independent reference point cloud for the intersection by slicing
S1 with N isolines and intersecting each with S2 via bez_csx, then measures
how much of that cloud lies within 5*atol of the SSX branches. Reports
missing parameter ranges (lost fragments) and per-branch point spacing
(rough-tracing indicator).

Usage:
    python examples/ssx/bez_ssx5_coverage_check.py 10 11   # check cases 10, 11
    python examples/ssx/bez_ssx5_coverage_check.py         # check all known cases
"""
import sys

import numpy as np

from mmcore.numeric.bern import de_casteljau_split_nd
from mmcore.numeric.intersection.csx._bez_csx4 import bez_csx
from mmcore.numeric.intersection.ssx._bez_ssx5 import bez_ssx

ALL_CASES = (5, 6, 7, 8, 9, 10, 11)


def _extract_isoline(S, axis, value):
    if axis == 0:
        left, _ = de_casteljau_split_nd(S, axis=0, t=value)
        return left[-1, :, :]
    left, _ = de_casteljau_split_nd(S, axis=1, t=value)
    return left[:, -1, :]


def reference_cloud(S1, S2, atol=1e-3, n=200, axis=0):
    """Slice S1 along `axis`, CSX each isoline against S2."""
    S1h = np.concatenate([S1, np.ones(S1.shape[:-1] + (1,))], axis=-1)
    S2h = np.concatenate([S2, np.ones(S2.shape[:-1] + (1,))], axis=-1)
    pts, params = [], []
    for w in np.linspace(1e-9, 1.0 - 1e-9, n):
        r = bez_csx(_extract_isoline(S1h, axis, float(w)), S2h, atol=atol, rational=True)
        for p in r.get("isolated", []):
            pts.append(np.asarray(p["point"], dtype=float))
            params.append(float(w))
    return np.asarray(pts), np.asarray(params)


def point_to_polyline_dist(p, poly):
    if len(poly) == 1:
        return float(np.linalg.norm(poly[0] - p))
    a, b = poly[:-1], poly[1:]
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    denom = np.where(denom < 1e-30, 1e-30, denom)
    t = np.clip(np.einsum("ij,ij->i", p[None, :] - a, ab) / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return float(np.linalg.norm(proj - p[None, :], axis=1).min())


def load_case_surfaces(case):
    import os
    path = os.path.join(os.path.dirname(__file__), f"bez_ssx5_case{case}.py")
    ns = {}
    exec(compile(open(path).read().split("if __name__")[0], path, "exec"), ns)
    s1 = ns.get("S1", ns.get("s1"))
    s2 = ns.get("S2", ns.get("s2"))
    if s1 is None:
        # Surfaces defined inside the case function: capture via bez_ssx hook.
        captured = {}

        def _capture(a, b, *args, **kw):
            captured["s1"], captured["s2"] = a, b
            return {"branches": [], "points": []}

        ns["bez_ssx"] = _capture
        fn = next(v for k, v in ns.items() if callable(v) and k.startswith("bez_ssx_case"))
        fn()
        s1, s2 = captured["s1"], captured["s2"]
    return np.asarray(s1, dtype=float), np.asarray(s2, dtype=float)


def check_case(case, atol=1e-3, n_ref=200):
    import time
    S1, S2 = load_case_surfaces(case)
    t0 = time.time()
    res = bez_ssx(S1, S2, atol, rational=False)
    dt = time.time() - t0

    polys = []
    print(f"=== case {case}: {dt:.2f}s, {len(res['branches'])} branches, "
          f"{len(res['points'])} points")
    for bi, b in enumerate(res["branches"]):
        xyz = np.asarray(b.curve[1])
        seg = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        seg = seg[seg > 1e-12]
        med = float(np.median(seg)) if len(seg) else 0.0
        mx = float(seg.max()) if len(seg) else 0.0
        print(f"    branch {bi}: {len(xyz):4d} pts, length {float(seg.sum()):.3f}, "
              f"step med/max {med:.4f}/{mx:.4f}")
        polys.append(xyz)

    for g in res.get("singularities", []):
        print(f"    singularity {g.kind}: stuv={np.round(g.stuv, 5).tolist()} "
              f"xyz={np.round(g.xyz, 5).tolist()} links={g.branch_links}")
    # Ledger L24: the regular coverage cases must produce ZERO typed
    # singularities — enforced (exit code), not just printed, so CI
    # catches spurious-singularity regressions.
    clean_singularities = not res.get("singularities", [])
    if not clean_singularities:
        print(f"    SPURIOUS SINGULARITIES: {len(res['singularities'])} (expected 0)")

    ref, ws = reference_cloud(S1, S2, atol=atol, n=n_ref)
    if not len(ref):
        print("    (no reference points)")
        return clean_singularities
    miss_tol = 5 * atol
    dists = np.array([min((point_to_polyline_dist(p, poly) for poly in polys),
                          default=np.inf) for p in ref])
    missed = dists > miss_tol
    print(f"    coverage {int((~missed).sum())}/{len(ref)} within {miss_tol}"
          + (f"; MISSED {int(missed.sum())} "
             f"(worst {float(dists[missed].max()):.4f} at s={ws[missed][np.argmax(dists[missed])]:.4f})"
             if missed.any() else ""))
    return (not missed.any()) and clean_singularities


if __name__ == "__main__":
    cases = [int(a) for a in sys.argv[1:]] or list(ALL_CASES)
    ok = all([check_case(c) for c in cases])
    sys.exit(0 if ok else 1)
