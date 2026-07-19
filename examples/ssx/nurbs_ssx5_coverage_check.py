"""Objective coverage check for the NURBS SSX adapter (_nssx5.py).

Independent reference cloud: isoline slices of surf1 (via
extract_isocurve) intersected with surf2 through nurbs_csx (the v4
adapter). Coverage = fraction of cloud points within 5*atol of the
adapter's branch polylines. Mirrors examples/ssx/bez_ssx5_coverage_check.py
at the NURBS level; the fixture pickles' stored curves are NOT used
(old-engine artifacts — see tests/test_nssx5.py fixture docstring).

GATE semantics per case: coverage == 100% AND the status is honest —
either complete=True, or complete=False with reasons that are ALL
typed-structural engine limits (tangential zones, multiplicity,
overlap-region structure). Resource exhaustion (work_budget,
depth_limit, caps) at default budgets FAILS the gate.

Usage:
    .venv/bin/python examples/ssx/nurbs_ssx5_coverage_check.py           # all cases
    .venv/bin/python examples/ssx/nurbs_ssx5_coverage_check.py 5 8      # subset
"""
import pathlib
import pickle
import sys
import time

import numpy as np

from mmcore.geom.nurbs_iso import extract_isocurve
from mmcore.numeric.intersection.csx._ncsx4 import nurbs_csx
from mmcore.numeric.intersection.ssx._nssx5 import nurbs_ssx
from mmcore.numeric.intersection.ssx._bez_ssx5 import _dist_point_polyline

CASE_DIR = pathlib.Path(__file__).parent
ALL_CASES = (1, 2, 4, 5, 6, 7, 8, 9, 10, 11)
ATOL = 1e-3
N_SLICES = 100

# Typed-structural reasons = honest engine limits (established Tasks 4-6);
# anything else (work_budget, depth_limit, caps) fails the gate. Values
# verified against mmcore/numeric/_work_budget.py:
#   REASON_TANGENTIAL_ZONE   = "unresolved_tangential_zone"
#   REASON_MULTIPLICITY      = "unresolved_multiplicity"
#   REASON_OVERLAP_REGION    = "overlap_region_unsupported"
#   REASON_PARAMETER_FIBER   = "parameter_fiber"
#   REASON_SINGULAR_SET      = "unresolved_singular_set"
STRUCTURAL_REASONS = {
    'unresolved_tangential_zone',
    'unresolved_multiplicity',
    'overlap_region_unsupported',
    'parameter_fiber',
    'unresolved_singular_set',
}


def load_pair(num):
    with open(CASE_DIR / f"nurbs_nurbs_intersection_{num}.pkl", "rb") as f:
        data = pickle.load(f)
    return data[0]


def reference_cloud(s1, s2, atol=ATOL, n=N_SLICES):
    (u0, u1), _ = s1.interval()
    pts = []
    incomplete_slices = 0
    for w in np.linspace(u0 + 1e-9, u1 - 1e-9, n):
        iso = extract_isocurve(s1, float(w), direction="u")
        isolated, _overlaps, status = nurbs_csx(iso, s2, tol=atol)
        if not status.get('complete', True):
            incomplete_slices += 1
        for entry in (isolated if isolated is not None else []):
            pts.append(np.asarray(entry['point'], dtype=np.float64)[:3])
    if incomplete_slices:
        print(f"    [warn] {incomplete_slices}/{n} CSX slices incomplete "
              "(reference cloud may be partial there)")
    return np.asarray(pts) if pts else np.zeros((0, 3))


def run_case(num):
    s1, s2 = load_pair(num)
    t0 = time.time()
    res = nurbs_ssx(s1, s2, atol=ATOL)
    dt = time.time() - t0
    t1 = time.time()
    cloud = reference_cloud(s1, s2)
    dt_ref = time.time() - t1
    polys = [np.asarray(b.curve[1], dtype=np.float64)
             for b in res['branches'] if len(b.curve[1]) >= 2]
    misses = []
    for p in cloud:
        d = (min(_dist_point_polyline(p, poly) for poly in polys)
             if polys else np.inf)
        if d > 5 * ATOL:
            misses.append((p, d))
    n = max(1, len(cloud))
    coverage = 100.0 * (len(cloud) - len(misses)) / n
    reasons = set(res['status']['reasons'])
    status_ok = res['complete'] or (reasons and reasons <= STRUCTURAL_REASONS)
    ok = coverage >= 100.0 and status_ok
    print(f"case {num}: {len(res['branches'])} branches, "
          f"{len(res['points'])} points, "
          f"{len(res['singularities'])} singularities, "
          f"{len(res['overlap_regions'])} regions | "
          f"complete={res['complete']} reasons={sorted(reasons)} | "
          f"coverage {coverage:.2f}% "
          f"({len(cloud) - len(misses)}/{len(cloud)}) | "
          f"ssx {dt:.1f}s + ref {dt_ref:.1f}s | "
          f"{'OK' if ok else 'FAIL'}")
    for p, d in misses[:5]:
        print(f"    miss: {np.array2string(p, precision=5)} "
              f"dist={d / ATOL:.1f}*atol")
    return ok


def main():
    cases = ([int(a) for a in sys.argv[1:]] if len(sys.argv) > 1
             else list(ALL_CASES))
    ok = True
    for num in cases:
        try:
            ok = run_case(num) and ok
        except Exception as exc:            # noqa: BLE001 — harness report
            print(f"case {num}: EXCEPTION {type(exc).__name__}: {exc}")
            ok = False
    print("GATE:", "PASS" if ok else "INCOMPLETE — see rows above")


if __name__ == "__main__":
    main()
