"""Objective coverage check for the NURBS SSX adapter (_nssx5.py).

Independent reference cloud: isoline slices of surf1 (via
extract_isocurve) intersected with surf2 through nurbs_csx (the v4
adapter). Coverage = fraction of cloud points within 5*atol of the
adapter's branch polylines. Mirrors examples/ssx/bez_ssx5_coverage_check.py
at the NURBS level; the fixture pickles' stored curves are NOT used
(old-engine artifacts — see tests/test_nssx5.py fixture docstring).

GATE semantics per case (three-state row verdict):

- OK      — coverage 100% AND (complete=True OR reasons all
            typed-structural engine limits).
- PARTIAL — coverage < 100% AND complete=False AND nonempty reasons
            all typed-structural (an honest, typed engine limit — the
            pre-sanctioned outcome class; clearly printed, NOT a pass).
- FAIL    — everything else: any coverage shortfall with complete=True
            (silent loss — the damning case), any resource reason
            (work_budget/depth_limit/caps) at default budgets, any
            exception.

The gate PASSes iff no row is FAIL. An empty reference cloud (no
transversal crossings — the tangential-contact class) makes coverage
vacuously 100%; the status check still applies unchanged.

The gate certifies transversal-branch coverage only; points/
singularities/overlap_regions are reported but not coverage-checked.

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
#   REASON_TRACE_UNVERIFIED  = "trace_unverified"  (in _work_budget.py's
#     documented "Structural family — raising budgets cannot help"; its
#     omission from the plan's set was an enumeration oversight)
STRUCTURAL_REASONS = {
    'unresolved_tangential_zone',
    'unresolved_multiplicity',
    'overlap_region_unsupported',
    'parameter_fiber',
    'unresolved_singular_set',
    'trace_unverified',
}

# Per-case context printed with the row (controller-reviewed engine truth).
CASE_NOTES = {
    6: ("typed partial: bez-level trace continuation loses an SSI arm; "
        "honestly reported complete=False"),
    10: ("tangential contact at z=5: the u-family cloud is empty; the "
         "v-family samples the tangential curve, covered by the "
         "tangential branch (verified 100%, no longer vacuous)"),
    11: ("far-coordinate geometry; work_budget at candidate-scaled "
         "defaults — see budget probe in the Task-7 report"),
}


def load_pair(num):
    with open(CASE_DIR / f"nurbs_nurbs_intersection_{num}.pkl", "rb") as f:
        data = pickle.load(f)
    return data[0]


def reference_cloud(s1, s2, atol=ATOL, n=N_SLICES):
    """Union of u-sliced and v-sliced independent reference clouds.

    Two slice families close the orientation blind spot: an SSI fragment
    running parallel to one isoline family is transversal to the other.
    """
    (u0, u1), (v0, v1) = s1.interval()
    pts = []
    incomplete_slices = 0
    total_slices = 0
    for direction, (w0, w1) in (("u", (u0, u1)), ("v", (v0, v1))):
        for w in np.linspace(w0 + 1e-9, w1 - 1e-9, n):
            iso = extract_isocurve(s1, float(w), direction=direction)
            isolated, _overlaps, status = nurbs_csx(iso, s2, tol=atol)
            total_slices += 1
            if not status.get('complete', True):
                incomplete_slices += 1
            for entry in (isolated if isolated is not None else []):
                pts.append(np.asarray(entry['point'], dtype=np.float64)[:3])
    if incomplete_slices:
        print(f"    [warn] {incomplete_slices}/{total_slices} CSX slices "
              "incomplete (reference cloud may be partial there)")
    return np.asarray(pts) if pts else np.zeros((0, 3))


def run_case(num):
    """Run one case; return its three-state verdict: OK | PARTIAL | FAIL."""
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
    if len(cloud) == 0:
        # No transversal crossings for the isoline x CSX probe to find:
        # nothing can be missed, so coverage is vacuously full. The
        # status check below still applies unchanged.
        coverage = 100.0
    else:
        coverage = 100.0 * (len(cloud) - len(misses)) / len(cloud)
    reasons = set(res['status']['reasons'])
    structural = bool(reasons) and reasons <= STRUCTURAL_REASONS
    if coverage >= 100.0 and (res['complete'] or structural):
        verdict = 'OK'
    elif coverage < 100.0 and not res['complete'] and structural:
        # Typed, honest engine limit — pre-sanctioned outcome class.
        # Clearly printed; it is NOT a pass, but it is not silent loss
        # or resource exhaustion either.
        verdict = 'PARTIAL'
    else:
        # Silent loss (shortfall while claiming complete), a resource
        # reason at default budgets, or an untyped partial.
        verdict = 'FAIL'
    print(f"case {num}: {len(res['branches'])} branches, "
          f"{len(res['points'])} points, "
          f"{len(res['singularities'])} singularities, "
          f"{len(res['overlap_regions'])} regions | "
          f"complete={res['complete']} reasons={sorted(reasons)} | "
          f"coverage {coverage:.2f}% "
          f"({len(cloud) - len(misses)}/{len(cloud)}) | "
          f"ssx {dt:.1f}s + ref {dt_ref:.1f}s | "
          f"{verdict}")
    if len(cloud) == 0:
        print("    reference cloud EMPTY (tangential contact, or reference "
              "found no crossings); coverage vacuous")
    if num in CASE_NOTES:
        print(f"    note: {CASE_NOTES[num]}")
    for p, d in misses[:5]:
        print(f"    miss: {np.array2string(p, precision=5)} "
              f"dist={d / ATOL:.1f}*atol")
    return verdict


def main():
    cases = ([int(a) for a in sys.argv[1:]] if len(sys.argv) > 1
             else list(ALL_CASES))
    verdicts = {}
    for num in cases:
        try:
            verdicts[num] = run_case(num)
        except Exception as exc:            # noqa: BLE001 — harness report
            print(f"case {num}: EXCEPTION {type(exc).__name__}: {exc}")
            verdicts[num] = 'FAIL'
    n_ok = sum(1 for v in verdicts.values() if v == 'OK')
    n_partial = sum(1 for v in verdicts.values() if v == 'PARTIAL')
    n_fail = sum(1 for v in verdicts.values() if v == 'FAIL')
    partial_cases = sorted(n for n, v in verdicts.items() if v == 'PARTIAL')
    if partial_cases:
        print(f"PARTIAL(typed) cases: {partial_cases}")
    print(f"GATE: {n_ok} OK, {n_partial} PARTIAL(typed), {n_fail} FAIL")
    print("GATE:", "PASS" if n_fail == 0 else "INCOMPLETE — see rows above")


if __name__ == "__main__":
    main()
