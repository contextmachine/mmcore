"""Budget/status contract gate for bez_ssx5 (ledger L52; review doc §11 step 5).

Institutionalizes the 2026-07-12 review probes: every registered coverage
case is solved TWICE and validated against the schema-v2 contract —

  1. schema presence: ``branches / points / singularities / overlap_regions /
     complete / status.reasons / status.work``;
  2. honesty invariant: ``complete == (not status.reasons)`` and every
     reason is from the documented vocabulary;
  3. output sanity: all branch ``stuv`` finite and inside [0,1]^4 (+eps),
     all xyz finite, and every branch vertex evaluates onto BOTH surfaces
     within 6*atol (independent evaluation through eval_surface);
  4. determinism: the md5 digest of all branch xyz polylines is identical
     across the two runs (no module-global PRNG, no dict-order leaks).

Exit code 0 only if every case passes every check. Run:
    python examples/ssx/bez_ssx5_budget_contract.py            # all cases
    python examples/ssx/bez_ssx5_budget_contract.py 5 15       # subset
"""
import hashlib
import sys

import numpy as np

from mmcore.numeric.intersection._bezier_common import eval_surface
from mmcore.numeric.intersection.ssx import _bez_ssx5 as engine
from bez_ssx5_coverage_check import ALL_CASES, load_case_surfaces

ATOL = 1e-3

REASON_VOCABULARY = {
    value for name, value in vars(engine).items()
    if name.startswith("REASON_") and isinstance(value, str)
}


def _digest(result):
    h = hashlib.md5()
    for b in result["branches"]:
        h.update(np.ascontiguousarray(
            np.asarray(b.curve[1], dtype=np.float64)).tobytes())
    return h.hexdigest()


def check_case(case):
    S1, S2, rational = load_case_surfaces(case)
    runs = [engine.bez_ssx(S1, S2, ATOL, rational=rational) for _ in range(2)]
    r = runs[0]
    ok = True

    def fail(msg):
        nonlocal ok
        ok = False
        print(f"    CONTRACT VIOLATION: {msg}")

    for key in ("branches", "points", "singularities", "overlap_regions",
                "complete", "status"):
        if key not in r:
            fail(f"missing result key {key!r}")
    status = r.get("status", {})
    for key in ("reasons", "work"):
        if key not in status:
            fail(f"missing status key {key!r}")

    reasons = list(status.get("reasons", []))
    if bool(r.get("complete")) != (not reasons):
        fail(f"complete={r.get('complete')} inconsistent with reasons={reasons}")
    unknown = [x for x in reasons if x not in REASON_VOCABULARY]
    if unknown:
        fail(f"undocumented reasons {unknown}")
    # §11.6 de-budget invariant (declared L52 slice 9, 2026-07-12): no
    # registered GATE case may report work_budget — the gate geometries
    # complete far below every allowance, so a work_budget here means a
    # misbilled structural condition or a genuine headroom regression.
    if "work_budget" in reasons:
        fail(f"gate case reports work_budget: {reasons}")
    work = status.get("work", {})
    for k, v in work.items():
        if isinstance(v, (int, float)) and not np.isfinite(v):
            fail(f"non-finite work counter {k}={v}")

    if rational:
        S1h, S2h = S1, S2
    else:
        S1h = np.concatenate([S1, np.ones(S1.shape[:-1] + (1,))], axis=-1)
        S2h = np.concatenate([S2, np.ones(S2.shape[:-1] + (1,))], axis=-1)
    worst = 0.0
    for bi, b in enumerate(r["branches"]):
        stuv = np.asarray(b.curve[0], dtype=np.float64)
        xyz = np.asarray(b.curve[1], dtype=np.float64)
        if not np.all(np.isfinite(stuv)) or not np.all(np.isfinite(xyz)):
            fail(f"branch {bi}: non-finite samples")
            continue
        if np.any(stuv < -1e-9) or np.any(stuv > 1.0 + 1e-9):
            fail(f"branch {bi}: stuv outside [0,1]^4 "
                 f"(min {stuv.min():.3g}, max {stuv.max():.3g})")
        for q, p in zip(stuv, xyz):
            p1 = eval_surface(S1h, float(q[0]), float(q[1]), rational=True)
            p2 = eval_surface(S2h, float(q[2]), float(q[3]), rational=True)
            worst = max(worst,
                        float(np.linalg.norm(p1 - p)),
                        float(np.linalg.norm(p2 - p)))
    if worst > 6.0 * ATOL:
        fail(f"branch sample residual {worst:.5f} > {6.0 * ATOL}")

    digests = {_digest(run) for run in runs}
    if len(digests) != 1:
        fail(f"nondeterministic branches across identical calls: {digests}")

    print(f"=== case {case}: complete={r.get('complete')} "
          f"reasons={reasons} branches={len(r['branches'])} "
          f"worst_resid={worst:.2e} "
          f"{'OK' if ok else 'VIOLATIONS ABOVE'}")
    return ok


if __name__ == "__main__":
    cases = [int(a) for a in sys.argv[1:]] or list(ALL_CASES)
    results = [check_case(c) for c in cases]
    sys.exit(0 if all(results) else 1)
