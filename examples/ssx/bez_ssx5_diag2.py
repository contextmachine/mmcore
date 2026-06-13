"""Deeper diagnostic — capture per-cell fragment endpoints (start/end stuv, id) for cases 8 and 10.

The goal: see which BoundaryPoint identities are shared between fragments,
so we can tell if assembly should chain them but doesn't, or if they
genuinely don't share endpoints.
"""

from __future__ import annotations

import warnings
import numpy as np

warnings.filterwarnings("ignore")

from mmcore.numeric.intersection.ssx import _bez_ssx5 as M

orig_trace = M._trace_cell_by_registrations
orig_assemble = M._assemble_fragments

LOG = []  # list of (depth, box, [(start_id, start_stuv, end_id, end_stuv), ...])


def _capture(case_name, S1, S2, expected, atol=1e-3):
    LOG.clear()

    def trace_hook(cell, atol):
        fr, pt = orig_trace(cell, atol)
        info = []
        for f in fr:
            sid = id(f.start_point) if f.start_point else None
            eid = id(f.end_point) if f.end_point else None
            ss = tuple(round(float(x), 4) for x in f.stuv_path[0])
            ee = tuple(round(float(x), 4) for x in f.stuv_path[-1])
            info.append((sid, ss, eid, ee, len(f.stuv_path)))
        LOG.append((cell.depth, tuple(cell.box), info))
        return fr, pt

    def assemble_hook(fragments):
        # Capture fragment-level details before assembly
        print(f"\n  ASSEMBLE called with {len(fragments)} fragments:")
        for k, f in enumerate(fragments):
            sid = id(f.start_point) if f.start_point else None
            eid = id(f.end_point) if f.end_point else None
            ss = tuple(round(float(x), 4) for x in f.stuv_path[0])
            ee = tuple(round(float(x), 4) for x in f.stuv_path[-1])
            print(f"    f[{k}]: sid={sid}  start={ss}  ->  eid={eid}  end={ee}  len={len(f.stuv_path)}")
        return orig_assemble(fragments)

    M._trace_cell_by_registrations = trace_hook
    M._assemble_fragments = assemble_hook
    try:
        res = M.bez_ssx(S1, S2, atol, rational=False)
    finally:
        M._trace_cell_by_registrations = orig_trace
        M._assemble_fragments = orig_assemble

    print(f"\n=== {case_name} (expected={expected}) ===")
    for depth, box, frags in LOG:
        if not frags:
            continue
        sb = " ".join(f"[{lo:.3f},{hi:.3f}]" for lo, hi in box)
        print(f"  depth={depth} box={sb}")
        for sid, ss, eid, ee, ln in frags:
            print(f"    fr: sid={sid}  start={ss}  ->  eid={eid}  end={ee}  len={ln}")

    branches = res["branches"]
    print(f"  RESULT: {len(branches)} branches")
    for i, b in enumerate(branches):
        stuv, xyz = b.curve
        print(f"    branch[{i}]: {len(stuv)} pts  start={tuple(round(float(x),4) for x in stuv[0])}  "
              f"end={tuple(round(float(x),4) for x in stuv[-1])}")


def main():
    from examples.ssx.bez_ssx5_case8 import S1 as S1_8, S2 as S2_8
    from examples.ssx.bez_ssx5_case10 import S1 as S1_10, S2 as S2_10

    _capture("case 8", S1_8, S2_8, expected=1)
    print()
    print("=" * 80)
    print()
    _capture("case 10", S1_10, S2_10, expected=2)


if __name__ == "__main__":
    main()
