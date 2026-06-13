"""Diagnostic harness for cases 7-10.

For each case prints:
  - top-level boundary crossings count + their stuv values
  - per-iteration cell info (depth, box span, #crossings, what happened)
  - fragments produced
  - whether each top-level crossing was 'used' as a fragment endpoint
"""

from __future__ import annotations

import warnings
import numpy as np

warnings.filterwarnings("ignore")

from mmcore.numeric.intersection.ssx import _bez_ssx5 as M

orig_trace = M._trace_cell_by_registrations
orig_aabb = M._aabb_disjoint
orig_loop_free = M._check_loop_free
orig_check_tang = M._check_tangency
orig_deflate = M._deflate_tangent_cell

LOG = []


def reset_log():
    LOG.clear()


def install_hooks():
    def trace_hook(cell, atol):
        n_cx = len(cell.crossings)
        fr, pt = orig_trace(cell, atol)
        LOG.append(("trace", cell.depth, n_cx, len(fr), len(pt), tuple(cell.box)))
        return fr, pt

    def aabb_hook(S1_h, S2_h, atol):
        ok = orig_aabb(S1_h, S2_h, atol)
        if ok:
            LOG.append(("aabb_disjoint",))
        return ok

    def loop_free_hook(g1, g2, T1=None, T2=None, T3=None, T4=None):
        ok = orig_loop_free(g1, g2, T1, T2, T3, T4)
        return ok

    def deflate_hook(*args, **kwargs):
        fr, pt = orig_deflate(*args, **kwargs)
        LOG.append(("deflate", len(fr), len(pt)))
        return fr, pt

    M._trace_cell_by_registrations = trace_hook
    M._aabb_disjoint = aabb_hook
    M._check_loop_free = loop_free_hook
    M._deflate_tangent_cell = deflate_hook


def restore_hooks():
    M._trace_cell_by_registrations = orig_trace
    M._aabb_disjoint = orig_aabb
    M._check_loop_free = orig_loop_free
    M._deflate_tangent_cell = orig_deflate


def diag_case(name, S1, S2, expected, atol=1e-3):
    print(f"\n=== {name} (expected={expected}) ===")
    cx, ovl = M._find_ssx_boundary_zeros(S1, S2, atol, rational=False)
    print(f"  top-level boundary crossings: {len(cx)}")
    for i, c in enumerate(cx):
        print(f"    [{i}] stuv={tuple(round(float(x), 4) for x in c.stuv)}  "
              f"face={c.face}")
    print(f"  top-level overlaps: {len(ovl)}")

    reset_log()
    install_hooks()
    try:
        res = M.bez_ssx(S1, S2, atol, rational=False)
    finally:
        restore_hooks()

    branches = res["branches"]
    points = res["points"]
    print(f"  RESULT: {len(branches)} branches, {len(points)} standalone pts")
    for i, b in enumerate(branches):
        stuv, xyz = b.curve
        endpoints = (tuple(round(float(x), 4) for x in stuv[0]),
                     tuple(round(float(x), 4) for x in stuv[-1]))
        closed = bool(np.allclose(stuv[0], stuv[-1], atol=1e-4))
        print(f"    branch[{i}]: {len(stuv)} pts  closed={closed}  "
              f"start={endpoints[0]}  end={endpoints[1]}")

    n_aabb = sum(1 for e in LOG if e[0] == "aabb_disjoint")
    n_trace = sum(1 for e in LOG if e[0] == "trace")
    n_deflate = sum(1 for e in LOG if e[0] == "deflate")
    print(f"  cells terminated by AABB={n_aabb}  by trace={n_trace}  "
          f"by deflate={n_deflate}")
    if n_trace:
        print("  trace events (depth, n_cx_in_cell, n_fragments, n_pts):")
        for e in LOG:
            if e[0] == "trace":
                _, d, ncx, nfr, npt, box = e
                print(f"    depth={d}  cx={ncx}  fr={nfr}  pt={npt}")


def main():
    from examples.ssx.bez_ssx5_case7 import S1 as S1_7, S2 as S2_7
    from examples.ssx.bez_ssx5_case8 import S1 as S1_8, S2 as S2_8
    from examples.ssx.bez_ssx5_case9 import S1 as S1_9, S2 as S2_9
    from examples.ssx.bez_ssx5_case10 import S1 as S1_10, S2 as S2_10

    diag_case("case 7", S1_7, S2_7, expected=1)
    diag_case("case 8", S1_8, S2_8, expected=1)
    diag_case("case 9", S1_9, S2_9, expected=2)
    diag_case("case 10", S1_10, S2_10, expected=2)


if __name__ == "__main__":
    main()
