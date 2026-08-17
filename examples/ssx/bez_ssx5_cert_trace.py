"""Diagnostic: record per-cell loop-absence certificate outcomes on Case 5.

For each cell in the subdivision tree we log:
  - depth
  - which TΨᵢ (if any) is definite-sign, else the four min/max ranges
  - whether Gauss-map separability succeeded
  - whether Krawczyk tangency was attempted and what it returned
  - what the cell did: traced / subdivided / stopped

The intent: answer 'does case 5 satisfy monotonicity at some shallow depth,
or does it genuinely need deep subdivision?' without reading the decision
logic off a stack trace.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import mmcore.numeric.intersection.ssx._bez_ssx5 as mod

warnings.filterwarnings("ignore")

# Case 5 inputs
S1_CASE5 = np.array([[[-19., -51., 3.], [-19., -46., 8.], [-19., -41., 3.], [-19., -36., 8.]],
                     [[-14., -51., 8.], [-14., -46., 3.], [-14., -41., 8.], [-14., -36., 3.]],
                     [[-9., -51., 3.], [-9., -46., 8.], [-9., -41., 3.], [-9., -36., 8.]],
                     [[-4., -51., 8.], [-4., -46., 3.], [-4., -41., 8.], [-4., -36., 3.]]])

S2_CASE5 = np.array([[[-20.35213885, -55.05885716, 0.], [-19., -46., 0.],
                      [-18.09739608, -42.8559574, 0.], [-19.39149131, -33.02632827, 0.]],
                     [[-15.52250972, -56.35295239, 7.], [-14., -46., 7.],
                      [-14., -41., 13.], [-14.56186218, -31.73223305, 7.]],
                     [[-10.69288059, -57.64704761, 7.], [-9., -46., 13.],
                      [-9., -41., 13.], [-9.73223305, -30.43813782, 7.]],
                     [[-5.86325146, -58.94114284, 0.], [-4., -46., 6.],
                      [-4., -41., 0.], [-4.90260392, -29.1440426, 0.]]])


@dataclass
class CellRecord:
    depth: int
    box: tuple
    n_crossings: int
    t_ranges: list  # [(min, max) per TΨᵢ]
    mono_axis: Optional[int]  # 0..3 or None
    gauss_ok: bool
    tangency: Optional[str]  # 'True' / 'False' / 'None' / 'not called'
    outcome: str  # 'trace-loopfree' / 'trace-tangent' / 'subdivide' / 'stop' / 'maxdepth'


records: list[CellRecord] = []
cell_stack: list[CellRecord] = []


def main() -> None:
    # -- Instrument _check_monotonicity --
    orig_mono = mod._check_monotonicity

    def mono_recorder(T1, T2, T3, T4):
        ranges = []
        for T in (T1, T2, T3, T4):
            arr = mod._tpsi_to_numpy(T)
            ranges.append((float(np.min(arr)), float(np.max(arr))))
        is_mono, axis = orig_mono(T1, T2, T3, T4)
        cell_stack[-1].t_ranges = ranges
        if is_mono:
            cell_stack[-1].mono_axis = axis
        return is_mono, axis
    mod._check_monotonicity = mono_recorder

    # -- Instrument separate_gauss_maps via _check_loop_free --
    orig_lf = mod._check_loop_free

    def lf_recorder(g1, g2, T1=None, T2=None, T3=None, T4=None):
        result = orig_lf(g1, g2, T1, T2, T3, T4)
        if result and cell_stack and cell_stack[-1].mono_axis is None:
            cell_stack[-1].gauss_ok = True
        return result
    mod._check_loop_free = lf_recorder

    # -- Instrument _check_tangency --
    orig_tan = mod._check_tangency

    def tan_recorder(T1, T2, T3, T4, P1, P2, box):
        res = orig_tan(T1, T2, T3, T4, P1, P2, box)
        cell_stack[-1].tangency = str(res)
        return res
    mod._check_tangency = tan_recorder

    # -- Instrument _Cell to track lifecycle --
    orig_init = mod._Cell.__init__

    def trk_init(self, *a, **kw):
        orig_init(self, *a, **kw)

    mod._Cell.__init__ = trk_init

    # -- Instrument _trace_cell_by_registrations --
    orig_trace = mod._trace_cell_by_registrations

    def trace_recorder(cell, atol):
        if cell_stack:
            rec = cell_stack[-1]
            if rec.mono_axis is not None:
                rec.outcome = f"trace-mono(axis={rec.mono_axis})"
            elif rec.gauss_ok:
                rec.outcome = "trace-gauss"
            else:
                rec.outcome = "trace-(no-cert)"
        return orig_trace(cell, atol)
    mod._trace_cell_by_registrations = trace_recorder

    # -- Instrument _deflate_tangent_cell --
    orig_defl = mod._deflate_tangent_cell

    def defl_recorder(*a, **kw):
        if cell_stack:
            cell_stack[-1].outcome = "trace-phi"
        return orig_defl(*a, **kw)
    mod._deflate_tangent_cell = defl_recorder

    # -- Instrument the main loop by wrapping stack.pop through bez_ssx --
    # Easier: monkey-patch by walking cells as they're popped. We tee into
    # _check_loop_free as the first per-cell call and open a record there.
    def open_record(cell):
        rec = CellRecord(
            depth=cell.depth, box=cell.box,
            n_crossings=len(cell.crossings),
            t_ranges=[], mono_axis=None, gauss_ok=False,
            tangency='not called', outcome='?',
        )
        records.append(rec)
        cell_stack.append(rec)

    def close_record(cell):
        if cell_stack:
            cell_stack.pop()

    # Tap into loop body via _check_loop_free first use on a cell.
    prev_cell_id = [None]

    def lf_with_open(g1, g2, T1=None, T2=None, T3=None, T4=None):
        # This is tricky — _check_loop_free is called on both sub-cells in
        # a subdivision AND the top-level call. Use a separate entry hook.
        return lf_recorder(g1, g2, T1, T2, T3, T4)
    mod._check_loop_free = lf_with_open

    # Wrap the main loop differently: patch _Cell.__init__ to start a record
    # when a cell is CREATED, and infer outcome from what happens next.
    # But creation != popping. Let's patch stack.pop via wrapping bez_ssx
    # indirectly.

    # Cleaner approach: walk the logic by intercepting at the entrance of
    # each iteration. Use a sentinel on _check_loop_free to open the record:
    cell_stack.clear()
    records.clear()

    def lf_open(g1, g2, T1=None, T2=None, T3=None, T4=None):
        # A cell is being examined — open a record using the box we can
        # infer from g1.surface degree... no, we don't have that.
        return lf_recorder(g1, g2, T1, T2, T3, T4)

    # Best path: re-run bez_ssx body manually. Too invasive. Instead,
    # augment _Cell.__init__ to add itself to a "created" list; and after
    # the algorithm, inspect each cell's partitions.registrations to know
    # whether it traced anything.
    created_cells: list = []

    def trk_init2(self, *a, **kw):
        orig_init(self, *a, **kw)
        created_cells.append(self)
    mod._Cell.__init__ = trk_init2

    # Restore un-wrapped hooks that touched cell_stack.
    mod._check_monotonicity = orig_mono
    mod._check_loop_free = orig_lf
    mod._check_tangency = orig_tan
    mod._trace_cell_by_registrations = orig_trace
    mod._deflate_tangent_cell = orig_defl

    # Run the algorithm.
    r = mod.bez_ssx(S1_CASE5, S2_CASE5, atol=1e-3, rational=False)

    print(f"\nbez_ssx → {len(r['branches'])} branches, {len(r['points'])} points")
    print(f"{len(created_cells)} cells created\n")

    # Post-hoc certificate check on each cell: re-run the same tests.
    from mmcore.numeric.intersection.ssx._bez_ssx5 import (
        _check_monotonicity as chk_mono,
        _check_tangency as chk_tan,
    )
    from mmcore.numeric.intersection.ssx._ssx_substrate import separate_gauss_maps

    print(f"{'depth':>5}  {'#cx':>3}  {'box':<52}  {'mono':<14} {'gauss':<5} {'tang':<6}")
    print("-" * 100)

    for cell in created_cells:
        # Mono
        if cell.T1 is not None:
            is_mono, mono_axis = chk_mono(cell.T1, cell.T2, cell.T3, cell.T4)
        else:
            is_mono, mono_axis = False, None

        # Gauss
        try:
            p1, p2 = separate_gauss_maps(cell.g1.map_dirs(), cell.g2.map_dirs())
            gauss_ok = p1 is not None and p2 is not None
        except Exception:
            gauss_ok = False

        # Tangency — only if neither cheap test passed
        tan = 'skip'
        if not is_mono and not gauss_ok and cell.T1 is not None:
            P1c = cell.g1.surface[..., :-1] / cell.g1.surface[..., -1:]
            P2c = cell.g2.surface[..., :-1] / cell.g2.surface[..., -1:]
            res = chk_tan(cell.T1, cell.T2, cell.T3, cell.T4,
                          P1c, P2c, ((0.0, 1.0),) * 4)
            tan = str(res)

        box_str = "  ".join(f"[{lo:.3f},{hi:.3f}]" for lo, hi in cell.box)
        mono_str = f"axis={mono_axis}" if is_mono else "—"
        g_str = "Y" if gauss_ok else "—"
        print(f"{cell.depth:>5}  {len(cell.crossings):>3}  {box_str:<52}  "
              f"{mono_str:<14} {g_str:<5} {tan:<6}")

    # Summary histogram.
    print("\nSummary:")
    total = len(created_cells)
    mono_count = 0
    gauss_count = 0
    neither_count = 0
    for cell in created_cells:
        if cell.T1 is None:
            continue
        is_mono, _ = chk_mono(cell.T1, cell.T2, cell.T3, cell.T4)
        try:
            p1, p2 = separate_gauss_maps(cell.g1.map_dirs(), cell.g2.map_dirs())
            gauss_ok = p1 is not None and p2 is not None
        except Exception:
            gauss_ok = False
        if is_mono:
            mono_count += 1
        elif gauss_ok:
            gauss_count += 1
        else:
            neither_count += 1

    print(f"  Total cells with T arrays: {mono_count + gauss_count + neither_count}")
    print(f"  Monotonic                 : {mono_count}")
    print(f"  Gauss-separable           : {gauss_count}")
    print(f"  Neither (sub-or-tangent)  : {neither_count}")

    # Depth distribution
    max_depth = max(c.depth for c in created_cells)
    print(f"\n  Max depth reached: {max_depth}")
    print("  Cells per depth:")
    for d in range(max_depth + 1):
        at_d = [c for c in created_cells if c.depth == d]
        print(f"    depth {d}: {len(at_d)} cells")


if __name__ == "__main__":
    main()
