"""Dump the Bernstein-coefficient min/max of each TΨᵢ for every cell in
case 5's decomposition, alongside the interval T_box(B) evaluation that
`_check_tangency` uses. This tells us how close TΨᵢ is to definite sign
per cell — i.e. how close case 5 is to satisfying monotonicity.
"""
from __future__ import annotations

import warnings

import numpy as np

import mmcore.numeric.intersection.ssx._bez_ssx5 as mod

warnings.filterwarnings("ignore")

S1 = np.array([[[-19., -51., 3.], [-19., -46., 8.], [-19., -41., 3.], [-19., -36., 8.]],
               [[-14., -51., 8.], [-14., -46., 3.], [-14., -41., 8.], [-14., -36., 3.]],
               [[-9., -51., 3.], [-9., -46., 8.], [-9., -41., 3.], [-9., -36., 8.]],
               [[-4., -51., 8.], [-4., -46., 3.], [-4., -41., 8.], [-4., -36., 3.]]])

S2 = np.array([[[-20.35213885, -55.05885716, 0.], [-19., -46., 0.],
                [-18.09739608, -42.8559574, 0.], [-19.39149131, -33.02632827, 0.]],
               [[-15.52250972, -56.35295239, 7.], [-14., -46., 7.],
                [-14., -41., 13.], [-14.56186218, -31.73223305, 7.]],
               [[-10.69288059, -57.64704761, 7.], [-9., -46., 13.],
                [-9., -41., 13.], [-9.73223305, -30.43813782, 7.]],
               [[-5.86325146, -58.94114284, 0.], [-4., -46., 6.],
                [-4., -41., 0.], [-4.90260392, -29.1440426, 0.]]])


def main() -> None:
    created = []
    orig_init = mod._Cell.__init__

    def trk(self, *a, **kw):
        orig_init(self, *a, **kw)
        created.append(self)
    mod._Cell.__init__ = trk

    from mmcore.numeric.intersection.ssx._ssx4 import separate_gauss_maps

    mod.bez_ssx(S1, S2, atol=1e-3, rational=False)

    print(f"{len(created)} cells created.\n")
    print(f"{'depth':>5}  {'mono':<4}  {'gauss':<5}  "
          f"{'T1 range':<22}  {'T2 range':<22}  {'T3 range':<22}  {'T4 range':<22}")
    print("-" * 150)

    for cell in created:
        ranges = []
        for T in (cell.T1, cell.T2, cell.T3, cell.T4):
            arr = mod._tpsi_to_numpy(T)
            ranges.append((float(np.min(arr)), float(np.max(arr))))

        # Monotonicity
        mono_ax = None
        for i, (lo, hi) in enumerate(ranges):
            if lo >= 0 or hi <= 0:
                mono_ax = i
                break

        # Gauss
        try:
            p1, p2 = separate_gauss_maps(cell.g1.map_dirs(), cell.g2.map_dirs())
            gauss_ok = p1 is not None and p2 is not None
        except Exception:
            gauss_ok = False

        def fmt(lo, hi):
            sign = ("+"if lo >= 0 else ("-" if hi <= 0 else "±"))
            return f"{sign} [{lo:+.3f}, {hi:+.3f}]"

        mono_str = f"ax{mono_ax}" if mono_ax is not None else "—"
        g_str = "Y" if gauss_ok else "—"
        print(f"{cell.depth:>5}  {mono_str:<4}  {g_str:<5}  "
              f"{fmt(*ranges[0]):<22}  {fmt(*ranges[1]):<22}  "
              f"{fmt(*ranges[2]):<22}  {fmt(*ranges[3]):<22}")

    # Summary: how close are each cell's "best" TΨᵢ to one-sign?
    # Metric: min |hi| where lo<0, or min |lo| where hi>0 — "margin to flip"
    print("\nMargin-to-monotone per cell (min |component straddling 0|):")
    for cell in created:
        ranges = []
        for T in (cell.T1, cell.T2, cell.T3, cell.T4):
            arr = mod._tpsi_to_numpy(T)
            ranges.append((float(np.min(arr)), float(np.max(arr))))
        mono = any(lo >= 0 or hi <= 0 for lo, hi in ranges)
        # Best TΨᵢ's overshoot (smaller is closer to flipping definite)
        margins = []
        for lo, hi in ranges:
            if lo >= 0 or hi <= 0:
                margins.append(0.0)  # already mono
            else:
                # overshoot = min(|lo|, |hi|) — how much the off-sign side extends
                margins.append(min(abs(lo), abs(hi)))
        best_margin = min(margins)
        total_span = max(abs(lo) + abs(hi) for lo, hi in ranges)
        print(f"  depth={cell.depth}  mono={'Y' if mono else 'N'}  "
              f"closest-to-one-sign: {best_margin:+.4f} / span ≈ {total_span:.4f}  "
              f"({'already' if mono else f'{100*best_margin/total_span:.1f}% to flip'})")


if __name__ == "__main__":
    main()
