"""Baseline harness for bez_ssx v5.

Runs the 5 representative test cases, reports per-case:
  - number of branches, total points, open/closed tag
  - max residual |S1(s,t) - S2(u,v)| over all branch points
  - number of standalone SSXPoint results
  - min timing over N_RUNS

Used as the starting point and as the regression/perf check after each design iteration.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np

from mmcore.numeric.intersection._bezier_common import eval_surface
from mmcore.numeric.intersection.ssx._bez_ssx6 import bez_ssx, BoundaryOverlap

warnings.filterwarnings("ignore")

N_RUNS = 3


@dataclass
class Case:
    name: str
    S1: np.ndarray
    S2: np.ndarray
    expected_branches: int
    notes: str


def make_cases() -> list[Case]:
    cases: list[Case] = []

    s1 = np.array([[[0., 0., 5.], [0., 10., 5.]],
                   [[10., 0., 5.], [10., 10., 5.]]])
    s2 = np.array([[[0., 0., 0.], [0., 10., 0.]],
                   [[10., 0., 10.], [10., 10., 10.]]])
    cases.append(Case("planes", s1, s2, 1, "two planes, line intersection x=z=5"))

    s1 = np.array([[[0., 0., 0.], [0., 10., 0.]],
                   [[10., 0., 0.], [10., 10., 10.]]])
    s2 = np.array([[[0., 0., 3.], [0., 10., 3.]],
                   [[10., 0., 3.], [10., 10., 3.]]])
    cases.append(Case("transversal", s1, s2, 1, "bilinear raised corner vs flat z=3"))

    s1 = np.array([[[0., 0., 10.], [5., 5., 10.], [5., 10., 10.], [0., 15., 10.]],
                   [[5., 0., 0.], [10., 5., 0.], [10., 10., 0.], [5., 15., 0.]],
                   [[10., 0., 10.], [15., 5., 10.], [15., 10., 10.], [10., 15., 10.]]])
    s2 = np.array([[[0., 0., 0.], [5., 5., 0.], [5., 10., 0.], [0., 15., 0.]],
                   [[5., 0., 10.], [10., 5., 10.], [10., 10., 10.], [5., 15., 10.]],
                   [[10., 0., 0.], [15., 5., 0.], [15., 10., 0.], [10., 15., 0.]]])
    cases.append(Case("tangential", s1, s2, 1, "crossed bicubic saddles, tangent curve via Φ-tracer"))

    s1 = np.array([[[-128.25, -129.86, 67.44], [-128.25, 129.86, 0.]],
                   [[128.25, -46.98, 0.], [128.25, 129.86, 0.]]])
    s2 = np.array([[[-128.25, -129.86, 0.], [-128.25, 129.86, 0.]],
                   [[128.25, -129.86, 0.], [128.25, 129.86, 0.]]])
    cases.append(Case("overlaps", s1, s2, 2, "bilinear with 2 edges on plane, 2 overlap branches"))

    s1 = np.array([[[-19., -51., 3.], [-19., -46., 8.], [-19., -41., 3.], [-19., -36., 8.]],
                   [[-14., -51., 8.], [-14., -46., 3.], [-14., -41., 8.], [-14., -36., 3.]],
                   [[-9., -51., 3.], [-9., -46., 8.], [-9., -41., 3.], [-9., -36., 8.]],
                   [[-4., -51., 8.], [-4., -46., 3.], [-4., -41., 8.], [-4., -36., 3.]]])
    s2 = np.array([[[-20.35213885, -55.05885716, 0.], [-19., -46., 0.],
                    [-18.09739608, -42.8559574, 0.], [-19.39149131, -33.02632827, 0.]],
                   [[-15.52250972, -56.35295239, 7.], [-14., -46., 7.],
                    [-14., -41., 13.], [-14.56186218, -31.73223305, 7.]],
                   [[-10.69288059, -57.64704761, 7.], [-9., -46., 13.],
                    [-9., -41., 13.], [-9.73223305, -30.43813782, 7.]],
                   [[-5.86325146, -58.94114284, 0.], [-4., -46., 6.],
                    [-4., -41., 0.], [-4.90260392, -29.1440426, 0.]]])
    cases.append(Case("case5", s1, s2, 2, "wavy bicubic surfaces, two open branches"))

    s1 = np.array([[[7.4968198, -34.44808135, 6.627417],
                    [3.96128589, -40.81204238, -8.372583],
                    [-3.8168887, -48.59021697, 6.627417]],
                   [[14.73328128, -36.02768858, 6.627417],
                    [6.95510669, -43.80586318, -9.372583],
                    [3.41957278, -47.34139708, 6.627417]],
                   [[17.72710208, -39.02150938, 6.627417],
                    [9.94892749, -46.79968397, -7.372583],
                    [6.41339358, -50.33521788, 6.627417]],
                   [[17.18538897, -45.55086408, 6.627417],
                    [15.94274828, -49.79350477, -2.372583],
                    [13.16457369, -49.57167936, -7.372583]]])
    s2 = np.array([[[0., -51., 6.29241333], [0., -46., 6.09945352],
                    [-0., -46., -4.68504969], [0., -36., -4.70758667]],
                   [[5., -51., 6.09945352], [5., -46., -4.68504969],
                    [5., -41., 6.09945352], [10., -36., -4.68504969]],
                   [[6., -51., -4.68504969], [10., -46., 6.09945352],
                    [10., -41., -4.68504969], [10., -36., 6.09945352]],
                   [[15., -51., -4.70758667], [15., -42., -4.68504969],
                    [15., -41., 6.09945352], [15., -36., 6.29241333]]])
    cases.append(Case("case6", s1, s2, 2, "loop + open branch (loop currently MISSING)"))

    s1_swept = np.array([[[33.05079627, -57.09987394, 0.],
                          [29.5295466, -63.44484237, 6.7646494],
                          [21.73708777, -71.24200956, 0.]],
                         [[40.28725776, -58.67948118, 0.],
                          [32.51384961, -66.4481508, 9.37051336],
                          [28.97354926, -69.99318967, 0.]],
                         [[43.28107855, -61.67330197, 0.],
                          [35.49815262, -69.45145922, 9.37051336],
                          [28.73859119, -79.68670826, 0.]],
                         [[45.10433572, -68.20265667, 0.],
                          [41.48244052, -72.46428996, 4.71678541],
                          [38.71855016, -76.6579268, 0.]]])

    s2 = np.array([[[40.25282656, -76.40733562, 3.35169568],
                    [45.30378577, -65.64948729, 4.77804379]],
                   [[23.11248642, -70.28329548, 4.87380266],
                    [30.39942473, -58.97443598, 3.35169568]]])
    cases.append(Case("case7", s1_swept, s2, 1, "one internal loop (currently MISSED)"))

    s2 = np.array([[[40.25282656, -76.40733562, 2.23739797],
                    [45.30378577, -65.64948729, 3.66374609]],
                   [[23.11248642, -70.28329548, 3.75950495],
                    [30.39942473, -58.97443598, 2.23739797]]])
    cases.append(Case("case8", s1_swept, s2, 1, "one branch (segments dropping out)"))

    s2 = np.array([[[40.25282656, -76.40733562, -0.05990905],
                    [45.30378577, -65.64948729, 1.36643906]],
                   [[23.11248642, -70.28329548, 1.46219793],
                    [30.39942473, -58.97443598, -0.05990905]]])
    cases.append(Case("case9", s1_swept, s2, 2, "two branches (one currently MISSED)"))

    s2 = np.array([[[29.63685574, -70.79194487, 4.04308391],
                    [33.99717923, -70.79194487, 7.50248027],
                    [39.66180486, -70.79194487, 4.18744742]],
                   [[29.63685574, -66.43162138, 0.58368755],
                    [33.99717923, -66.43162138, 4.04308391],
                    [39.66180486, -66.43162138, 0.72805106]],
                   [[29.63685574, -60.76699576, 3.89872039],
                    [33.99717923, -60.76699576, 7.35811675],
                    [39.66180486, -60.76699576, 4.04308391]]])
    cases.append(Case("case10", s1_swept, s2, 2, "two branches (both partial)"))

    s2 = np.array([[[39.589021714123604, -77.29117490559284, 3.5489239217672024],
                    [44.639980924198085, -66.5333265772515, 4.9752720370625365]],
                   [[22.4486815782254, -71.16713476157344, 5.071030903961587],
                    [29.735619889641292, -59.85827526536872, 3.5489239217672024]]])
    cases.append(Case("case11", s1_swept, s2, 1, "single closed loop, interior — currently fragmented"))

    return cases


def max_residual(branches: list, S1: np.ndarray, S2: np.ndarray, rational: bool) -> float:
    err = 0.0
    for b in branches:
        stuv, xyz = b.stuv, b.xyz

        for i in range(len(stuv)):
            s, t, u, v = stuv[i]

            p1 = eval_surface(S1, s, t, rational=rational)
            p2 = eval_surface(S2, u, v, rational=rational)
            d = float(np.linalg.norm(p1 - p2))
            if d > err:
                err = d
    return err


def run_case(case: Case, atol: float = 1e-3) -> dict:
    S1h = np.concatenate([case.S1, np.ones(case.S1.shape[:-1] + (1,))], axis=-1)
    S2h = np.concatenate([case.S2, np.ones(case.S2.shape[:-1] + (1,))], axis=-1)

    best_t = float("inf")
    result = None
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        result = bez_ssx(case.S1, case.S2, atol=atol, rational=False)
        t1 = time.perf_counter()
        best_t = min(best_t, t1 - t0)

    branches = result["branches"]
    points = result["points"]
    total_pts = sum(len(b.xyz) if not isinstance(b,BoundaryOverlap) else 0 for b in branches)
    err = max_residual(branches, S1h, S2h, rational=True)

    return dict(
        name=case.name,
        expected=case.expected_branches,
        actual=len(branches),
        total_pts=total_pts,
        n_points=len(points),
        max_err=err,
        time_s=best_t,
        notes=case.notes,
    )


def main() -> None:
    rows = []
    for case in make_cases():
        r = run_case(case)
        rows.append(r)
        status = "OK" if r["actual"] == r["expected"] else "MISMATCH"
        print(f"[{status:8}] {r['name']:12} | exp={r['expected']} act={r['actual']:>2}  "
              f"pts={r['total_pts']:>4}  +{r['n_points']:>2}pt  err={r['max_err']:.2e}  "
              f"t={r['time_s']*1000:7.1f}ms  // {r['notes']}")

    print()
    print("| case | expected | actual | branches pts | standalone pts | max |S1-S2| | time (ms) | status |")
    print("|------|---------:|-------:|------------:|---------------:|-----------:|----------:|:------:|")
    for r in rows:
        status = "OK" if r["actual"] == r["expected"] else "MISMATCH"
        print(f"| {r['name']} | {r['expected']} | {r['actual']} | {r['total_pts']} | "
              f"{r['n_points']} | {r['max_err']:.2e} | {r['time_s']*1000:.1f} | {status} |")


if __name__ == "__main__":
    main()