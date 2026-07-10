"""case 14 (USER 2026-07-10): two rational cone segments, single singular
tangent branch at their intersection — HANG.

Both surfaces are degree (2,1) rational cone patches (apex rows of repeated
control points; 0.707 middle-row weights). The true SSI is one singular
tangent branch (cone-cone tangency along a line). Observed: bez_ssx NEVER
TERMINATES — no-hang invariant violated. Apex = a degenerate
parameterization row (Sigma = 0 at the apex: C1-adjacent territory) AND
rational weights: exercises L9/L26 rational paths + tangent-curve handling
at once.

Homogeneous nets provided (RATIONAL=True). Run under a watchdog only.
"""
import numpy as np

RATIONAL = True

_P1 = np.array([[[26.44006464, -11.11732309, 46.51978915], [26.44006464, -24.15357802, 0.]],
                [[26.44006464, -11.11732309, 46.51978915], [13.40380971, -24.15357802, 0.]],
                [[26.44006464, -11.11732309, 46.51978915], [13.40380971, -11.11732309, 0.]]])
_W1 = np.array([[1., 1.], [0.70710678, 0.70710678], [1., 1.]])

_P2 = np.array([[[21.61898362, -23.22934688, 0.], [28.29755124, -12.03378182, 46.51978915]],
                [[21.61898362, -23.22934688, 0.], [17.10198618, -5.3552142, 46.51978915]],
                [[21.61898362, -23.22934688, 0.], [10.42341856, -16.55077926, 46.51978915]]])
_W2 = np.array([[1., 1.], [0.70710678, 0.70710678], [1., 1.]])

S1 = np.concatenate([_P1 * _W1[..., None], _W1[..., None]], axis=-1)
S2 = np.concatenate([_P2 * _W2[..., None], _W2[..., None]], axis=-1)


if __name__ == "__main__":
    from mmcore.numeric.intersection.ssx._bez_ssx5 import bez_ssx
    res = bez_ssx(S1, S2, 1e-3, rational=True)
    print(f"branches={len(res['branches'])} points={len(res['points'])} "
          f"singularities={[g.kind for g in res.get('singularities', [])]}")
