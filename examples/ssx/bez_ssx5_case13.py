"""case 13 (USER 2026-07-10): rational sphere patch vs bilinear patch — HANG.

S1: degree (2,2) rational sphere segment (weights 0.707/0.5 pattern);
S2: bilinear ruled quad. User titled it "overlapping sphere segments";
regardless of the true topology the observed behavior is: bez_ssx NEVER
TERMINATES — a violation of the budget-bounded no-hang invariant.

Homogeneous nets provided (RATIONAL=True). Run under a watchdog only.
"""
import numpy as np

RATIONAL = True

_P1 = np.array([[[4.9056771, -3.79942682, -0.], [4.9056771, -3.79942682, 0.65971369],
                 [4.9056771, -4.45914051, 0.65971369]],
                [[4.24596341, -3.79942682, -0.], [4.24596341, -3.79942682, 0.65971369],
                 [4.9056771, -4.45914051, 0.65971369]],
                [[4.24596341, -4.45914051, -0.], [4.24596341, -4.45914051, 0.65971369],
                 [4.9056771, -4.45914051, 0.65971369]]])
_W1 = np.array([[1., 0.70710678, 1.], [0.70710678, 0.5, 0.70710678], [1., 0.70710678, 1.]])

_P2 = np.array([[[5.09151335, -3.56518057, 0.04708861], [5.26654768, -3.91168979, 0.82282564]],
                [[3.75042844, -4.24261104, 0.04708861], [3.92546278, -4.58912026, 0.82282564]]])
_W2 = np.ones((2, 2))

# homogeneous form (P*w, w) — the bez_ssx rational convention
S1 = np.concatenate([_P1 * _W1[..., None], _W1[..., None]], axis=-1)
S2 = np.concatenate([_P2 * _W2[..., None], _W2[..., None]], axis=-1)


if __name__ == "__main__":
    from mmcore.numeric.intersection.ssx._bez_ssx5 import bez_ssx
    res = bez_ssx(S1, S2, 1e-3, rational=True)
    print(f"branches={len(res['branches'])} points={len(res['points'])} "
          f"singularities={[g.kind for g in res.get('singularities', [])]}")
