

import numpy as np


def bernstein_eval_homog(ctrl, t):
    # ctrl: (m, d) array, homogeneous for rational: d=3 (X,Y,W) or d=4 (X,Y,Z,W)
    pts = np.array(ctrl, dtype=float)
    n = len(pts) - 1
    tmp = pts.copy()
    for r in range(1, n+1):
        tmp[:n-r+1] = (1-t)*tmp[:n-r+1] + t*tmp[1:n-r+2]
    return tmp[0]

def jet_homog(ctrl, t):
    # Return (C, C', C'') in Euclidean from homogeneous control net (planar)
    ctrl = np.asarray(ctrl, float)
    n = len(ctrl)-1
    # First and second difference control polygons in homogeneous space:
    d1 = n * (ctrl[1:] - ctrl[:-1])
    d2 = (n-1) * n * (ctrl[2:] - 2*ctrl[1:-1] + ctrl[:-2]) if n >=2 else np.zeros((1,ctrl.shape[1]))
    # Evaluate
    H  = bernstein_eval_homog(ctrl, t)        # (X,Y,W)
    H1 = bernstein_eval_homog(d1, t)          # (X',Y',W')
    H2 = bernstein_eval_homog(d2, t) if n>=2 else np.zeros_like(H)
    c, w  = H[:-1],  H[-1]
    c1, w1 = H1[:-1], H1[-1]
    c2, w2 = H2[:-1], H2[-1]
    C  = c / w
    Cp = c1 / w - (w1 / w) * C
    Cpp= c2 / w - 2*(w1 / w)*Cp - (w2 / w - 2*(w1/w)**2) * C
    return C, Cp, Cpp

def signed_curvature_2d(Cp, Cpp, eps=1e-14):
    num = Cp[0]*Cpp[1] - Cp[1]*Cpp[0]
    den = (np.linalg.norm(Cp)**3)
    if den < eps:  # degenerate tangent
        return np.inf * np.sign(num if num!=0 else 1.0)
    return num/den



def _jet_kappa(ctrl, t):
    C, Cp, Cpp = jet_homog(ctrl, t)
    k = signed_curvature_2d(Cp, Cpp)
    return C, Cp, k



def classify_contact(ctrlA, tA, ctrlB, tB, pos_tol=1e-9, angle_tol=1e-6, k_tol=1e-6):
    CA, TA, KA = _jet_kappa(ctrlA, tA)
    CB, TB, KB = _jet_kappa(ctrlB, tB)

    if np.linalg.norm(CA - CB) > pos_tol:
        return "no-contact"

    nTA = np.linalg.norm(TA); nTB = np.linalg.norm(TB)
    if nTA == 0 or nTB == 0:
        return "tangent-degenerate"

    detT = TA[0]*TB[1] - TA[1]*TB[0]
    sin_theta = detT / (nTA * nTB)

    if abs(sin_theta) > angle_tol:
        return "transversal"

    # Tangential regime:
    if np.isinf(KA) or np.isinf(KB):
        return "tangent-degenerate"

    # Different curvature signs ⇒ touch-then-cross (still simple, but tangent at contact)
    if np.sign(KA) != np.sign(KB):
        return "tangent-crossing"

    # Same sign; equal magnitude within tol ⇒ higher-order contact likely
    if abs(KA - KB) <= k_tol:
        return "tangential (≥2), check higher order"

    # Same sign, different magnitude ⇒ simple tangency
    return "tangent (simple touch)"
import numpy as np

def perturb(ctrl, sigma=1e-9, seed=0):
    rng = np.random.default_rng(seed)
    out = np.array(ctrl, float)
    out[:,:-1] += rng.normal(0.0, sigma, size=out[:,:-1].shape)  # don't shake W too much
    return out

def check_multiplicity_stability(ctrlA, tA, ctrlB, tB, trials=15, sigma=1e-9):
    base = classify_contact(ctrlA, tA, ctrlB, tB)
    counts = {}
    for k in range(trials):
        ca = perturb(ctrlA, sigma, k)
        cb = perturb(ctrlB, sigma, k+1337)
        c = classify_contact(ca, tA, cb, tB)
        counts[c] = counts.get(c,0)+1
    return base, counts
if __name__ == "__main__":
    # both rational with w=1 (polynomial Bézier)
    ctrl_line = np.array([[0.0, 0.0, 1.0],
                          [1.0, 0.0, 1.0],
                          [2.0, 0.0, 1.0]])
    
    # same tangent at t=0, but nonzero curvature
    ctrl_arc = np.array([[0.0, 0.0, 1.0],
                         [1.0, 0.0, 1.0],
                         [2.0, 0.5, 1.0]])
    # At t=0: C'(0)= (2,0) for both; line has C''=0, arc has C''=(0, 1.0)·2=(0,1)·2 -> signed curvature > 0.
    # Expected: "tangent (simple touch)" (not coincident).
    print(classify_contact(ctrl_line,0.0, ctrl_arc,0.0))
    print(check_multiplicity_stability(ctrl_line,0.0, ctrl_arc,0.0))