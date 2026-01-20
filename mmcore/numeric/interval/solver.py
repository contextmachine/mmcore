from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from mmcore.numeric.ndinterval import get_iarray, get_lu, interval as interval_dtype

# ---------------------------------------------------------------------
# Interval scalar construction helpers
# ---------------------------------------------------------------------

Interval = interval_dtype  # scalar interval type


def iconst(x: float) -> Interval:
    """
    Degenerate interval [x, x], constructed via NumPy casting (works with your dtype).
    """
    return np.array(x, dtype=Interval).item()


def ival(l: float, u: float) -> Interval:
    """
    Interval [l, u]. We try calling the type (in case interval(l,u) exists).
    Otherwise, construct a degenerate interval and set .l/.u explicitly.
    """
    try:
        return Interval(l, u)  # may or may not exist in your extension
    except Exception:
        it = iconst(0.0)
        it.l = float(l)
        it.u = float(u)
        return it


# ---------------------------------------------------------------------
# Interval scalar predicates and transforms (using .l/.u members)
# ---------------------------------------------------------------------

def i_is_empty(a: Interval) -> bool:
    # Common representation of empty interval: lower bound > upper bound.
    # Also treat NaNs as empty to avoid propagating garbage.
    return (a.l > a.u) or (np.isnan(a.l) or np.isnan(a.u))


def i_mid(a: Interval) -> float:
    return 0.5 * (a.l + a.u)


def i_width(a: Interval) -> float:
    return a.u - a.l


def i_contains_zero(a: Interval) -> bool:
    return (a.l <= 0.0) and (a.u >= 0.0)


# ---------------------------------------------------------------------
# Box utilities: a "box" is a 1D numpy array of Interval of length n
# ---------------------------------------------------------------------

Box = np.ndarray  # shape (n,) dtype Interval
IMat = np.ndarray  # shape (m,n) dtype Interval


def box_from_bounds(lows: Sequence[float], highs: Sequence[float]) -> Box:
    lows = list(lows)
    highs = list(highs)
    if len(lows) != len(highs):
        raise ValueError("lows and highs must have same length")
    out = np.empty((len(lows),), dtype=Interval)
    for k, (lo, hi) in enumerate(zip(lows, highs)):
        out[k] = ival(float(lo), float(hi))
    return out


def box_copy(X: Box) -> Box:
    # For custom dtypes, explicit copy is safer than view semantics.
    return np.array(X, dtype=Interval, copy=True)


def box_is_empty(X: Box) -> bool:
    return any(i_is_empty(xi) for xi in X)


def box_mid(X: Box) -> np.ndarray:
    return np.array([i_mid(xi) for xi in X], dtype=float)


def box_widths(X: Box) -> np.ndarray:
    return np.array([i_width(xi) for xi in X], dtype=float)


def box_max_width(X: Box) -> float:
    w = box_widths(X)
    return float(np.max(w)) if w.size else 0.0


def box_intersection(A: Box, B: Box) -> Box:
    """
    Elementwise interval intersection A ∩ B.
    Uses the scalar method .intersection from your interval object.
    """
    if A.shape != B.shape:
        raise ValueError("Box shapes differ")
    out = np.empty_like(A, dtype=Interval)
    for k in range(len(A)):
        out[k] = A[k].intersection(B[k])
    return out


def box_equal(A: Box, B: Box) -> bool:
    if A.shape != B.shape:
        return False
    return all(A[k].equal(B[k]) for k in range(len(A)))


def box_subseteq(A: Box, B: Box) -> bool:
    """
    True iff A ⊆ B (componentwise).
    """
    if A.shape != B.shape:
        return False
    return all(A[k].subseteq(B[k]) for k in range(len(A)))


def box_subset_strict(A: Box, B: Box) -> bool:
    """
    True iff A ⊂ B (strict componentwise subset). This corresponds to "inside interior"
    when used with Krawczyk uniqueness tests.
    """
    if A.shape != B.shape:
        return False
    return all(A[k].subset(B[k]) for k in range(len(A)))


def box_bisect(X: Box) -> Tuple[Box, Box]:
    """
    Bisect the widest dimension at its midpoint.
    Returns (left_box, right_box).
    """
    n = len(X)
    if n == 0:
        raise ValueError("Cannot bisect empty box")
    widths = box_widths(X)
    i = int(np.argmax(widths))
    xi = X[i]
    m = i_mid(xi)

    L = box_copy(X)
    R = box_copy(X)

    # Split [l,u] into [l,m] and [m,u]
    L[i] = ival(xi.l, m)
    R[i] = ival(m, xi.u)
    return L, R


# ---------------------------------------------------------------------
# Vector/matrix ops that must work with mixed float/interval
# ---------------------------------------------------------------------

def interval_mid_matrix(A: np.ndarray) -> np.ndarray:
    """
    Midpoint of an interval matrix as a float matrix.
    A may be (n,n) dtype Interval or contain Interval scalars.
    """
    out = np.empty(A.shape, dtype=float)
    it = np.nditer(A, flags=["multi_index", "refs_ok"], op_flags=["readonly"])
    for a in it:
        out[it.multi_index] = i_mid(a.item())
    return out


def matmul_float_interval(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute A @ B where:
      - A is float (m,n)
      - B is Interval (n,) or (n,p)

    Returns Interval (m,) or (m,p)
    """
    A = np.asarray(A, dtype=float)
    if B.ndim == 1:
        m, n = A.shape
        if B.shape[0] != n:
            raise ValueError("Shape mismatch in matmul_float_interval")
        out = np.empty((m,), dtype=Interval)
        for i in range(m):
            acc = iconst(0.0)
            for j in range(n):
                acc = acc + (A[i, j] * B[j])
            out[i] = acc
        return out

    if B.ndim == 2:
        m, n = A.shape
        n2, p = B.shape
        if n2 != n:
            raise ValueError("Shape mismatch in matmul_float_interval")
        out = np.empty((m, p), dtype=Interval)
        for i in range(m):
            for k in range(p):
                acc = iconst(0.0)
                for j in range(n):
                    acc = acc + (A[i, j] * B[j, k])
                out[i, k] = acc
        return out

    raise ValueError("B must be 1D or 2D")


def matmul_interval_interval(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Interval @ Interval multiplication.
    Implemented explicitly (small dimensions) to avoid relying on matmul ufunc coverage.
    """
    if A.ndim == 2 and B.ndim == 1:
        m, n = A.shape
        if B.shape[0] != n:
            raise ValueError("Shape mismatch in matmul_interval_interval")
        out = np.empty((m,), dtype=Interval)
        for i in range(m):
            acc = iconst(0.0)
            for j in range(n):
                acc = acc + (A[i, j] * B[j])
            out[i] = acc
        return out

    if A.ndim == 2 and B.ndim == 2:
        m, n = A.shape
        n2, p = B.shape
        if n2 != n:
            raise ValueError("Shape mismatch in matmul_interval_interval")
        out = np.empty((m, p), dtype=Interval)
        for i in range(m):
            for k in range(p):
                acc = iconst(0.0)
                for j in range(n):
                    acc = acc + (A[i, j] * B[j, k])
                out[i, k] = acc
        return out

    raise ValueError("Unsupported shapes for matmul_interval_interval")


def vector_contains_zero(FX: np.ndarray) -> bool:
    """
    FX is an interval vector. True iff 0 is contained in every component interval.
    """
    return all(i_contains_zero(fi) for fi in FX)


# ---------------------------------------------------------------------
# Krawczyk (interval Newton) step and classification
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class KrawczykStep:
    K: Box                 # Krawczyk image K(X)
    X_new: Box             # contracted: X ∩ K(X)
    exists: bool           # K(X) ⊆ X  -> at least one root in X
    unique: bool           # K(X) ⊂ int(X) -> exactly one root in X


def krawczyk_operator(
    F: Callable[[np.ndarray], np.ndarray],
    J: Callable[[np.ndarray], np.ndarray],
    X: Box,
) -> Optional[KrawczykStep]:
    """
    Compute the Krawczyk operator for the system F(x) = 0 on box X.

    K(X) = x0 - A*F(x0) + (I - A*J(X))*(X - x0)

    where:
      x0 = mid(X)
      A  = inv(J(x0))  (using a point Jacobian at x0)
      J(X) is an interval Jacobian enclosure over X.

    Returns None if J(x0) is singular / inversion fails (then you should bisect).
    """
    n = len(X)
    if n == 0:
        raise ValueError("Empty system")

    # Midpoint (float) and degenerate interval vector at x0
    x0 = box_mid(X)
    x0I = np.asarray(x0, dtype=Interval)

    # Evaluate at midpoint (float)
    F0 = np.asarray(F(x0), dtype=Interval)

    # Point Jacobian from degenerate-interval evaluation (gives exact point Jacobian if your code is well-behaved)
    J0I = J(x0I)  # should be intervals with l=u if everything is smooth
    J0 = interval_mid_matrix(J0I)

    try:
        A = np.linalg.inv(J0)
    except np.linalg.LinAlgError:
        return None

    # Interval Jacobian enclosure on the full box
    JX = J(X)  # interval matrix (n,n)

    # Compute M = I - A*JX as interval matrix
    AJ = matmul_float_interval(A, JX)     # interval (n,n)
    I = np.eye(n, dtype=float)
    _l,_u=get_lu(AJ)
    M = get_iarray(I -_l ,I-_u) # float - interval => interval (elementwise)

    # Compute correction term: M*(X - x0)
    Xm = X - x0  # interval vector
    corr = matmul_interval_interval(M, Xm)  # interval vector

    # Base: x0 - A*F0 (float), cast to degenerate interval vector
    base = x0 - (A @ F0)
    baseI = np.asarray(base, dtype=Interval)

    # Krawczyk image
    K = baseI + corr

    # Contract with X
    X_new = box_intersection(X, K)

    if box_is_empty(X_new):
        return KrawczykStep(K=K, X_new=X_new, exists=False, unique=False)

    # Existence / uniqueness checks (standard):
    # - if K(X) ⊆ X then at least one root exists in X
    # - if K(X) ⊂ int(X) then the root is unique in X
    exists = box_subseteq(K, X)
    unique = box_subset_strict(K, X)

    return KrawczykStep(K=K, X_new=X_new, exists=exists, unique=unique)


def contract_krawczyk(
    F: Callable[[np.ndarray], np.ndarray],
    J: Callable[[np.ndarray], np.ndarray],
    X: Box,
    max_iters: int = 8,
) -> Tuple[Box, Optional[KrawczykStep]]:
    """
    Repeatedly apply X := X ∩ K(X) until it stops changing or reaches max_iters.
    Returns (X_contracted, last_step_or_None).
    If inversion fails (step is None), returns current X and None.
    """
    Xc = box_copy(X)
    last: Optional[KrawczykStep] = None
    for _ in range(max_iters):
        step = krawczyk_operator(F, J, Xc)
        if step is None:
            return Xc, None
        last = step
        if box_is_empty(step.X_new) or box_equal(step.X_new, Xc):
            return step.X_new, step
        Xc = step.X_new
    return Xc, last


# ---------------------------------------------------------------------
# All-roots solver via branch-and-prune using the contractor
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class RootBox:
    box: Box
    x_mid: np.ndarray   # midpoint approximation
    certified_unique: bool


def solve_all_roots_krawczyk(
    F: Callable[[np.ndarray], np.ndarray],
    J: Callable[[np.ndarray], np.ndarray],
    X0: Box,
    *,
    tol: float = 1e-10,
    max_depth: int = 60,
    max_boxes: int = 200000,
    contractor_iters: int = 8,
) -> Tuple[List[RootBox], List[Box]]:
    """
    Find all roots of F(x)=0 in X0 using interval Newton/Krawczyk.

    Returns:
      - roots: list of RootBox, each either certified_unique=True, or (rare) certified_unique=False
               if you hit tol/max_depth without proof but still possible root region.
      - undecided: boxes that could not be classified within tol/max_depth (optional post-processing)

    Notes:
      - Quick prune: if 0 not in some component of F(X), discard.
      - Proof:
          unique root if K(X) ⊂ int(X)
          no root if X ∩ K(X) is empty OR 0 ∉ F(X) for some component
    """
    if box_is_empty(X0):
        return [], []

    roots: List[RootBox] = []
    undecided: List[Box] = []

    stack: List[Tuple[Box, int]] = [(box_copy(X0), 0)]
    processed = 0

    while stack:
        X, depth = stack.pop()
        processed += 1
        if processed > max_boxes:
            undecided.append(X)
            continue

        if box_is_empty(X):
            continue

        # Fast reject using interval evaluation of F on the box
        FX = F(X)
        if not vector_contains_zero(FX):
            continue

        # Contract
        Xc, last = contract_krawczyk(F, J, X, max_iters=contractor_iters)
        if box_is_empty(Xc):
            continue

        # If we got a usable Krawczyk step, we may have a proof
        if last is not None and last.unique:
            roots.append(RootBox(box=Xc, x_mid=box_mid(Xc), certified_unique=True))
            continue

        # If we are small enough, stop (cluster / undecided root region)
        if box_max_width(Xc) <= tol or depth >= max_depth:
            roots.append(RootBox(box=Xc, x_mid=box_mid(Xc), certified_unique=False))
            continue

        # Otherwise, split and continue
        L, R = box_bisect(Xc)
        stack.append((R, depth + 1))
        stack.append((L, depth + 1))

    return roots, undecided
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple
import numpy as np

# Reuse dot that works for float and interval vectors
def dot(a: np.ndarray, b: np.ndarray):
    return np.sum(a * b)


@dataclass(frozen=True)
class Entity:
    """
    Parametric entity in R^d.

    nparam:
      - curve: 1
      - surface: 2
      - point (constant): 0

    pos(*p) -> (d,) vector
    d1(*p)  -> list length nparam of first partial derivative vectors
    d2(*p)  -> list-of-lists (nparam x nparam) of second partial derivative vectors
    """
    nparam: int
    pos: Callable[..., np.ndarray]
    d1: Callable[..., Sequence[np.ndarray]]
    d2: Callable[..., Sequence[Sequence[np.ndarray]]]


def point_entity(P: np.ndarray) -> Entity:
    P = np.asarray(P, dtype=float)

    def pos() -> np.ndarray:
        return P

    def d1() -> Sequence[np.ndarray]:
        return []

    def d2() -> Sequence[Sequence[np.ndarray]]:
        return []

    return Entity(nparam=0, pos=pos, d1=d1, d2=d2)


def make_stationary_system(A: Entity, B: Entity):
    """
    Build F(x)=0 and J(x) for stationary points of ||A(p)-B(q)||^2.

    Unknown vector:
      x = [p0..p(m-1), q0..q(n-1)]
      where m=A.nparam, n=B.nparam

    Equations:
      For each A-parameter i:  dot(D, A_i) = 0
      For each B-parameter j:  dot(D, B_j) = 0
      where D = A(p) - B(q)

    Jacobian formula (rigorous, needs 2nd partials):
      For i,k in A:     dot(A_k, A_i) + dot(D, A_{ik})
      For i in A, l in B:  -dot(B_l, A_i)
      For j in B, k in A:   dot(A_k, B_j)
      For j,l in B:      -dot(B_l, B_j) + dot(D, B_{jl})
    """

    m = A.nparam
    n = B.nparam
    N = m + n

    def F(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        p = x[:m]
        q = x[m:]

        Ap = A.pos(*p)
        Bq = B.pos(*q)
        D = Ap - Bq

        A1 = list(A.d1(*p))
        B1 = list(B.d1(*q))

        out = np.empty((N,), dtype=D.dtype)

        # A-side equations
        for i in range(m):
            out[i] = dot(D, A1[i])

        # B-side equations
        for j in range(n):
            out[m + j] = dot(D, B1[j])

        return out

    def J(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        p = x[:m]
        q = x[m:]

        Ap = A.pos(*p)
        Bq = B.pos(*q)
        D = Ap - Bq

        A1 = list(A.d1(*p))
        B1 = list(B.d1(*q))
        A2 = [list(row) for row in A.d2(*p)]
        B2 = [list(row) for row in B.d2(*q)]

        M = np.empty((N, N), dtype=D.dtype)

        # Block (A,A)
        for i in range(m):
            for k in range(m):
                M[i, k] = dot(A1[k], A1[i]) + dot(D, A2[i][k])

        # Block (A,B)
        for i in range(m):
            for l in range(n):
                M[i, m + l] = -dot(B1[l], A1[i])

        # Block (B,A)
        for j in range(n):
            for k in range(m):
                M[m + j, k] = dot(A1[k], B1[j])

        # Block (B,B)
        for j in range(n):
            for l in range(n):
                M[m + j, m + l] = -dot(B1[l], B1[j]) + dot(D, B2[j][l])

        return M

    return F, J

def curve_entity(C, Cp, Cpp) -> Entity:
    def pos(t): return C(t)
    def d1(t):  return [Cp(t)]
    def d2(t):  return [[Cpp(t)]]
    return Entity(nparam=1, pos=pos, d1=d1, d2=d2)
def deriv_ctrl(P):  # first derivative control net
    n = P.shape[0] - 1
    return n * (P[1:] - P[:-1])
def second_deriv_ctrl(P):
    n = P.shape[0] - 1
    if n < 2:
        return np.zeros((1, P.shape[1]), dtype=P.dtype)
    return n * (n - 1) * (P[2:] - 2 * P[1:-1] + P[:-2])
def bernstein_eval_1d(P, u):
    # Horner/de Casteljau eval
    P = np.asarray(P)
    dtype = P.dtype
    Q = np.asarray(P, dtype=dtype).copy()
    u = np.asarray(u, dtype=dtype)
    one = np.asarray(1, dtype=dtype)
    n = Q.shape[0] - 1
    for r in range(1, n + 1):
        Q = (one - u) * Q[:-1] + u * Q[1:]
    return Q[0]
def bez_curve_entity(P) -> Entity:
    dP=deriv_ctrl(P)
    ddP=second_deriv_ctrl(P)
    def pos(t): return bernstein_eval_1d(P,t)
    def d1(t):  return [bernstein_eval_1d(dP,t)]
    def d2(t):  return [[bernstein_eval_1d(ddP,t)]]
    return Entity(nparam=1, pos=pos, d1=d1, d2=d2)
if __name__ == "__main__":

    curve1 = np.array(
        [
            [-19.77608536, 23.10065701, 0.0],
            [-14.86834768, 28.69713066, 0.0],
            [-5.8568525, 25.12677787, 0.0],
            [-12.62581769, 15.26478654, 0.0],
        ]
    )
    curve2 = np.array(
        [
            [-22.0315362, 18.75969713, 0.0],
            [-19.42270945, 28.2502867, 0.0],
            [-8.46791623, 27.56878356, 0.0],
            [-10.43007782, 19.78973126, 0.0],
        ]
    )
    curve3 = np.array(
        [
            [-28.46565557, -11.09883504, 0.0],
            [-31.79098016, 13.62423043, 0.0],
            [-12.99566723, 16.66039636, 0.0],
            [8.11291498, -6.32771715, 0.0],
        ]
    )

    curve4 = np.array(
        [
            [-45.36434109, -7.12015504, 0.0],
            [-25.49612403, 13.94186047, 0.0],
            [-2.13178295, -17.35271318, 0.0],
            [12.02325581, 20.42248062, 0.0],
        ]
    )
    curve6 = np.array([[-13.12449258, 9.10030377, 0.0], [-27.74989311, 10.37986052, 0.0], [-29.02944985, -4.24554001, 0.0]])
    s = time.perf_counter()
    # Example setup (you supply C/Cp/Cpp and P):
    A = bez_curve_entity(curve3.astype(Interval))
    B = bez_curve_entity(curve4.astype(Interval))
    F, J = make_stationary_system(A, B)

    X0 = box_from_bounds([0.,0.], [1.,1.])
    print(X0)
    roots, undecided = solve_all_roots_krawczyk(F,
                                                J, X0, tol=1e-6)
    print('time:',time.perf_counter()-s)
    for root in roots:
        print(root.x_mid.tolist())
    s=time.perf_counter()
    A = bez_curve_entity(curve3.astype(Interval))
    B = bez_curve_entity(curve6.astype(Interval))
    F, J = make_stationary_system(A, B)

    X0 = box_from_bounds([0.,0.], [1.,1.])
    print(X0)
    roots, undecided = solve_all_roots_krawczyk(F,
                                                J, X0, tol=1e-6)
    print('time:',time.perf_counter()-s)
    print(undecided)
    for root in roots:
        d=bernstein_eval_1d(curve3,root.x_mid[0])-bernstein_eval_1d(curve6,root.x_mid[1])
        if np.dot(d,d)<1e-6:

            print('intersection',root.x_mid.tolist())
        else:
            print('maximum',root.x_mid.tolist())

    s=time.perf_counter()
    A = bez_curve_entity(curve1.astype(Interval))
    B = bez_curve_entity(curve2.astype(Interval))
    F, J = make_stationary_system(A, B)

    X0 = box_from_bounds([0.,0.], [1.,1.])
    print(X0)
    roots, undecided = solve_all_roots_krawczyk(F,
                                                J, X0, tol=1e-6)
    print('time:',time.perf_counter()-s)
    print(undecided)
    for root in roots:
        d=bernstein_eval_1d(curve1,root.x_mid[0])-bernstein_eval_1d(curve2,root.x_mid[1])
        if np.dot(d,d)<1e-6:

            print('intersection',root.x_mid.tolist())
        else:
            print('maximum',root.x_mid.tolist())
