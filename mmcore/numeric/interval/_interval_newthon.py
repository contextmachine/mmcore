from mmcore.numeric.interval._interval import Interval,IntervalND

# ──────────────────────────────────────────────────────────
# 1. Certified Interval‑Newton root finder
# ──────────────────────────────────────────────────────────
def interval_newton(f_scalar, df_interval, X:Interval, tol=1e-10, max_depth=30):
    """Return list of disjoint intervals each provably containing a root of f"""
    result=[]
    def recurse(I, depth):
        # evaluate f, derivative
        m=I.mid()
        f_m=f_scalar(m)
        dI=df_interval(I)
        # contraction step if slope interval does not contain 0
        if 0 not in dI:
            N=Interval(m,m) - f_m / dI
            I_new=I & N
            if I_new is None:
                return  # root not in this interval
        else:
            I_new=I  # cannot contract
        if I_new.width() < tol:
            result.append(I_new)
            return
        if depth<=0:
            result.append(I_new)  # reached max depth
            return
        # subdivide and continue
        left,right=I_new._subdivide_step()
        recurse(left, depth-1)
        recurse(right, depth-1)
    recurse(X, max_depth)
    # merge overlapping intervals
    result.sort(key=lambda i: i.low)
    merged=[]
    for I in result:
        if not merged: merged.append(I); continue
        if I.low<=merged[-1].upp+tol:   # overlap or touch
            merged[-1]=Interval(merged[-1].low, max(merged[-1].upp, I.upp))
        else:
            merged.append(I)
    return merged

# ───────────────────────────────────────────────────────────────
# 2.  Multivariate Interval‑Newton (scalar equation f(x)=0)
# ───────────────────────────────────────────────────────────────
def interval_newton_nd(
        f,          # f(p)  – ordinary float evaluation at a point list

        grad_interval,     # ∇f(B) – list[Interval] of partial derivatives
        domain   : IntervalND,
        tol      = 1e-8,   # stop when every edge of the IntervalND < spt
        max_depth= 25):    # recursion safeguard
    """
    Certified root-finder for f:ℝⁿ→ℝ inside a *single* axis IntervalND.
    Returns a list of disjoint IntervalNDes, each guaranteed to contain ≥1 root.
    """
    roots = []
    def solve(B: IntervalND, depth: int):
        # 2.1  discard IntervalNDes whose image doesn’t straddle 0
        if 0 not in f(B):
            return
        # 2.2  Newton contraction:  xi  ←  mi − f(m) / ∂f/∂xi(B)
        m  = B.mid()
        fm = f(m)
        gB = grad_interval(B)
        contracted = []
        for i, (I, gI) in enumerate(zip(B.iv, gB)):
            if 0 not in gI:                       # safe to divide
                N  = Interval(m[i], m[i]) - fm / gI
                Ii = I & N
                if Ii is None:
                    # no intersection ⇒ no root
                    return
                contracted.append(Ii)
            else:
                # derivative interval hits 0
                contracted.append(I)
        Bc = IntervalND(contracted)
        # 2.3  termination tests
        if Bc.width() < tol or depth == 0:
            roots.append(Bc)
            return
        # 2.4  otherwise split and recurse
        L, R = Bc.bisect()
        solve(L, depth-1)
        solve(R, depth-1)
    solve(domain, max_depth)
    return roots

