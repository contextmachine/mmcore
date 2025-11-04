import numpy as np
from mmcore.geom._nurbs_eval import NURBSCurveTuple, evaluate_nurbs_curve



def _nurbs_curve_param_tol_conservative(P, w, U, p, tol):
    """
    P : (n, dim)   original control points
    w : (n,)       original positive weights
    U : (n + p + 1,) knot vector ("flat knots")
    p : int        degree
    tol : float    3D tolerance
    Returns: tol_u (parametric tolerance)
    """
    n, dim = P.shape
    assert U.shape[0] == n + p + 1
    min_w = float(np.min(w))
    if min_w <= 0:
        raise ValueError("All original weights must be > 0 for this bound.")
    Lmax = 0.0
    # edges i-1 -> i, with their knot-block [U_i, U_{i+p}]
    for i in range(1, n):
        du = U[i + p] - U[i]
        if du <= 0:
            continue  # degenerate or repeated knots; skip or handle specially
        inv_du = 1.0 / du
        # OCC's neighborhood: [i-(p+1), i+(2p+1))
        lower = max(0, i - (p + 1))
        upper = min(n, i + (2 * p + 1))
        Wi, Wim1 = w[i], w[i - 1]
        Pi, Pim1 = P[i], P[i - 1]
        # inner max over j in neighborhood
        # vector form; loop is fine too if you prefer clarity
        Pj = P[lower:upper]                       # (m, dim)
        term = np.abs((Pj - Pi) * Wi - (Pj - Pim1) * Wim1).sum(axis=1)  # (m,)
        value = term.max() * inv_du
        if value > Lmax:
            Lmax = value
    # degree scaling and denominator lower bound, exactly like OCC
    L = (p * Lmax) / min_w
    # final parametric tolerance
    RealSmall = np.finfo(float).tiny  # mirror OCC's "RealSmall()" idea
    tol_u = tol / max(L, RealSmall)
    return tol_u

from ._nurbs_ders import derivative_nurbs
_TINY=np.finfo(float).tiny
def _nurbs_curve_param_tol_optimistic(curve: NURBSCurveTuple, tol: float, der:NURBSCurveTuple=None) -> float:
    if np.allclose(curve.control_points,0):
        return tol
    if der is None:
        der=derivative_nurbs(curve)
    u0, u1 = curve.interval()
    du = (u1 - u0)
    cpts=np.abs(der.control_points)
    res=np.linalg.norm(cpts, axis=1).max()
    if res<_TINY:
        return tol
    
    tol_u = tol * (du / res)
    return tol_u

def nurbs_curve_param_tolerance(curve: NURBSCurveTuple, tol: float, der:NURBSCurveTuple=None) -> float:
    if np.any(curve.weights<0):
        return _nurbs_curve_param_tol_conservative(curve.control_points, curve.weights, curve.knot, curve.order - 1, tol)
    else:
        return _nurbs_curve_param_tol_optimistic(curve,tol,der)
    
    
if __name__=="__main__":
    import tqdm
    from mmcore.geom._nurbs_knots import generate_knots
    from itertools import pairwise
    
    color_interpolation = NURBSCurveTuple(3, np.array([1., 1., 1., 0., 0., 0.
                                                       ]), np.array(
        [[27 / 255, 222 / 255, 95 / 255], [222 / 255, 219 / 255, 27 / 255], [222 / 255, 27 / 255, 79 / 255]]),
                                          np.array([1., 1., 1.]))
    
    
    def rgb_to_hex(color: np.ndarray) -> str:
        """
        Convert an RGB color given as a NumPy array of floats in [0, 1]
        to a hex string in the format '#rrggbb'.

        Parameters
        ----------
        color : np.ndarray
            A 1D array of length 3 with float values in the range [0.0, 1.0].

        Returns
        -------
        str
            A hex color string, e.g. '#326fa8'.
        """
        # Ensure input is the right shape
        if color.shape != (3,):
            raise ValueError(f"Expected color array of shape (3,), got {color.shape}")
        
        # Clip values to [0,1], scale to [0,255], and convert to integers
        rgb_int = np.clip(color, 0.0, 1.0) * 255
        r, g, b = rgb_int.astype(int)
        
        return f"#{r:02x}{g:02x}{b:02x}"
    
    
    def test_approach(curve: NURBSCurveTuple, tol=1e-3):
    
        u0, u1 = curve.interval()
        du = (u1 - u0)
        tol_u = nurbs_curve_param_tolerance(curve, tol)
        steps, _ = divmod(du, tol_u)
        
        steps = int(steps)
        steps_sizes = np.ones(steps)
        steps_sizes[:] = tol_u
        u0_vals = np.clip(np.cumsum(steps_sizes), u0, u1)
        
        u1_vals = np.clip(u0_vals + tol_u, u0, u1)
        mask = ~np.isclose(u1_vals - u0_vals, 0)
        u0_vals = u0_vals[mask]
        u1_vals = u1_vals[mask]
        red = tol
        green = tol / 2
        color_interpolation = NURBSCurveTuple(3, np.array([green, green, green, red, red, red
                                                           ]), np.array(
            [[27 / 255, 222 / 255, 95 / 255], [222 / 255, 219 / 255, 27 / 255], [222 / 255, 27 / 255, 79 / 255]]),
                                              np.array([1., 1., 1.]))
        
        prec = int(np.abs(np.log10(tol_u)).item()) + 2
        
        print(u0_vals + tol_u)
        pb = tqdm.tqdm(zip(u0_vals, u1_vals), total=steps, dynamic_ncols=True,
                       colour=rgb_to_hex(evaluate_nurbs_curve(color_interpolation, 0)['C']))
        best = float('inf')
        wrong = -float('inf')
        
        def format_num(n):
            return f"{n:{1}.{prec}f}"
        
        def fun(t0, t1):
            nonlocal best, wrong, pb, prec
            val = float(np.linalg.norm(evaluate_nurbs_curve(curve, t0)['C'] - evaluate_nurbs_curve(curve, t1)['C']))
            
            col = evaluate_nurbs_curve(color_interpolation, val)['C']
            pb.colour = rgb_to_hex(col)
            best = float(min(best, val))
            wrong = float(max(wrong, val))
            
            pb.set_description_str(
                f'({format_num(t0)},{format_num(t1)}; ptol: {tol_u} curr: {format_num(val)}; best: {format_num(best)}; wrong: {format_num(wrong)}')
        
        for interv in pb:
            fun(interv[0], interv[1])
    
    from mmcore.construction import circle
    curve3 = circle(10, normal=np.array((1.,1.,0.5))/np.linalg.norm((1.,1.,0.5)))
    curve4 = np.array([[-45.36434109, -7.12015504, 0.],
                       [-25.49612403, 13.94186047, 0.],
                       [-2.13178295, -17.35271318, 0.],
                       [12.02325581, 20.42248062, 0.]])
    tpl = NURBSCurveTuple(4, generate_knots(curve4.shape[0], 3), curve4, np.ones((curve4.shape[0])))

    #test_approach(tpl)
    test_approach(curve3)