import numpy as np


def nurbs_rational_snap_cert(ctrl4: np.ndarray,
                              M_world_to_clip: np.ndarray,
                              u: float, v: float,
                              eps: float = 0.0):
    """
    Cheap snap pre-filter for a rational Bézier curve.

    Parameters
    ----------
    ctrl4 : (n+1, 4) array
        Homogeneous control points [w*x, w*y, w*z, w].
    M_world_to_clip : (4,4) array
        Combined world->clip matrix (P*V or P*V*M if ctrl4 in object space).
    u, v : float
        Pixel NDC coordinates in [-1,1]. (Use the pixel center in NDC.)
    eps : float
        Tolerance band around each plane; eps=0 tests exact incidence,
        eps>0 creates a "snap cone" around the viewing line.

    Returns
    -------
    ok : bool
        True if the curve is a snap candidate.
    score : float
        A small nonnegative number estimating closeness; 0 means it crosses both planes,
        larger means farther. You can use it to rank candidates.
    """
    # Two world-space planes that define the pixel line
    n_x, n_y = pixel_planes_world(M_world_to_clip, u, v)  # shape (4,)

    # Evaluate planes on all homogeneous control points
    # di = n^T * P̂_i
    d_x = ctrl4 @ n_x
    d_y = ctrl4 @ n_y

    # Interval hulls (Bernstein convex-hull property)
    min_x, max_x = np.min(d_x), np.max(d_x)
    min_y, max_y = np.min(d_y), np.max(d_y)

    # Test whether each interval crosses the zero band [-eps, eps]
    def crosses_zero_band(a_min, a_max, tol):
        if tol <= 0.0:
            return (a_min <= 0.0) and (a_max >= 0.0)
        # Band test: interval intersects [-tol, tol]
        return not (a_max < -tol or a_min > tol)

    hit_x = crosses_zero_band(min_x, max_x, eps)
    hit_y = crosses_zero_band(min_y, max_y, eps)

    ok = bool(hit_x and hit_y)

    # A small ranking score: distance of each interval from the zero band, max of the two
    def interval_distance_to_band(a_min, a_max, tol):
        if a_min <= tol and a_max >= -tol:
            return 0.0
        return min(abs(a_min - tol), abs(a_max + tol))
    
    score_x = interval_distance_to_band(min_x, max_x, eps)
    score_y = interval_distance_to_band(min_y, max_y, eps)
    score = max(score_x, score_y)
    
    return ok, float(score)
import numpy as np

from mmcore.numeric.bern import bernstein_eval_1d,bern_roots_1d

# --- projective pixel-planes (same construction as before) -------------------
def pixel_planes_world(M_world_to_clip: np.ndarray, u_ndc: float, v_ndc: float):
    p_x = np.array([1.0, 0.0, 0.0, -u_ndc], dtype=float)  # x - u w = 0
    p_y = np.array([0.0, 1.0, 0.0, -v_ndc], dtype=float)  # y - v w = 0
    MinvT = np.linalg.inv(M_world_to_clip).T
    n_x = MinvT @ p_x
    n_y = MinvT @ p_y
    return n_x, n_y

# --- main: snap on a rational Bézier using your 1D Bernstein rooter ----------
def snap_bezier_rational_with_bern_roots(
    ctrl4: np.ndarray,                       # (n+1,4) homogeneous [wx,wy,wz,w] in WORLD space
    M_world_to_clip: np.ndarray,             # (4,4) PV or PVM
    u_ndc: float, v_ndc: float,              # pixel in NDC [-1,1]^2
    bern_roots_1d=bern_roots_1d,                           # your function
    eval_scalar=bernstein_eval_1d,           # scalar evaluator (can pass your own)
    eps_root: float = 1e-6,                  # tolerance used by your rooter
    cross_tol: float = 1e-6                  # |other residual| acceptance at candidate u
):
    """
    Returns (hit: bool, u_star: float|None, P_world: (3,)|None, score: float)
    score is sqrt(rx^2+ry^2) at u_star (smaller is better).
    """

    # 1) build world-space planes for the pixel line
    
    n_x, n_y = pixel_planes_world(M_world_to_clip, u_ndc, v_ndc)

    # 2) build scalar Bernstein coefficient arrays for r_x and r_y
    #    (dot plane with each homogeneous control point)
    rx_ctrl = (ctrl4 @ n_x).astype(float)     # shape (n+1,)
    ry_ctrl = (ctrl4 @ n_y).astype(float)     # shape (n+1,)

    # Quick cull: convex-hull band test—if either interval misses 0, no snap
    if (rx_ctrl.min() > 0 and rx_ctrl.max() > 0) or (rx_ctrl.min() < 0 and rx_ctrl.max() < 0):
        return False, None, None, float('inf')
    if (ry_ctrl.min() > 0 and ry_ctrl.max() > 0) or (ry_ctrl.min() < 0 and ry_ctrl.max() < 0):
        return False, None, None, float('inf')

    # 3) find 1D roots independently
    roots_x = bern_roots_1d(rx_ctrl, eps=eps_root).roots
    roots_y = bern_roots_1d(ry_ctrl, eps=eps_root).roots

    # 4) test cross-residuals at each candidate
    candidates = []
    for u in roots_x:
        if 0.0 < u < 1.0:
            ry = eval_scalar(ry_ctrl, float(u))
            if abs(ry) <= cross_tol:
                rx = eval_scalar(rx_ctrl, float(u))
                candidates.append((float(u), float(np.hypot(rx, ry))))
    for u in roots_y:
        if 0.0 < u < 1.0:
            rx = eval_scalar(rx_ctrl, float(u))
            if abs(rx) <= cross_tol:
                ry = eval_scalar(ry_ctrl, float(u))
                candidates.append((float(u), float(np.hypot(rx, ry))))

    if not candidates:
        return False, None, None, float('inf')

    # 5) pick best u by smallest combined residual
    u_star, score = min(candidates, key=lambda t: t[1])

    # 6) dehomogenize point on curve (cheap de Casteljau)
    #    small local eval to place the cursor; reuse your existing de Casteljau if preferred
    def eval_homo(ctrl: np.ndarray, u: float) -> np.ndarray:
        a = ctrl.astype(float).copy()
        for _ in range(len(a) - 1):
            a = (1.0 - u) * a[:-1] + u * a[1:]
        return a[0]

    Ch = eval_homo(ctrl4, u_star)
    w = Ch[3]
    if abs(w) < 1e-30:
        return False, None, None, score
    Pw = Ch[:3] / w

    return True, float(u_star), Pw, float(score)
import numpy as np

# --- Homogeneous Bézier evaluation ------------------------------------------
def bezier_eval_homog(ctrl4: np.ndarray, u: float) -> np.ndarray:
    """
    Evaluate homogeneous Bézier at parameter u ∈ [0,1].
    ctrl4: (n+1, 4) array of [w*x, w*y, w*z, w] in WORLD space.
    Returns a 4-vector [Xh, Yh, Zh, Wh] on the homogeneous curve.
    """
    a = ctrl4.astype(float).copy()
    n = a.shape[0] - 1
    for _ in range(n):
        a = (1.0 - u) * a[:-1] + u * a[1:]
    return a[0]

# --- NDC -> pixels helper ----------------------------------------------------
def ndc_to_pixels(ndc_xy: np.ndarray, width: int, height: int, origin: str = "top-left") -> np.ndarray:
    """
    Map NDC (-1..1) to pixel coords. origin: "top-left" or "bottom-left".
    """
    x_ndc, y_ndc = ndc_xy
    x_px = (x_ndc * 0.5 + 0.5) * width
    y_px_ndc_up = (y_ndc * 0.5 + 0.5) * height  # 0 at bottom if origin=bottom-left
    if origin == "top-left":
        y_px = height - y_px_ndc_up
    else:
        y_px = y_px_ndc_up
    return np.array([x_px, y_px], dtype=float)

# --- Main utility ------------------------------------------------------------
def cursor_from_curve_param(
    ctrl4: np.ndarray,
    u: float,
    M_world_to_clip: np.ndarray,
    viewport_size: tuple[int, int],
    origin: str = "top-left",
    eps_w: float = 1e-30,
):
    """
    Compute cursor/screen coordinates for a rational Bézier at parameter u.

    Parameters
    ----------
    ctrl4 : (n+1, 4)
        Homogeneous control points [w*x, w*y, w*z, w] in WORLD space.
    u : float
        Parameter in [0,1].
    M_world_to_clip : (4,4)
        Combined projection * view *( * model ) matrix mapping WORLD homogeneous to CLIP.
    viewport_size : (width, height)
        Viewport dimensions in pixels.
    origin : str
        "top-left" (usual UI) or "bottom-left" (GL convention).
    eps_w : float
        Guard for near-zero w.

    Returns
    -------
    result : dict
        {
          "pixel": (x_px, y_px),
          "ndc": (x_ndc, y_ndc, z_ndc),
          "clip": (x_clip, y_clip, z_clip, w_clip),
          "world": (x_world, y_world, z_world)
        }
    """
    width, height = viewport_size

    # 1) evaluate homogeneous world point on the curve
    Ch = bezier_eval_homog(ctrl4, u)  # [Xh, Yh, Zh, Wh]
    Wh = Ch[3]
    if abs(Wh) < eps_w:
        # Degenerate parameter (on/near w=0). Nudge u or early return.
        # Here we early-return with NaNs for world but still try to carry on for clip/ndc.
        world_pt = np.array([np.nan, np.nan, np.nan], dtype=float)
    else:
        world_pt = Ch[:3] / Wh

    # 2) world homogeneous -> clip
    clip = M_world_to_clip @ Ch  # [x', y', z', w']
    w_clip = clip[3]

    # 3) perspective divide -> NDC
    if abs(w_clip) < eps_w:
        # Off the view frustum / at infinity: mark NDC as NaN
        ndc = np.array([np.nan, np.nan, np.nan], dtype=float)
        pixel = np.array([np.nan, np.nan], dtype=float)
    else:
        ndc = clip[:3] / w_clip  # (x_ndc, y_ndc, z_ndc) in [-1,1]
        pixel = ndc_to_pixels(ndc[:2], width, height, origin=origin)

    return {
        "pixel": tuple(pixel.tolist()),
        "ndc": tuple(ndc.tolist()),
        "clip": tuple(clip.tolist()),
        "world": tuple(world_pt.tolist()),
    }
