from __future__ import annotations
import numpy as np

from mmcore.geom._nurbs_eval import NURBSSurfaceTuple, NURBSCurveTuple, NURBSTupleType
from mmcore.numeric.intersection.ssx import ssx

NURBSType =NURBSTupleType




# ---------- Core building blocks ----------

def _plane_coeffs(plane_point: np.ndarray,
                  plane_normal: np.ndarray,
                  *,
                  normalize_normal: bool = False) -> np.ndarray:
    """
    Return plane coefficients [a,b,c,d] for the plane passing through
    'plane_point' with normal 'plane_normal', i.e. a x + b y + c z + d = 0.

    If normalize_normal=True, the normal is normalized first (optional).
    """
    p0 = np.asarray(plane_point, dtype=float).reshape(3)
    n  = np.asarray(plane_normal, dtype=float).reshape(3)

    if not np.all(np.isfinite(n)) or not np.any(np.abs(n) > 0):
        raise ValueError("plane_normal must be a non-zero finite vector.")

    if normalize_normal:
        norm = np.linalg.norm(n)
        if norm == 0:
            raise ValueError("plane_normal has zero magnitude after normalization.")
        n = n / norm

    d = -np.dot(n, p0)
    return np.array([n[0], n[1], n[2], d], dtype=float)


def parallel_projection_matrix(plane_point: np.ndarray,
                               plane_normal: np.ndarray,
                               direction: np.ndarray,
                               *,
                               normalize_normal: bool = False) -> np.ndarray:
    """
    Construct a 4x4 matrix for parallel (directional) projection onto a plane.

    Parameters
    ----------
    plane_point : (3,) point on the plane.
    plane_normal : (3,) normal to the plane.
    direction : (3,) projection direction (must not be parallel to the plane).

    Returns
    -------
    T : (4,4) affine matrix whose last row is [0,0,0,1].

    Notes
    -----
    The mapping is affine:
        P' = [I - d n^T/(n·d)] P + ( (O·n)/(n·d) ) d,
    where O=plane_point, n=plane_normal, d=direction. We return the equivalent
    4x4 affine transform with last row [0,0,0,1].
    """
    n = np.asarray(plane_normal, dtype=float).reshape(3)
    d = np.asarray(direction, dtype=float).reshape(3)
    O = np.asarray(plane_point, dtype=float).reshape(3)

    if normalize_normal:
        nn = np.linalg.norm(n)
        if nn == 0:
            raise ValueError("plane_normal must be non-zero.")
        n = n / nn

    nd = float(np.dot(n, d))
    if np.isclose(nd, 0.0):
        raise ValueError("Projection direction is parallel to the plane (n·d = 0).")

    # A = I - (d n^T)/(n·d), t = ((O·n)/(n·d)) d
    A = np.eye(3) - np.outer(d, n) / nd
    t = (float(np.dot(O, n)) / nd) * d

    T = np.eye(4)
    T[:3, :3] = A
    T[:3,  3] = t
    # Last row is exactly [0,0,0,1] -> affine
    return T


def perspective_projection_matrix(plane_point: np.ndarray,
                                  plane_normal: np.ndarray,
                                  eye_point: np.ndarray,
                                  *,
                                  normalize_normal: bool = False) -> np.ndarray:
    """
    Construct a 4x4 projective matrix for perspective projection from 'eye_point'
    onto the plane defined by 'plane_point' and 'plane_normal'.

    Returns
    -------
    M : (4,4) projective matrix (not affine in general).

    Notes
    -----
    Let π = [a,b,c,d] be the plane (ax + by + cz + d = 0), and e = [Ex,Ey,Ez,1].
    The mapping that sends any point X to the intersection of line e->X with π is:

        M = (π·e) * I - e * π^T

    (This is the standard “planar shadow” / perspective projection matrix.)
    After multiplying homogeneous coordinates, dehomogenize as usual.
    """
    pi = _plane_coeffs(plane_point, plane_normal, normalize_normal=normalize_normal)
    e  = np.array([*np.asarray(eye_point, dtype=float).reshape(3), 1.0])
    s  = float(np.dot(pi, e))
    if np.isclose(s, 0.0):
        raise ValueError("eye_point lies on the plane (π·e = 0): perspective projection undefined.")

    M = s * np.eye(4) - np.outer(e, pi)  # shape (4,4)
    return M


# ---------- NURBS application ----------

def _apply_affine_to_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    A = T[:3, :3]
    t = T[:3, 3]
    return points @ A.T + t


def _apply_projective_to_rational(points: np.ndarray,
                                  weights: np.ndarray,
                                  T: np.ndarray,
                                  *,
                                  eps: float = 1e-14) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a general 4x4 projective matrix T to rational control points.

    points: shape (...,3)
    weights: shape (...,)

    Returns (new_points, new_weights). Raises if a resulting weight is ~0.
    """
    P = np.asarray(points, dtype=float).reshape(-1, 3)
    w = np.asarray(weights, dtype=float).reshape(-1, 1)
    Ph = np.concatenate([P * w, w], axis=1)        # (K,4) = [w x, w y, w z, w]
    PhT = Ph @ T.T                                  # row-vector convention
    w_new = PhT[:, 3]
    if np.any(np.isclose(w_new, 0.0, atol=eps)):
        idx = np.where(np.isclose(w_new, 0.0, atol=eps))[0][:5]
        raise ValueError(
            "Projection produced w' ≈ 0 for some control points (points at infinity). "
            f"Example indices: {idx.tolist()}."
        )
    P_new = PhT[:, :3] / w_new[:, None]
    return P_new.reshape(points.shape), w_new.reshape(weights.shape)


def _return_same_type(nurbs: NURBSType,
                      control_points: np.ndarray,
                      weights: np.ndarray) -> NURBSType:
    """Repack into the same tuple type, keeping orders and knots unchanged."""
    if isinstance(nurbs, NURBSCurveTuple):
        return NURBSCurveTuple(
            order=nurbs.order,
            knot=np.array(nurbs.knot, copy=True),
            control_points=control_points,
            weights=np.array(weights, copy=True),
        )
    elif isinstance(nurbs, NURBSSurfaceTuple):
        return NURBSSurfaceTuple(
            order_u=nurbs.order_u,
            order_v=nurbs.order_v,
            knot_u=np.array(nurbs.knot_u, copy=True),
            knot_v=np.array(nurbs.knot_v, copy=True),
            control_points=control_points,
            weights=np.array(weights, copy=True),
        )
    else:
        raise TypeError("nurbs must be NURBSCurveTuple or NURBSSurfaceTuple.")


def project_nurbs_parallel(
    nurbs: NURBSType,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    direction: np.ndarray,
    *,
    normalize_normal: bool = False,
) -> NURBSType:
    """
    Parallel-project a rational NURBS curve/surface onto a plane along 'direction'.

    This is an affine mapping; weights are unchanged.
    """
    T = parallel_projection_matrix(plane_point, plane_normal, direction,
                                   normalize_normal=normalize_normal)

    P = np.asarray(nurbs.control_points, dtype=float)
    w = np.asarray(nurbs.weights, dtype=float)
    P_new = _apply_affine_to_points(P, T)
    return _return_same_type(nurbs, P_new, w)  # weights unchanged

def transform_nurbs(
    nurbs: NURBSType,
    T: np.ndarray,
    *,
    enforce_affine: bool = True,
    atol: float = 1e-12,
) -> NURBSType:
    """
    Apply a 4x4 transform T to a rational NURBS curve/surface.

    Parameters
    ----------
    nurbs : NURBSCurveTuple | NURBSSurfaceTuple
        The input NURBS entity (rational, 3D control points).
    T : (4,4) array_like
        The transformation matrix. For affine transforms, the last row is [0,0,0,1]
        (within tolerance). Handles all affine transforms (rotate, translate, scale,
        shear, reflections). If `enforce_affine=False`, general projective transforms
        are supported via homogeneous control points.
    enforce_affine : bool, default True
        If True, raises on non-affine T; if False, applies general projective transform.
    atol : float, default 1e-12
        Tolerance for checking the affine last row.

    Returns
    -------
    A new NURBS tuple of the same type with transformed control points (and possibly
    transformed weights if `enforce_affine=False` and T is non-affine).

    Notes
    -----
    - For affine transforms on rational NURBS, weights do not change.
    - Knots and orders are unchanged.
    """
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"T must have shape (4,4), got {T.shape}.")

    is_affine = np.allclose(T[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=atol)
    if enforce_affine and not is_affine:
        raise ValueError(
            "Non-affine 4x4 matrix provided but enforce_affine=True. "
            "Either pass an affine matrix (last row [0,0,0,1]) or set enforce_affine=False."
        )

    # Dispatch by type
    if isinstance(nurbs, NURBSCurveTuple):
        P = np.asarray(nurbs.control_points, dtype=float)
        w = np.asarray(nurbs.weights, dtype=float)

        if is_affine:
            P_new = _apply_affine_to_points(P, T)
            # Weights unchanged under affine transforms
            w_new = w.copy()
        else:
            # General projective case (only when enforce_affine=False)
            P_new, w_new = _apply_projective_to_rational(P, w, T)

        return NURBSCurveTuple(
            order=nurbs.order,
            knot=np.array(nurbs.knot, copy=True),
            control_points=P_new,
            weights=np.array(w_new, copy=True),
        )

    elif isinstance(nurbs, NURBSSurfaceTuple):
        P = np.asarray(nurbs.control_points, dtype=float)
        w = np.asarray(nurbs.weights, dtype=float)

        if is_affine:
            P_new = _apply_affine_to_points(P, T)  # broadcasts over (N,M,3)
            w_new = w.copy()
        else:
            P_new, w_new = _apply_projective_to_rational(P, w, T)

        return NURBSSurfaceTuple(
            order_u=nurbs.order_u,
            order_v=nurbs.order_v,
            knot_u=np.array(nurbs.knot_u, copy=True),
            knot_v=np.array(nurbs.knot_v, copy=True),
            control_points=P_new,
            weights=np.array(w_new, copy=True),
        )

    else:
        raise TypeError(
            "nurbs must be an instance of NURBSCurveTuple or NURBSSurfaceTuple."
        )

from typing import Union, Tuple
import numpy as np

# --- Types from your setup (assumed available) ---
# class NURBSCurveTuple(NamedTuple):
#     order:int
#     knots: np.ndarray  # shape (N,)
#     control_points: np.ndarray  # shape (N,3)
#     weights: np.ndarray  # shape (N,)

# class NURBSSurfaceTuple(NamedTuple):
#     order_u:int
#     order_v:int
#     knot_u: np.ndarray  # shape (Nu,)
#     knot_v: np.ndarray  # shape (Nv,)
#     control_points: np.ndarray  # shape (Nu, Nv, 3)
#     weights: np.ndarray  # shape (Nu, Nv)

NURBSType = Union["NURBSCurveTuple", "NURBSSurfaceTuple"]

# --- transform_nurbs is assumed from earlier message ---
# def transform_nurbs(nurbs: NURBSType, T: np.ndarray, *, enforce_affine: bool = True, atol: float = 1e-12) -> NURBSType:
#     ...

# =========================
# Utilities (private)
# =========================

_EPS = 1e-12

def _as_vec3(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    if v.size < 3:
        raise ValueError("Vector must have at least 3 components.")
    return v[:3]

def _normalize(v: np.ndarray, *, eps: float = _EPS) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError("Cannot normalize near-zero vector.")
    return v / n

def _skew(u: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix for cross product: [u]_x v = u × v."""
    ux, uy, uz = u
    return np.array([[0.0, -uz,  uy],
                     [uz,  0.0, -ux],
                     [-uy, ux,  0.0]], dtype=float)

def _build_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Assemble a 4x4 from 3x3 and 3x1 using row-vector convention in transform_nurbs."""
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3]  = _as_vec3(t)
    return T

def _frame_from_plane_or_cplane(plane: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (origin, xhat, yhat, zhat) as an orthonormal frame.
    plane: shape (2,3) or (4,3).
    """
    P = np.asarray(plane, dtype=float)
    if P.shape == (2, 3):
        O = P[0]
        z = _normalize(P[1])
        # Build x,y orthonormal to z
        # Pick helper 'a' not parallel to z
        a = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = _normalize(np.cross(a, z))
        y = _normalize(np.cross(z, x))
        return O, x, y, z
    elif P.shape == (4, 3):
        O, x, y, z = P[0], P[1], P[2], P[3]  # normal as plane[-1] per requirement
        # Normalize to ensure pure rotation w/out scale
        x = _normalize(x); y = _normalize(y); z = _normalize(z)
        return O, x, y, z
    else:
        raise ValueError("plane must have shape (2,3) [origin,normal] or (4,3) [origin,xaxis,yaxis,normal].")

def _axis_from_plane(plane: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (origin, normalized_axis) from Plane or CPlane."""
    P = np.asarray(plane, dtype=float)
    O = P[0]
    n = _normalize(P[-1])  # plane[-1] works for both shapes by design
    return O, n

def _affine_rotate_about_axis(origin: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotation about 'axis' through 'origin' by 'angle' radians.
    Returns 4x4.
    """
    O = _as_vec3(origin)
    k = _normalize(_as_vec3(axis))
    K = _skew(k)
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    R = np.eye(3) * c + (1.0 - c) * np.outer(k, k) + s * K
    # About origin O: p' = O + R (p - O) => R, t = O - R O
    t = O - R @ O
    return _build_T(R, t)

def _affine_mirror_about_plane(origin: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Reflection across plane (origin, normal)."""
    O = _as_vec3(origin)
    n = _normalize(_as_vec3(normal))
    R = np.eye(3) - 2.0 * np.outer(n, n)
    t = O - R @ O   # (equivalently 2 * n * (n·O))
    return _build_T(R, t)

def _affine_project_orth(origin: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Orthogonal projection onto plane (origin, normal)."""
    O = _as_vec3(origin)
    n = _normalize(_as_vec3(normal))
    R = np.eye(3) - np.outer(n, n)
    t = O - R @ O   # (equivalently n * (n·O))
    return _build_T(R, t)

def _affine_project_along(origin: np.ndarray, normal: np.ndarray, direction: np.ndarray, *, eps: float = _EPS) -> np.ndarray:
    """
    Oblique projection along 'direction' onto plane (origin, normal).
    """
    O = _as_vec3(origin)
    n = _normalize(_as_vec3(normal))
    d = _as_vec3(direction)
    denom = float(n @ d)
    if abs(denom) < eps:
        raise ValueError("Projection direction is parallel to the plane; projection undefined.")
    # Column-vector form: p' = (I - d n^T / (n·d)) p + d * (n·O)/(n·d)
    R = np.eye(3) - np.outer(d, n) / denom
    t = d * (float(n @ O) / denom)
    return _build_T(R, t)

def _affine_orient(source_cplane: np.ndarray, target_cplane: np.ndarray) -> np.ndarray:
    """
    Map source CPlane frame to target CPlane frame:
      p' = O_t + F_t F_s^T (p - O_s)
    where F_* has columns (x,y,z).
    """
    Os, xs, ys, zs = _frame_from_plane_or_cplane(source_cplane)
    Ot, xt, yt, zt = _frame_from_plane_or_cplane(target_cplane)
    Fs = np.column_stack([xs, ys, zs])
    Ft = np.column_stack([xt, yt, zt])
    R = Ft @ Fs.T
    t = Ot - R @ Os
    return _build_T(R, t)

def _affine_shear_in_cplane(plane_c: np.ndarray, shx: float, shy: float) -> np.ndarray:
    """
    Shear in the given CPlane with factors along x and y relative to the plane's normal:
        local mapping: [u', v', w']^T = [ [1,0,shx],[0,1,shy],[0,0,1] ] [u, v, w]^T
    """
    O, x, y, z = _frame_from_plane_or_cplane(plane_c)
    F = np.column_stack([x, y, z])  # 3x3
    S_loc = np.array([[1.0, 0.0, shx],
                      [0.0, 1.0, shy],
                      [0.0, 0.0, 1.0]], dtype=float)
    R = F @ S_loc @ F.T
    t = O - R @ O
    return _build_T(R, t)

def _minimal_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    3x3 rotation that maps unit vector a -> b with minimal angle.
    Handles near-parallel and antiparallel cases robustly.
    """
    a = _normalize(a); b = _normalize(b)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(np.dot(a, b))
    if s < 1e-15:
        # a and b are parallel or antiparallel
        if c > 0.0:
            return np.eye(3)
        # 180°: choose any axis orthogonal to a
        u = _normalize(np.cross(a, np.array([1.0, 0.0, 0.0])) if abs(a[0]) < 0.9 else np.cross(a, np.array([0.0, 1.0, 0.0])))
        return 2.0 * np.outer(u, u) - np.eye(3)
    K = _skew(v / s)  # normalized axis
    # Rodrigues with known c, s
    return np.eye(3) + K * s + (K @ K) * (1.0 - c)

# =========================
# Public API (requested)
# =========================

def combine_transform(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """
    Combine two 4x4 transforms into one.
    Semantics (row-vector pipeline as used in transform_nurbs):
      result = t2 @ t1    # applying t1, then t2.
    """
    T1 = np.asarray(t1, dtype=float)
    T2 = np.asarray(t2, dtype=float)
    if T1.shape != (4, 4) or T2.shape != (4, 4):
        raise ValueError("Both t1 and t2 must have shape (4,4).")
    return T2 @ T1

def translate(nurbs_object: NURBSType, direction: np.ndarray) -> NURBSType:
    """Translate by vector `direction`."""
    t = _as_vec3(direction)
    T = _build_T(np.eye(3), t)
    return transform_nurbs(nurbs_object, T)

def scale_uniform(nurbs_object: NURBSType, factor: float = 1.0) -> NURBSType:
    """
    Uniform scale about world origin by `factor`.
    (To scale about a point P, compose: T = T_translate(P) ◦ Scale ◦ T_translate(-P).)
    """
    s = float(factor)
    R = np.diag([s, s, s])
    T = _build_T(R, np.zeros(3))
    return transform_nurbs(nurbs_object, T)

def scale_non_uniform(
    nurbs_object: NURBSType,
    factor_x: float = 1.0,
    factor_y: float | None = None,
    factor_z: float | None = None,
) -> NURBSType:
    """
    Non-uniform scale about world origin.
    If factor_y or factor_z is None, they default to factor_x.
    """
    sx = float(factor_x)
    sy = float(factor_x if factor_y is None else factor_y)
    sz = float(factor_x if factor_z is None else factor_z)
    R = np.diag([sx, sy, sz])
    T = _build_T(R, np.zeros(3))
    return transform_nurbs(nurbs_object, T)

def scale(nurbs_object: NURBSType, *args, **kwargs) -> NURBSType:
    """
    Wrapper for scale:
      - scale(obj, s) -> uniform
      - scale(obj, sx, sy, sz) -> non-uniform
      Also supports keywords: factor, factor_x, factor_y, factor_z.
    """
    if len(args) == 1 and not kwargs:
        return scale_uniform(nurbs_object, factor=float(args[0]))
    if len(args) == 3 and not kwargs:
        fx, fy, fz = args
        return scale_non_uniform(nurbs_object, float(fx), float(fy), float(fz))

    # Keyword routes
    if "factor" in kwargs and not any(k in kwargs for k in ("factor_x", "factor_y", "factor_z")):
        return scale_uniform(nurbs_object, factor=float(kwargs["factor"]))
    return scale_non_uniform(
        nurbs_object,
        factor_x=float(kwargs.get("factor_x", kwargs.get("factor", 1.0))),
        factor_y=float(kwargs.get("factor_y", None)),
        factor_z=float(kwargs.get("factor_z", None)),
    )

# ----- Plane / CPlane-based operations -----

def rotate(nurbs_object: NURBSType, plane: np.ndarray) -> NURBSType:
    """
    Pure rotation determined by plane orientation (no translation).
    - If plane is CPlane (4x3): rotate so world axes map to (xaxis,yaxis,normal).
    - If plane is Plane  (2x3): rotate so world Z aligns to plane normal (minimal rotation).
    """
    P = np.asarray(plane, dtype=float)
    if P.shape == (4, 3):
        _, x, y, z = _frame_from_plane_or_cplane(P)
        R = np.column_stack([x, y, z])  # columns are images of world basis
        T = _build_T(R, np.zeros(3))
    else:
        # Minimal rotation taking world Z -> plane normal
        _, n = _axis_from_plane(P)
        R = _minimal_rotation(np.array([0.0, 0.0, 1.0]), n)
        T = _build_T(R, np.zeros(3))
    return transform_nurbs(nurbs_object, T)

def mirror(nurbs_object: NURBSType, plane: np.ndarray) -> NURBSType:
    """Mirror across the given plane (origin, normal)."""
    O, n = _axis_from_plane(plane)
    T = _affine_mirror_about_plane(O, n)
    return transform_nurbs(nurbs_object, T)

def project(nurbs_object: NURBSType, plane: np.ndarray) -> NURBSType:
    """Orthogonal projection onto the given plane."""
    O, n = _axis_from_plane(plane)
    T = _affine_project_orth(O, n)
    return transform_nurbs(nurbs_object, T)

def project_along(nurbs_object: NURBSType, plane: np.ndarray, direction: np.ndarray) -> NURBSType:
    """Oblique projection along `direction` onto plane."""
    O, n = _axis_from_plane(plane)
    T = _affine_project_along(O, n, direction)
    return transform_nurbs(nurbs_object, T)

def orient(nurbs_object: NURBSType, source: np.ndarray, target: np.ndarray) -> NURBSType:
    """
    Map `source` CPlane to `target` CPlane (rotation + translation).
    World coordinates are transformed so that the source frame coincides with the target frame.
    """
    T = _affine_orient(source, target)
    return transform_nurbs(nurbs_object, T)

def shear(nurbs_object: NURBSType, plane: np.ndarray, grip: np.ndarray, target: np.ndarray) -> NURBSType:
    """
    Shear in the plane so that a reference point (grip) is moved *within the plane directions*
    toward `target` while preserving its signed distance to the plane.
    Semantics:
      - Convert grip/target to local (u,v,w). If w_grip ≈ 0, shear is undefined (would require ∞).
      - Set shx = (u_target - u_grip)/w_grip,  shy = (v_target - v_grip)/w_grip.
      - Apply shear: u' = u + shx*w, v' = v + shy*w, w' = w.
    Note: target's w component is ignored; shear keeps w invariant.
    """
    # Ensure we have a CPlane frame
    if np.asarray(plane).shape == (2, 3):
        # Build a CPlane from the simple Plane to avoid extra checks.
        O, x, y, z = _frame_from_plane_or_cplane(plane)
        plane_c = np.vstack([O, x, y, z])
    else:
        plane_c = np.asarray(plane, dtype=float)

    O, x, y, z = _frame_from_plane_or_cplane(plane_c)
    F = np.column_stack([x, y, z])  # local -> world
    Finv = F.T                      # world -> local (orthonormal)

    g = _as_vec3(grip)
    t = _as_vec3(target)
    ug, vg, wg = Finv @ (g - O)
    ut, vt, wt = Finv @ (t - O)

    if abs(wg) < 1e-14:
        raise ValueError("Grip point lies on the plane (w ≈ 0); shear factors would be unbounded.")

    shx = (ut - ug) / wg
    shy = (vt - vg) / wg

    T = _affine_shear_in_cplane(plane_c, shx=shx, shy=shy)
    return transform_nurbs(nurbs_object, T)

def shear_angle(nurbs_object: NURBSType, plane: np.ndarray, angle_x: float, angle_y: float) -> NURBSType:
    """
    Shear using angles (radians) along the plane's x- and y-axes.
    Conventions:
      shx = tan(angle_x),  shy = tan(angle_y)
      Local mapping: u' = u + shx*w, v' = v + shy*w, w' = w
    """
    shx = float(np.tan(angle_x))
    shy = float(np.tan(angle_y))

    # Ensure we have a CPlane
    if np.asarray(plane).shape == (2, 3):
        O, x, y, z = _frame_from_plane_or_cplane(plane)
        plane_c = np.vstack([O, x, y, z])
    else:
        plane_c = np.asarray(plane, dtype=float)

    T = _affine_shear_in_cplane(plane_c, shx=shx, shy=shy)
    return transform_nurbs(nurbs_object, T)

def project_nurbs_perspective(
    nurbs: NURBSType,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    eye_point: np.ndarray,
    *,
    normalize_normal: bool = False,
    eps: float = 1e-14,
) -> NURBSType:
    """
    Perspective-project a rational NURBS curve/surface from 'eye_point' onto a plane.

    This is a projective mapping; weights generally change.
    """
    M = perspective_projection_matrix(plane_point, plane_normal, eye_point,
                                      normalize_normal=normalize_normal)

    P = np.asarray(nurbs.control_points, dtype=float)
    w = np.asarray(nurbs.weights, dtype=float)
    P_new, w_new = _apply_projective_to_rational(P, w, M, eps=eps)
    return _return_same_type(nurbs, P_new, w_new)


if __name__ == "__main__":
    from mmcore.construction import circle,cylinder_surface_2pt
    from mmcore.numeric.vectors import scalar_unit
    
    curve = circle(2.0, center=np.array([5.,5., 0.]), normal=scalar_unit(np.array((1., 1., 1.))))
    print(curve)
    start=np.array([0.533136, -2.144876, -1])
    end=np.array([2.294869, -0.144876, 0.683482])
    
    surface=cylinder_surface_2pt(start,end, 2.0)
    # Curve example
    T = np.array([
        [0.0, -1.0, 0.0, 2.0],  # rotate 90° about z and translate by (2,0,0)
        [1.0, 0.0, 0.0, -1.0],
        [0.0, 0.0, 1.0, 0.5],
        [0.0, 0.0, 0.0, 1.0],
    ])
  
    curve2 = transform_nurbs(curve, T)  # curve is a NURBSCurveTuple
    print(curve2)
    # Surface example

    surface2 = transform_nurbs(surface, T)  # surface is a NURBSSurfaceTuple
 
    # 1) Parallel projection onto the XY-plane along +Z (standard orthographic)
    surface_xy = project_nurbs_parallel(
        surface,  # NURBSSurfaceTuple
        plane_point=np.array([0, 0, 0]),
        plane_normal=np.array([0, 0, 1]),
        direction=np.array([0, 0, 1])
    )
  
    # 2) Oblique projection onto the plane z = 1 along direction (1,1,1)
    curve_oblique = project_nurbs_parallel(
        curve,  # NURBSCurveTuple
        plane_point=np.array([0, 0, 1]),
        plane_normal=np.array([0, 0, 1]),
        direction=np.array([1, 1, 1])
    )
    print(curve_oblique)
    # 3) Perspective projection onto z=0 from eye at (0,0,5)
    curve_persp = project_nurbs_perspective(
        curve,
        plane_point=np.array([0, 0, 0]),
        plane_normal=np.array([0, 0, 1]),
        eye_point=np.array([0, 0, 5])
    )
    print(curve_persp)
    
    print(surface)
    print(surface2)
    print(surface_xy)
    result = ssx(surface, surface2, tol=1e-6, spt=1e-3)
    try:
        from mmcore.extras.renderer import CADRenderer, Camera
        
        from mmcore.geom._nurbs_eval import _nurbs_to_tuple, _tuple_to_nurbs
        def draw_ssx(s1, s2, result, renderer=None):
            if isinstance(s1, NURBSSurfaceTuple):
                s1 = _tuple_to_nurbs(s1)
            if isinstance(s2, NURBSSurfaceTuple):
                s2 = _tuple_to_nurbs(s2)
            
            renderer = renderer if renderer is not None else CADRenderer(camera=Camera(zoom=50.0, near=1.))
            
            renderer.add_nurbs_surface(s1, color=(1.0, 1.0, 1.0))
            renderer.add_nurbs_surface(s2, color=(1.0, 1.0, 1.0), )
            
            for crv, uv1, uv2 in result[0]:
                renderer.add_nurbs_curve(crv, color=(0.0, 1.0, 0.5))
            return renderer
        
        from scipy.interpolate import interp1d
        def interp_color(colors, params, new_params):
          
           
            return interp1d(params, np.asarray(colors).T)(new_params).T
        
        
        renderer:CADRenderer = draw_ssx(surface, surface2, result)
        curves_colors = interp_color(np.array([(0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]), np.linspace(0, 1, 2),
                          np.linspace(0, 1, 4)).tolist()
        
        renderer.add_nurbs_curve(_tuple_to_nurbs(curve), color=tuple(curves_colors[0]))
        renderer.add_nurbs_curve(_tuple_to_nurbs(curve2), color=tuple(curves_colors[1]))
        renderer.add_nurbs_curve(_tuple_to_nurbs(curve_oblique), color=tuple(curves_colors[2]))
        renderer.add_nurbs_curve(_tuple_to_nurbs(curve_persp), color=tuple(curves_colors[3]))
        renderer.run()
        
    except ImportError as err:
        print("mmcore.renderer is not installed, skip preview.")
        for i, (spatial, uv1, uv2) in enumerate(result[0]):
            print(f"\t{i + 1}. {spatial}, {uv1}, {uv2}")
            cpts = (spatial.control_points).tolist()
            cpts_repr = repr(cpts)
            # if len(cpts)>4:
            #    cpts_repr=f'[{cpts[1]}, {cpts[2]}, ... , {cpts[-2]}, {cpts[-1]}]'
            print(f"\t\tcontrol points: {cpts_repr}")
            print(f"\t\tdegree: {spatial.degree}")
            
   
  
    