from __future__ import annotations

import ctypes
import math
import sys
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Literal, TypedDict

import glfw
import numpy as np
from OpenGL.GL import *

from mmcore.geom._nurbs_eval import (
    NURBSCurveTuple,
    NURBSSurfaceTuple,
    _tuple_to_nurbs,
    to_homogeneous_1d,
)
from mmcore.geom._nurbs_knots import decompose_curve
from mmcore.geom.bvh.lbvh import AABB
from mmcore.geom.nurbs_iso import extract_isocurve, extract_surface_boundaries
from mmcore.topo.mesh.tess import surface_to_mesh, tessellate_brep_face


# =========================
# Matrix utilities (row-major)
# =========================

def orthographic(left, right, bottom, top, near, far) -> np.ndarray:
    """Standard row-major orthographic projection (OpenGL clip convention)."""
    rl = right - left
    tb = top - bottom
    fn = far - near
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = 2.0 / rl
    m[1, 1] = 2.0 / tb
    m[2, 2] = -2.0 / fn
    m[0, 3] = -(right + left) / rl
    m[1, 3] = -(top + bottom) / tb
    m[2, 3] = -(far + near) / fn
    return m


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n < eps else v / n


def look_at(eye, target, up) -> np.ndarray:
    """Row-major LookAt. Upload with transpose=True."""
    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    f = normalize(target - eye)
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)

    m = np.eye(4, dtype=np.float32)
    m[0, 0:3] = s
    m[1, 0:3] = u
    m[2, 0:3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


# =========================
# Viewport + conversions
# =========================

@dataclass
class ViewportInfo:
    vx: int
    vy: int
    vw: int
    vh: int
    fb_w: int
    fb_h: int
    sx: float
    sy: float
    win_w: int
    win_h: int


def read_viewport_info(window) -> ViewportInfo:
    vx, vy, vw, vh = glGetIntegerv(GL_VIEWPORT)
    fb_w, fb_h = glfw.get_framebuffer_size(window)
    sx, sy = glfw.get_window_content_scale(window)
    win_w, win_h = glfw.get_window_size(window)
    return ViewportInfo(
        int(vx),
        int(vy),
        int(vw),
        int(vh),
        int(fb_w),
        int(fb_h),
        float(sx),
        float(sy),
        int(win_w),
        int(win_h),
    )


def points_to_pixels(xy_pt, v: ViewportInfo) -> np.ndarray:
    """GLFW 'points' (origin top-left) -> framebuffer 'pixels' (origin bottom-left)."""
    x_px = xy_pt[0] * v.sx
    y_px_top = xy_pt[1] * v.sy
    y_px = (v.vy + v.vh) - y_px_top
    return np.array([x_px, y_px], dtype=np.float64)


def pixels_to_points(xy_px, v: ViewportInfo) -> np.ndarray:
    """Framebuffer pixels (origin bottom-left) -> GLFW points (origin top-left)."""
    x_pt = xy_px[0] / v.sx
    y_top = (v.vy + v.vh) - xy_px[1]
    y_pt = y_top / v.sy
    return np.array([x_pt, y_pt], dtype=np.float64)


def pixels_to_ndc(xy_px, v: ViewportInfo) -> np.ndarray:
    x_ndc = ((xy_px[0] - v.vx) / v.vw) * 2.0 - 1.0
    y_ndc = ((xy_px[1] - v.vy) / v.vh) * 2.0 - 1.0
    return np.array([x_ndc, y_ndc], dtype=np.float64)


def ndc_to_pixels(ndc_xy, v: ViewportInfo) -> np.ndarray:
    x_px = v.vx + (ndc_xy[0] + 1.0) * 0.5 * v.vw
    y_px = v.vy + (ndc_xy[1] + 1.0) * 0.5 * v.vh
    return np.array([x_px, y_px], dtype=np.float64)


def glfw_cursor_to_ndc(cursor_xy_points, v: ViewportInfo) -> np.ndarray:
    return pixels_to_ndc(points_to_pixels(cursor_xy_points, v), v)


# =========================
# Camera (orbit / pan / zoom)
# =========================

class OrbitCamera:
    def __init__(
        self,
        target=(0.0, 0.0, 0.0),
        near=0.1,
        far=10000.0,
        up=(0.0, 0.0, 1.0),
        distance=100.0,
        ortho_half_height=5.0,
        yaw=math.radians(35.0),
        pitch=math.radians(30.0),
    ):
        self.target = np.array(target, dtype=np.float32)
        self.distance = distance
        self.yaw = yaw
        self.pitch = pitch
        self.up_world = np.array(up, dtype=np.float32)  # Z-up CAD
        self.ortho_half_height = ortho_half_height
        self.near = near
        self.far = far
        self._lock_orbit = False

    def lock_orbit(self, lock: bool):
        self._lock_orbit = lock

    def eye(self) -> np.ndarray:
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        dir_world = np.array([cp * cy, cp * sy, sp], dtype=np.float32)
        return self.target + (-self.distance) * dir_world

    def view_matrix(self) -> np.ndarray:
        return look_at(self.eye(), self.target, self.up_world)

    def projection_matrix(self, aspect: float) -> np.ndarray:
        h = self.ortho_half_height
        w = h * aspect
        return orthographic(-w, +w, -h, +h, self.near, self.far)

    def orbit(self, dx_pixels: float, dy_pixels: float, v: ViewportInfo):
        if self._lock_orbit:
            return
        s = 2.0 * math.pi / max(v.vw, v.vh)
        self.yaw -= dx_pixels * s
        self.pitch -= dy_pixels * s
        self.pitch = max(-math.radians(89.0), min(math.radians(89.0), self.pitch))

    def pan(self, dx_pixels: float, dy_pixels: float, v: ViewportInfo):
        aspect = v.vw / max(1, v.vh)
        h = self.ortho_half_height
        w = h * aspect
        dx_ndc = (dx_pixels / max(1, v.vw)) * 2.0
        dy_ndc = (dy_pixels / max(1, v.vh)) * 2.0
        delta_world = np.array([dx_ndc * w, dy_ndc * h, 0.0], dtype=np.float32)

        V = self.view_matrix()
        right = V[0, 0:3]
        up = V[1, 0:3]
        move = right * (-delta_world[0]) + up * (delta_world[1])
        self.target += move

    def zoom_wheel(self, yoffset: float):
        factor = math.pow(1.1, -yoffset)
        self.ortho_half_height = max(1e-4, self.ortho_half_height * factor)


# =========================
# Rational Bézier curve (projective control net)
# =========================

class RationalBezier:
    """
    ctrl4: (n+1,4) array of projective control points:
        [X,Y,Z,W] = [w*x, w*y, w*z, w]
    """

    def __init__(self, ctrl4: np.ndarray):
        c = np.asarray(ctrl4, dtype=np.float64)
        if c.ndim != 2 or c.shape[1] != 4:
            raise ValueError("ctrl4 must be (n+1,4)")
        self.ctrl4 = c

    def degree(self) -> int:
        return self.ctrl4.shape[0] - 1

    def de_casteljau_homo(self, u: float) -> np.ndarray:
        a = self.ctrl4.copy()
        n = a.shape[0] - 1
        for _ in range(n):
            a[:-1, :] = (1.0 - u) * a[:-1, :] + u * a[1:, :]
            a = a[:-1, :]
        return a[0]

    def eval_world(self, u: float) -> np.ndarray:
        Ch = self.de_casteljau_homo(u)
        if abs(Ch[3]) < 1e-30:
            return np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        return Ch[:3] / Ch[3]

    def polyline(self, samples: int = 64) -> np.ndarray:
        us = np.linspace(0.0, 1.0, samples, dtype=np.float64)
        pts = [self.eval_world(float(u)) for u in us]
        return np.array(pts, dtype=np.float32)


# =========================
# Snap engine (projective prefilter + pixel refinement)
# =========================

def pixel_planes_world(
    M_world_to_clip_rowmajor_T: np.ndarray, u_ndc: float, v_ndc: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build world-space 4D planes that vanish on the pixel line at (u_ndc,v_ndc).
    We pass M^T (row-major CPU composite) so we can just do n = M^T @ p.
    p_x: x' - u * w' = 0, p_y: y' - v * w' = 0
    """
    p_x = np.array([1.0, 0.0, 0.0, -u_ndc], dtype=np.float64)
    p_y = np.array([0.0, 1.0, 0.0, -v_ndc], dtype=np.float64)
    n_x = M_world_to_clip_rowmajor_T @ p_x
    n_y = M_world_to_clip_rowmajor_T @ p_y
    return n_x, n_y


class SnapHit(TypedDict):
    u: float
    world: tuple[float, float, float]
    pixel: tuple[float, float]
    ndc: tuple[float, float]
    dist_px: float
    ref: Optional[int | Any]


def snap_curve_to_cursor(
    curve: RationalBezier,
    M_cpu: np.ndarray,
    v: ViewportInfo,
    cursor_ndc_xy: np.ndarray,
    snap_px: float = 8.0,
) -> Optional[SnapHit]:
    """
    Returns dict with hit info or None:
      { 'u': float, 'world': (x,y,z), 'ndc': (x,y,z), 'pixel': (x,y) }
    Strategy:
      1) Projective prefilter: intervals for r_x(u) and r_y(u) must cross a small band.
      2) Coarse sample along u to find minimal pixel distance to cursor.
      3) 1D refinement (golden-section) in a small neighborhood.
    """
    u_ndc, v_ndc = float(cursor_ndc_xy[0]), float(cursor_ndc_xy[1])
    n_x, n_y = pixel_planes_world(M_cpu, u_ndc, v_ndc)
    rx_ctrl = curve.ctrl4 @ n_x
    ry_ctrl = curve.ctrl4 @ n_y

    band = 1e-9
    if (rx_ctrl.max() < -band) or (rx_ctrl.min() > band):
        return None
    if (ry_ctrl.max() < -band) or (ry_ctrl.min() > band):
        return None

    def world_to_ndc(p_world: np.ndarray) -> np.ndarray:
        P_eu = np.array([p_world[0], p_world[1], p_world[2], 1.0], dtype=np.float64)
        clip = M_cpu.T @ P_eu
        if abs(clip[3]) < 1e-30:
            return np.array([np.nan, np.nan, np.nan])
        return clip[:3] / clip[3]

    def px_distance_from_u(u: float) -> Tuple[float, np.ndarray, np.ndarray]:
        Pw = curve.eval_world(u)
        ndc = world_to_ndc(Pw)
        if np.any(np.isnan(ndc)):
            return float("inf"), Pw, ndc
        px = ndc_to_pixels(ndc[:2], v)
        cur_px = ndc_to_pixels(cursor_ndc_xy, v)
        dist = np.linalg.norm(px - cur_px)
        return float(dist), Pw, ndc

    samples = 64
    best = (float("inf"), 0.0, np.zeros(3), np.zeros(3))
    for i in range(samples + 1):
        u = i / samples
        d, Pw, ndc = px_distance_from_u(u)
        if d < best[0]:
            best = (d, u, Pw, ndc)

    if best[0] > snap_px:
        return None

    def refine(
        u0: float, h: float = 1.0 / samples, iters: int = 20
    ) -> Tuple[float, np.ndarray, np.ndarray, float]:
        a = max(0.0, u0 - 2 * h)
        b = min(1.0, u0 + 2 * h)
        gr = (math.sqrt(5.0) - 1.0) / 2.0
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc, Pw_c, ndc_c = px_distance_from_u(c)
        fd, Pw_d, ndc_d = px_distance_from_u(d)
        for _ in range(iters):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - gr * (b - a)
                fc, Pw_c, ndc_c = px_distance_from_u(c)
            else:
                a, c, fc = c, d, fd
                d = a + gr * (b - a)
                fd, Pw_d, ndc_d = px_distance_from_u(d)
        if fc < fd:
            return c, Pw_c, ndc_c, fc
        return d, Pw_d, ndc_d, fd

    u_ref, Pw_ref, ndc_ref, dist_ref = refine(best[1])
    if dist_ref > snap_px:
        return None

    pix_ref = ndc_to_pixels(ndc_ref[:2], v)
    return SnapHit(
        **{
            "u": float(u_ref),
            "world": tuple(Pw_ref.tolist()),
            "ndc": tuple(ndc_ref.tolist()),
            "pixel": tuple(pix_ref.tolist()),
            "dist_px": float(dist_ref),
            "ref": curve,
        }
    )


# =========================
# Surface helpers
# =========================

def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Area-weighted smooth vertex normals for a triangle mesh.
    faces may be flat or shaped (n,3).
    """
    verts = np.ascontiguousarray(vertices, dtype=np.float64)
    tris = np.ascontiguousarray(faces, dtype=np.int64).reshape(-1, 3)

    normals = np.zeros_like(verts, dtype=np.float64)
    if tris.size == 0 or verts.size == 0:
        return normals.astype(np.float32)

    p0 = verts[tris[:, 0]]
    p1 = verts[tris[:, 1]]
    p2 = verts[tris[:, 2]]
    face_normals = np.cross(p1 - p0, p2 - p0)

    np.add.at(normals, tris[:, 0], face_normals)
    np.add.at(normals, tris[:, 1], face_normals)
    np.add.at(normals, tris[:, 2], face_normals)

    lens = np.linalg.norm(normals, axis=1)
    mask = lens > 1e-20
    normals[mask] /= lens[mask, None]
    normals[~mask] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return normals.astype(np.float32)


# =========================
# GL program
# =========================

VERT_SRC = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;

uniform mat4 uProjection;
uniform mat4 uView;
uniform mat4 uModel;
uniform float uPointSize;

out vec3 vViewNormal;

void main(){
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    gl_Position = uProjection * uView * worldPos;
    gl_PointSize = uPointSize;

    // Plasticity-style shading is camera-locked: everything is a function of
    // the VIEW-space normal. uView is rigid (rotation + translation), so
    // mat3(uView) is a pure rotation and safe for normals.
    mat3 normalMat = transpose(inverse(mat3(uModel)));
    vViewNormal = mat3(uView) * (normalMat * aNormal);
}
"""

FRAG_SRC = """
#version 330 core
// uRenderKind: 0 = flat color (lines/regular points), 1 = snap sprite,
//              2 = Plasticity-style shaded surface (fitted two-light NPR rig)
uniform vec4 uColor;
uniform vec4 uBorderColor;
uniform int uRenderKind;

uniform float uPtSize;
uniform float uBorderPx;
uniform float uInnerSizePx;
uniform float uCrossOutPx;
uniform float uCrossThickPx;

// ---- Plasticity rig: fitted constants come in as uniforms (all view-space) ----
uniform vec3  uKeyDir;      // warm key direction, view space
uniform vec3  uKeyColor;    // warm key color
uniform vec3  uAmbColor;    // cool ambient
uniform float uAlbedo;
uniform float uWrap;        // wrapped-Lambert softness
uniform vec3  uGndDir;      // ground-bounce direction, view space
uniform vec3  uGndColor;    // warm bounce color
uniform float uGndI;
uniform vec3  uGlareColor;
uniform float uGlareI;
uniform float uGlareE;
uniform float uRimK;
uniform float uRimE;
// ---- user controls (1.0 = fitted reference look) ----
uniform float uHighlight;   // glare brightness
uniform float uShadow;      // fill level: lower = darker shadows
uniform float uKeyGain;     // warm lit-side brightness

in vec3 vViewNormal;

out vec4 FragColor;

void main(){
    if(uRenderKind == 1){
        vec2 uv_px = gl_PointCoord * uPtSize;
        float half_pt = 0.5 * uPtSize;
        vec2 d = abs(uv_px - vec2(half_pt));

        float inner_half = 0.5 * uInnerSizePx;
        float border = uBorderPx;
        float cross_out = uCrossOutPx;
        float cross_half_thick = 0.5 * uCrossThickPx;

        bool inside_square = (d.x <= inner_half) && (d.y <= inner_half);
        bool in_fill   = (d.x <= inner_half - border) && (d.y <= inner_half - border);
        bool in_border = inside_square && !in_fill;

        bool in_cross = ((d.x <= cross_half_thick) && (d.y <= inner_half + cross_out)) ||
                        ((d.y <= cross_half_thick) && (d.x <= inner_half + cross_out));

        if(!(inside_square || in_cross)) discard;

        vec4 col = uColor;
        if(in_border || in_cross) col = uBorderColor;
        FragColor = col;
        return;
    }

    if(uRenderKind == 2){
        vec3 n = normalize(vViewNormal);
        // Two-sided shading for thin CAD sheets. Orthographic camera:
        // the view vector in view space is exactly +z, so back-facing
        // is simply n.z < 0.
        if(n.z < 0.0) n = -n;

        // warm key, wrapped Lambert: warmth and brightness rise together;
        // shaded faces fall to the cool ambient
        vec3 L = normalize(uKeyDir);
        float D = clamp((dot(n, L) + uWrap) / (1.0 + uWrap), 0.0, 1.0);
        vec3 c = uAlbedo * (uShadow * uAmbColor + uKeyGain * uKeyColor * D);

        // warm ground bounce (fills the underside; scaled by the shadow knob)
        c += uAlbedo * uGndI * uShadow * uGndColor * max(dot(n, normalize(uGndDir)), 0.0);

        // warm glare: with ortho projection V = (0,0,1), so H is constant
        vec3 H = normalize(L + vec3(0.0, 0.0, 1.0));
        c += uGlareColor * uHighlight * uGlareI * pow(max(dot(n, H), 0.0), uGlareE);

        // rim darkening into the dark background
        c *= 1.0 - uRimK * pow(1.0 - max(n.z, 0.0), uRimE);

        // per-object tint: (1,1,1) reproduces the fitted look exactly
        c *= uColor.rgb;

        FragColor = vec4(clamp(c, 0.0, 1.0), uColor.a);
        return;
    }

    FragColor = uColor;
}
"""


def make_program() -> int:
    def compile_shader(src, typ):
        s = glCreateShader(typ)
        glShaderSource(s, src)
        glCompileShader(s)
        if glGetShaderiv(s, GL_COMPILE_STATUS) != GL_TRUE:
            raise RuntimeError(glGetShaderInfoLog(s).decode())
        return s

    vs = compile_shader(VERT_SRC, GL_VERTEX_SHADER)
    fs = compile_shader(FRAG_SRC, GL_FRAGMENT_SHADER)
    prog = glCreateProgram()
    glAttachShader(prog, vs)
    glAttachShader(prog, fs)
    glLinkProgram(prog)
    if glGetProgramiv(prog, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(prog).decode())
    glDeleteShader(vs)
    glDeleteShader(fs)
    return prog


LINE_VERT_SRC = """
#version 330 core
// Screen-space thick lines: each segment is a quad; every vertex knows its own
// endpoint, the opposite endpoint, which end it is, and which side to offset.
layout(location=0) in vec3 aPos;      // this endpoint
layout(location=1) in vec3 aOther;    // opposite endpoint of the segment
layout(location=2) in vec2 aSideEnd;  // x: side (-1/+1), y: end (0/1)

uniform mat4 uMVP;
uniform vec2 uViewportPx;   // framebuffer size in pixels
uniform float uWidthPx;     // total line width in pixels

void main(){
    vec4 cs = uMVP * vec4(aPos, 1.0);
    vec4 co = uMVP * vec4(aOther, 1.0);
    vec2 ss = cs.xy / cs.w * 0.5 * uViewportPx;
    vec2 so = co.xy / co.w * 0.5 * uViewportPx;
    // consistent along-segment direction regardless of which end we are
    vec2 d = (aSideEnd.y < 0.5) ? (so - ss) : (ss - so);
    float len = max(length(d), 1e-6);
    vec2 n = vec2(-d.y, d.x) / len;
    vec2 off_px = n * aSideEnd.x * 0.5 * uWidthPx;
    cs.xy += off_px / (0.5 * uViewportPx) * cs.w;
    gl_Position = cs;
}
"""

LINE_FRAG_SRC = """
#version 330 core
uniform vec4 uLineColor;
out vec4 FragColor;
void main(){ FragColor = uLineColor; }
"""


def make_line_program() -> int:
    def compile_shader(src, typ):
        s = glCreateShader(typ)
        glShaderSource(s, src)
        glCompileShader(s)
        if glGetShaderiv(s, GL_COMPILE_STATUS) != GL_TRUE:
            raise RuntimeError(glGetShaderInfoLog(s).decode())
        return s

    vs = compile_shader(LINE_VERT_SRC, GL_VERTEX_SHADER)
    fs = compile_shader(LINE_FRAG_SRC, GL_FRAGMENT_SHADER)
    prog = glCreateProgram()
    glAttachShader(prog, vs)
    glAttachShader(prog, fs)
    glLinkProgram(prog)
    if glGetProgramiv(prog, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(prog).decode())
    glDeleteShader(vs)
    glDeleteShader(fs)
    return prog


def build_thick_line_geometry(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Polyline (N,3) -> interleaved quad vertices (4*(N-1), 8) and indices.

    Vertex layout: [self.xyz, other.xyz, side, end]. Butt joints: adequate for
    CAD linework at 2-4 px; switch to miters if sharp corners ever show notches.
    """
    pts = np.ascontiguousarray(pts, dtype=np.float32)
    p0, p1 = pts[:-1], pts[1:]
    nseg = len(p0)
    verts = np.empty((nseg, 4, 8), dtype=np.float32)
    for k, (a, b, side, end) in enumerate(
        [(p0, p1, -1.0, 0.0), (p0, p1, 1.0, 0.0), (p1, p0, -1.0, 1.0), (p1, p0, 1.0, 1.0)]
    ):
        verts[:, k, 0:3] = a
        verts[:, k, 3:6] = b
        verts[:, k, 6] = side
        verts[:, k, 7] = end
    idx = (
        np.arange(nseg, dtype=np.uint32)[:, None] * 4
        + np.array([0, 1, 2, 2, 1, 3], dtype=np.uint32)[None, :]
    ).ravel()
    return verts.reshape(-1, 8), idx


# =========================
# Viewer settings
# =========================

@dataclass
class SceneInfo:
    bbox: AABB = field(default_factory=lambda: AABB(np.zeros(3), np.zeros(3)))


@dataclass
class SnapSettings:
    snap_px: float = 30.0
    size_px: float = 12.0
    border_px: float = 2.0
    cross_out_px: float = 6.0
    cross_thick_px: float = 2.0
    color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    border_color: tuple[float, float, float, float] | Literal["by_object"] = "by_object"


@dataclass
class PlasticityShadingSettings:
    """Fitted two-light NPR rig (Plasticity-style viewport shading).

    All directions are in VIEW space — the rig is locked to the camera by
    construction, which is what keeps the shading stable while orbiting
    (this replaces the old world-light + headlight_mix machinery).
    Constants were least-squares fitted to pixel samples from a Plasticity
    screenshot; palette is display-space, so no output gamma is applied.
    """

    key_dir: tuple[float, float, float] = (-0.436, 0.900, 0.031)
    key_color: tuple[float, float, float] = (0.812, 0.710, 0.526)   # warm key
    amb_color: tuple[float, float, float] = (0.100, 0.136, 0.214)   # cool ambient
    albedo: float = 0.793
    wrap: float = 0.5
    gnd_dir: tuple[float, float, float] = (0.15, -0.92, 0.36)
    gnd_color: tuple[float, float, float] = (0.85, 0.68, 0.48)      # warm bounce
    gnd_intensity: float = 0.500
    glare_color: tuple[float, float, float] = (1.0, 0.90, 0.72)     # warm glare
    glare_intensity: float = 0.040
    glare_power: float = 21.1
    rim_strength: float = 0.889
    rim_power: float = 1.676

    # User controls, 1.0 = fitted reference look:
    highlight: float = 1.0   # glare brightness (0..3)
    shadow: float = 1.0      # fill level; lower = darker shadows (0.4..1.6)
    key_gain: float = 1.0    # warm lit-side brightness (0.8..1.3)


@dataclass
class EdgeSettings:
    """Plasticity-style linework. Background is ~0.09, so both edge colors sit
    BELOW it: dark lines stay readable on lit faces AND against the dark field.
    Widths are in window points; multiplied by the content scale at draw time,
    so 2.0 means 2 device-independent pixels on a retina display too."""

    inner_width: float = 1.   # isocurves and other interior linework
    outline_width: float = 1.5  # surface boundaries / B-rep edges
    inner_color: tuple[float, float, float, float] = (0.055, 0.058, 0.070, 1.0)
    outline_color: tuple[float, float, float, float] = (0.030, 0.032, 0.040, 1.0)


@dataclass
class ViewerSettings:
    snap: SnapSettings = field(default_factory=SnapSettings)
    shading: PlasticityShadingSettings = field(default_factory=PlasticityShadingSettings)
    edges: EdgeSettings = field(default_factory=EdgeSettings)


# =========================
# Viewer
# =========================

class Viewer:
    cam: OrbitCamera
    scene_info: SceneInfo

    def add(self, obj, *args, **kwargs):
        from mmcore.topo.brep import BRep

        if isinstance(obj, BRep):
            return self.add_brep(obj, *args, **kwargs)
        if isinstance(obj, RationalBezier):
            return self._add_curve(obj, *args, **kwargs)
        if isinstance(obj, NURBSCurveTuple):
            return self._add_nurbs_curve(obj, *args, **kwargs)
        if isinstance(obj, NURBSSurfaceTuple):
            return self.add_nurbs_surface(obj, *args, **kwargs)
        if isinstance(obj, np.ndarray):
            if len(obj.shape) == 2 and "rational" in kwargs:
                return self.add_bern_curve(obj, *args, **kwargs)
            if len(obj.shape) == 1 and obj.shape[0] == 3:
                return self.add_point3d(obj, *args, **kwargs)
            if len(obj.shape) == 1 and obj.shape[0] == 2:
                return self.add_point2d(obj, *args, **kwargs)
            raise ValueError(f"Unsupported shape {obj.shape}")
        if isinstance(obj, (tuple, list)):
            if len(obj) == 2:
                return self.add_point2d(obj, *args, **kwargs)
            if len(obj) == 3:
                return self.add_point3d(obj, *args, **kwargs)
            raise ValueError(f"Unknown type: {obj}")
        raise ValueError(f"Unsupported object type: {type(obj)!r}")

    def __init__(self, width=1200, height=800, camera=None, settings: ViewerSettings | None = None):
        if not glfw.init():
            raise RuntimeError("GLFW init failed")
        if settings is None:
            settings = ViewerSettings()
        self.settings = settings

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.COCOA_RETINA_FRAMEBUFFER, glfw.TRUE)
        # Thick screen-space line quads alias badly without multisampling.
        glfw.window_hint(glfw.SAMPLES, 8)

        self.window = glfw.create_window(width, height, "Snap Viewer", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("GLFW window creation failed")
        glfw.make_context_current(self.window)

        self.color_table: dict[int, tuple[float, float, float, float]] = {}
        self.program = make_program()
        self.line_program = make_line_program()
        self.loc_line_uMVP = glGetUniformLocation(self.line_program, "uMVP")
        self.loc_line_uViewportPx = glGetUniformLocation(self.line_program, "uViewportPx")
        self.loc_line_uWidthPx = glGetUniformLocation(self.line_program, "uWidthPx")
        self.loc_line_uColor = glGetUniformLocation(self.line_program, "uLineColor")

        self.loc_uP = glGetUniformLocation(self.program, "uProjection")
        self.loc_uV = glGetUniformLocation(self.program, "uView")
        self.loc_uM = glGetUniformLocation(self.program, "uModel")
        self.loc_uColor = glGetUniformLocation(self.program, "uColor")
        self.loc_uBorderColor = glGetUniformLocation(self.program, "uBorderColor")
        self.loc_uRenderKind = glGetUniformLocation(self.program, "uRenderKind")
        self.loc_uPtSize = glGetUniformLocation(self.program, "uPtSize")
        self.loc_uBorderPx = glGetUniformLocation(self.program, "uBorderPx")
        self.loc_uPointSize = glGetUniformLocation(self.program, "uPointSize")
        self.loc_uInnerSizePx = glGetUniformLocation(self.program, "uInnerSizePx")
        self.loc_uCrossOutPx = glGetUniformLocation(self.program, "uCrossOutPx")
        self.loc_uCrossThickPx = glGetUniformLocation(self.program, "uCrossThickPx")

        # Plasticity rig uniforms
        self.loc_uKeyDir = glGetUniformLocation(self.program, "uKeyDir")
        self.loc_uKeyColor = glGetUniformLocation(self.program, "uKeyColor")
        self.loc_uAmbColor = glGetUniformLocation(self.program, "uAmbColor")
        self.loc_uAlbedo = glGetUniformLocation(self.program, "uAlbedo")
        self.loc_uWrap = glGetUniformLocation(self.program, "uWrap")
        self.loc_uGndDir = glGetUniformLocation(self.program, "uGndDir")
        self.loc_uGndColor = glGetUniformLocation(self.program, "uGndColor")
        self.loc_uGndI = glGetUniformLocation(self.program, "uGndI")
        self.loc_uGlareColor = glGetUniformLocation(self.program, "uGlareColor")
        self.loc_uGlareI = glGetUniformLocation(self.program, "uGlareI")
        self.loc_uGlareE = glGetUniformLocation(self.program, "uGlareE")
        self.loc_uRimK = glGetUniformLocation(self.program, "uRimK")
        self.loc_uRimE = glGetUniformLocation(self.program, "uRimE")
        self.loc_uHighlight = glGetUniformLocation(self.program, "uHighlight")
        self.loc_uShadow = glGetUniformLocation(self.program, "uShadow")
        self.loc_uKeyGain = glGetUniformLocation(self.program, "uKeyGain")

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_PROGRAM_POINT_SIZE)
        self._rebuild_viewport()

        self.cam = OrbitCamera() if camera is None else camera
        self.dragging_orbit = False
        self.dragging_pan = False
        self.last_cursor_px = np.array([0.0, 0.0], dtype=np.float64)

        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_mouse_button_callback(self.window, self._on_mouse)
        glfw.set_scroll_callback(self.window, self._on_scroll)
        glfw.set_framebuffer_size_callback(self.window, self._on_resize)

        self.model = np.eye(4, dtype=np.float32)
        self.curves: List[RationalBezier] = []

        self.snap_hit: Optional[SnapHit] = None
        self.points: list[tuple[np.ndarray, tuple[float, float, float, float], float]] = []

        # Each line: (vao, vbo, ebo, index_count, color_rgba, width_pt)
        self.lines: list[tuple[int, int, int, int, np.ndarray, float]] = []
        # Each mesh: (vao, vbo, nbo, ebo, index_count, color_rgba, centroid_world)
        self.meshes: list[tuple[int, int, int, int, int, np.ndarray, np.ndarray]] = []

        self.scene_info = SceneInfo()

    @property
    def snap_px(self):
        return self.settings.snap.snap_px

    # ---------- GLFW callbacks ----------

    def _rebuild_viewport(self):
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        glViewport(0, 0, fb_w, fb_h)

    def _on_resize(self, window, w, h):
        self._rebuild_viewport()

    def _on_scroll(self, window, xoff, yoff):
        self.cam.zoom_wheel(yoff)

    def _on_mouse(self, window, button, action, mods):
        v = read_viewport_info(self.window)
        xpt, ypt = glfw.get_cursor_pos(self.window)
        self.last_cursor_px = points_to_pixels((xpt, ypt), v)
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.dragging_orbit = (action == glfw.PRESS) and not (mods & glfw.MOD_SHIFT)
            self.dragging_pan = (action == glfw.PRESS) and (mods & glfw.MOD_SHIFT)
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self.dragging_pan = (action == glfw.PRESS)

    def _on_cursor(self, window, xpt, ypt):
        v = read_viewport_info(self.window)
        cur_px = points_to_pixels((xpt, ypt), v)
        dp = cur_px - self.last_cursor_px
        if self.dragging_orbit:
            self.cam.orbit(dp[0], dp[1], v)
        elif self.dragging_pan:
            self.cam.pan(dp[0], -dp[1], v)
        self.last_cursor_px = cur_px

    # ---------- Scene build ----------

    def _add_curve(
        self,
        curve: RationalBezier,
        color=None,
        samples=128,
        width_px: float | None = None,
        role: Literal["inner", "outline"] = "outline",
    ):
        """role picks the EdgeSettings defaults when color/width_px are None."""
        e = self.settings.edges
        if color is None:
            color = e.inner_color if role == "inner" else e.outline_color
        if width_px is None:
            width_px = e.inner_width if role == "inner" else e.outline_width

        self.curves.append(curve)
        self.color_table[id(curve)] = color

        idx = len(self.curves) - 1
        pts = curve.polyline(samples=samples).astype(np.float32)
        verts, indices = build_thick_line_geometry(pts)

        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)

        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        stride = 8 * 4
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))

        ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        self.scene_info.bbox.merge(AABB.from_points(pts))
        self.lines.append((vao, vbo, ebo, indices.size, np.array(color, dtype=np.float32), float(width_px)))
        return idx

    def _mesh_upload(self, vertices: np.ndarray, faces: np.ndarray, color) -> Optional[int]:
        vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        faces = np.ascontiguousarray(faces, dtype=np.uint32)
        if vertices.size == 0 or faces.size == 0:
            return None

        normals = compute_vertex_normals(vertices, faces)
        centroid = vertices.mean(axis=0).astype(np.float32)

        if len(color) == 3:
            # Plasticity style reads best opaque; edges/isocurves carry structure.
            color = (*color, 1.0)
        color_arr = np.array(color, dtype=np.float32)

        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)

        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

        nbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, nbo)
        glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)

        ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.nbytes, faces, GL_STATIC_DRAW)

        self.scene_info.bbox.merge(AABB.from_points(vertices))
        self.meshes.append((vao, vbo, nbo, ebo, faces.size, color_arr, centroid))
        return len(self.meshes) - 1

    def _add_surface_mesh(self, surface, color=(1.0, 1.0, 1.0, 1.0), tol=0.05):
        surf = _tuple_to_nurbs(surface) if isinstance(surface, NURBSSurfaceTuple) else surface
        mesh = surface_to_mesh(surf, tol=tol)
        return self._mesh_upload(mesh["position"], mesh["faces"], color=color)

    def add_bern_curve(self, arr, *args, rational: bool = False, **kwargs):
        if not rational:
            A = np.zeros((arr.shape[0], 4))
            for i in range(arr.shape[1]):
                A[..., i] = arr[..., i]
            A[..., -1] = 1.0
            curve = RationalBezier(A)
        else:
            if arr.shape[1] != 4:
                A = np.zeros((arr.shape[0], 4))
                for i in range(arr.shape[1] - 1):
                    A[..., i] = arr[..., i]
                A[..., -1] = arr[..., -1]
            else:
                A = arr
            curve = RationalBezier(A)
        return self._add_curve(curve, *args, **kwargs)

    def add_point3d(self, arr, color=(0.8, 0.8, 0.8, 1.0), size_px=9):
        self.points.append((np.array(arr, dtype=np.float32), color, size_px))
        return len(self.points) - 1

    def add_point2d(self, arr, color=(0.8, 0.8, 0.8, 1.0), size_px=9):
        self.points.append((np.array((*arr, 0.0), dtype=np.float32), color, size_px))
        return len(self.points) - 1

    def _build_demo_scene(self):
        P = np.array(
            [
                [-4.0, -2.0, 0.0, 1.0],
                [-1.0, 3.0, 0.0, 0.7],
                [2.0, -3.0, 0.0, 1.2],
                [4.0, 2.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        ctrl4 = np.column_stack((P[:, :3] * P[:, 3:4], P[:, 3:4]))
        self._add_curve(RationalBezier(ctrl4), color=(1.0, 1.0, 1.0, 1.0))

        Q = np.array(
            [
                [-3.0, 3.5, 1.0, 1.0],
                [-1.0, -1.0, 2.0, 0.8],
                [1.0, 1.0, -2.0, 1.3],
                [3.0, -2.5, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        ctrl4b = np.column_stack((Q[:, :3] * Q[:, 3:4], Q[:, 3:4]))
        self._add_curve(RationalBezier(ctrl4b), color=(0.7, 0.9, 1.0, 1.0))

    # ---------- Render & snap ----------

    def _add_nurbs_curve(self, curve: NURBSCurveTuple, color=None, *args, **kwargs):
        beziers = decompose_curve(curve)
        return tuple(
            self.add(
                to_homogeneous_1d(bezier.control_points, bezier.weights),
                rational=True,
                color=color,
                *args,
                **kwargs,
            )
            for bezier in beziers
        )

    def add_nurbs_curve(self, curve: NURBSCurveTuple, color=None, *args, **kwargs):
        return self._add_nurbs_curve(curve, color, *args, **kwargs)

    def add_nurbs_surface(
        self,
        surface: NURBSSurfaceTuple,
        color=None,
        surface_color=(1.0, 1.0, 1.0, 1.0),
        u_count=1,
        v_count=1,
        show_edges: bool = True,
        show_isocurves: bool = True,
        *args,
        **kwargs,
    ):
        shade = kwargs.pop("shade", True)
        surface_tol = kwargs.pop("surface_tol", 0.01)

        meshes = []
        if shade:
            meshes.append(self._add_surface_mesh(surface, color=surface_color, tol=surface_tol))

        (u0, u1), (v0, v1) = surface.interval()
        us = np.linspace(u0, u1, u_count + 2)[1:-1]
        vs = np.linspace(v0, v1, v_count + 2)[1:-1]

        iso_color = None if color is None else (color[0] * 0.5, color[1] * 0.5, color[2] * 0.5, color[3])

        isolines = []
        if show_isocurves:
            for crv in [extract_isocurve(surface, u, "u") for u in us] + [
                extract_isocurve(surface, v, "v") for v in vs
            ]:
                isolines.append(self._add_nurbs_curve(crv, iso_color, role="inner", *args, **kwargs))

        bnds = []
        if show_edges:
            for bnd in extract_surface_boundaries(surface):
                bnds.append(self._add_nurbs_curve(bnd, color, role="outline", *args, **kwargs))

        return tuple(meshes) + tuple(bnds) + tuple(isolines)

    def add_brep(
        self,
        brep,
        edge_color=None,
        surface_color=(1.0, 1.0, 1.0, 1.0),
        tol=0.05,
        show_edges=True,
        shade=True,
    ):
        """Add a BRep to the viewer."""
        from mmcore.geom._nurbs_knots import trim_curve

        results = []

        if shade:
            for f_id, face in brep.F.items():
                if face.surf is None:
                    continue
                for lid in [face.outer] + face.inners:
                    for he_id in brep._loop_halfedges(lid):
                        he = brep.HE[he_id]
                        edge = brep.E[he.edge]
                        if edge.geom is not None and he.pcurve is None:
                            brep.compute_pcurve(he_id, tol=tol)

                try:
                    mesh = tessellate_brep_face(brep, f_id, tol=tol)
                    if mesh["position"].size > 0 and mesh["faces"].size > 0:
                        idx = self._add_mesh(mesh, color=surface_color)
                        results.append(("face", f_id, idx))
                except Exception as exc:
                    print(f"Warning: tessellation of face {f_id} failed: {exc}", file=sys.stderr)

        if show_edges:
            for e_id, edge in brep.E.items():
                if edge.geom is None:
                    continue
                crv = brep.G_CRV[edge.geom]
                t0, t1 = sorted(edge.param)
                try:
                    trimmed = trim_curve(crv, t0, t1)
                    self._add_nurbs_curve(trimmed, color=edge_color, role="outline")
                except Exception:
                    self._add_nurbs_curve(crv, color=edge_color, role="outline")

        return results

    def _add_mesh(self, mesh, color=(1.0, 1.0, 1.0, 1.0)):
        """Upload a Mesh dict (position, faces) to GL. Returns mesh index."""
        return self._mesh_upload(mesh["position"], mesh["faces"], color=color)

    def _upload_frame_uniforms(self, P_row: np.ndarray, V_row: np.ndarray, M_row: np.ndarray):
        """
        We build matrices row-major and ask GL to transpose them on upload.
        The shader then sees column-major matrices consistent with GLSL's math.
        CPU composite to match GLSL is: M_cpu = (P_row @ V_row @ M_row).T
        """
        glUseProgram(self.program)
        glUniformMatrix4fv(self.loc_uP, 1, GL_TRUE, P_row)
        glUniformMatrix4fv(self.loc_uV, 1, GL_TRUE, V_row)
        glUniformMatrix4fv(self.loc_uM, 1, GL_TRUE, M_row)

        s = self.settings.shading
        glUniform3fv(self.loc_uKeyDir, 1, normalize(np.array(s.key_dir, dtype=np.float32)))
        glUniform3fv(self.loc_uKeyColor, 1, np.array(s.key_color, dtype=np.float32))
        glUniform3fv(self.loc_uAmbColor, 1, np.array(s.amb_color, dtype=np.float32))
        glUniform1f(self.loc_uAlbedo, float(s.albedo))
        glUniform1f(self.loc_uWrap, float(s.wrap))
        glUniform3fv(self.loc_uGndDir, 1, normalize(np.array(s.gnd_dir, dtype=np.float32)))
        glUniform3fv(self.loc_uGndColor, 1, np.array(s.gnd_color, dtype=np.float32))
        glUniform1f(self.loc_uGndI, float(s.gnd_intensity))
        glUniform3fv(self.loc_uGlareColor, 1, np.array(s.glare_color, dtype=np.float32))
        glUniform1f(self.loc_uGlareI, float(s.glare_intensity))
        glUniform1f(self.loc_uGlareE, float(s.glare_power))
        glUniform1f(self.loc_uRimK, float(s.rim_strength))
        glUniform1f(self.loc_uRimE, float(s.rim_power))
        glUniform1f(self.loc_uHighlight, float(s.highlight))
        glUniform1f(self.loc_uShadow, float(s.shadow))
        glUniform1f(self.loc_uKeyGain, float(s.key_gain))

    def _draw_lines(self, P_row: np.ndarray, V_row: np.ndarray, M_row: np.ndarray, vinfo: ViewportInfo):
        if not self.lines:
            return
        glUseProgram(self.line_program)
        MVP = (P_row @ V_row @ M_row).astype(np.float32)
        glUniformMatrix4fv(self.loc_line_uMVP, 1, GL_TRUE, MVP)
        glUniform2f(self.loc_line_uViewportPx, float(vinfo.fb_w), float(vinfo.fb_h))
        # widths are authored in window points -> device pixels via content scale
        scale = 0.5 * (vinfo.sx + vinfo.sy)

        for vao, vbo, ebo, index_count, color, width_pt in self.lines:
            glBindVertexArray(vao)
            glUniform4fv(self.loc_line_uColor, 1, color)
            glUniform1f(self.loc_line_uWidthPx, width_pt * scale)
            glDrawElements(GL_TRIANGLES, index_count, GL_UNSIGNED_INT, None)

    def _draw_mesh_record(self, record):
        vao, vbo, nbo, ebo, index_count, color, centroid = record
        glBindVertexArray(vao)
        glUniform4fv(self.loc_uColor, 1, color)
        glDrawElements(GL_TRIANGLES, index_count, GL_UNSIGNED_INT, None)

    def _mesh_depth_key(self, record, V_row: np.ndarray) -> float:
        centroid = record[6]
        cam = V_row @ np.array([centroid[0], centroid[1], centroid[2], 1.0], dtype=np.float32)
        return float(cam[2])

    def _draw_surfaces(self, V_row: np.ndarray):
        if not self.meshes:
            return

        glUseProgram(self.program)
        glUniform1i(self.loc_uRenderKind, 2)
        glUniform1f(self.loc_uPtSize, 1.0)
        glUniform1f(self.loc_uPointSize, 1.0)
        glUniform1f(self.loc_uBorderPx, 0.0)

        opaque = [m for m in self.meshes if m[5][3] >= 0.999]
        transparent = [m for m in self.meshes if m[5][3] < 0.999]

        # Push the fills slightly back in depth so technical linework stays crisp
        # when it is drawn after the shaded pass.
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(1.0, 1.0)

        glDepthMask(GL_TRUE)
        for rec in opaque:
            self._draw_mesh_record(rec)

        if transparent:
            # Back-to-front centroid sort is a simple but effective improvement
            # for the semi-transparent technical fills used by this viewer.
            transparent.sort(key=lambda rec: self._mesh_depth_key(rec, V_row))
            glDepthMask(GL_FALSE)
            for rec in transparent:
                self._draw_mesh_record(rec)
            glDepthMask(GL_TRUE)

        glDisable(GL_POLYGON_OFFSET_FILL)

    def _draw_point(
        self,
        pos_world: np.ndarray,
        size_px=7.0,
        color=(1.0, 1.0, 0.0, 1.0),
        *,
        border_color=None,
        border_px: float = 0.0,
        render_kind: int = 0,
        inner_size_px: float = 0.0,
        cross_out_px: float = 0.0,
        cross_thick_px: float = 1.0,
    ):
        """
        When render_kind == 1 we render a custom sprite that uses the extra parameters.
        size_px is the total sprite size (gl_PointSize).
        inner_size_px is the square fill size; cross_out_px extends the crosshair past that square.
        """
        if border_color is None:
            border_color = color

        pts = np.array(pos_world, dtype=np.float32).reshape(1, 3)
        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)

        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, pts.nbytes, pts, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

        glUseProgram(self.program)
        glPointSize(max(1.0, float(size_px)))
        glUniform4fv(self.loc_uColor, 1, np.array(color, dtype=np.float32))
        glUniform4fv(self.loc_uBorderColor, 1, np.array(border_color, dtype=np.float32))
        glUniform1i(self.loc_uRenderKind, render_kind)
        glUniform1f(self.loc_uPtSize, float(size_px))
        glUniform1f(self.loc_uPointSize, float(size_px))
        glUniform1f(self.loc_uBorderPx, float(border_px))
        glUniform1f(self.loc_uInnerSizePx, float(inner_size_px))
        glUniform1f(self.loc_uCrossOutPx, float(cross_out_px))
        glUniform1f(self.loc_uCrossThickPx, float(cross_thick_px))
        glDrawArrays(GL_POINTS, 0, 1)

        glDeleteBuffers(1, [vbo])
        glDeleteVertexArrays(1, [vao])

    def _draw_snap_point(self, pos_world: np.ndarray):
        snap = self.settings.snap
        sprite_size = snap.size_px + 2 * max(snap.border_px, snap.cross_out_px)

        if self.snap_hit is not None and snap.border_color == "by_object" and self.snap_hit["ref"] is not None:
            border_color = self.color_table[id(self.snap_hit["ref"])]
        else:
            border_color = snap.border_color

        self._draw_point(
            pos_world,
            sprite_size,
            snap.color,
            border_color=border_color,
            border_px=snap.border_px,
            inner_size_px=snap.size_px,
            cross_out_px=snap.cross_out_px,
            cross_thick_px=snap.cross_thick_px,
            render_kind=1,
        )

    def _compute_snap(self, vinfo: ViewportInfo, M_cpu: np.ndarray):
        cursor_pt = glfw.get_cursor_pos(self.window)
        cursor_ndc = glfw_cursor_to_ndc(cursor_pt, vinfo)

        best = None
        for cv in self.curves:
            hit = snap_curve_to_cursor(cv, M_cpu, vinfo, cursor_ndc, snap_px=self.settings.snap.snap_px)
            if hit is None:
                continue
            if best is None or hit["dist_px"] < best["dist_px"]:
                best = hit
        self.snap_hit = best

    def run(self):
        while not glfw.window_should_close(self.window):
            glfw.poll_events()

            vinfo = read_viewport_info(self.window)
            aspect = vinfo.vw / max(1, vinfo.vh)
            P_row = self.cam.projection_matrix(aspect)
            V_row = self.cam.view_matrix()
            M_row = self.model

            M_cpu = (P_row @ V_row @ M_row).T

            self._upload_frame_uniforms(P_row, V_row, M_row)
            self._compute_snap(vinfo, M_cpu)

            # Background from the fitted reference (Plasticity uses ~0.09).
            glClearColor(0.09, 0.09, 0.10, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            self._draw_surfaces(V_row)
            self._draw_lines(P_row, V_row, M_row, vinfo)

            for point_arr, color, size_px in self.points:
                self._draw_point(point_arr, size_px=size_px, color=color)

            if self.snap_hit is not None:
                self._draw_snap_point(np.array(self.snap_hit["world"], dtype=np.float32))

            glfw.swap_buffers(self.window)

        for vao, vbo, ebo, _, _, _ in self.lines:
            glDeleteBuffers(1, [vbo])
            glDeleteBuffers(1, [ebo])
            glDeleteVertexArrays(1, [vao])

        for vao, vbo, nbo, ebo, _, _, _ in self.meshes:
            glDeleteBuffers(1, [vbo])
            glDeleteBuffers(1, [nbo])
            glDeleteBuffers(1, [ebo])
            glDeleteVertexArrays(1, [vao])

        glfw.terminate()


if __name__ == "__main__":
    Viewer().run()