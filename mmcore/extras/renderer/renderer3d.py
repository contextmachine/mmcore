from __future__ import annotations
import math
import time
import sys
from dataclasses import dataclass

from typing import Optional, Tuple, List, NamedTuple, Literal

import numpy as np
import glfw
from OpenGL.GL import *

from mmcore.geom._nurbs_eval import NURBSCurveTuple,to_homogeneous_1d,from_homogeneous_1d,to_homogeneous_2d
from mmcore.geom._nurbs_knots import decompose_curve
from mmcore.numeric.approx import adaptive_curve_sampler


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


def normalize(v: np.ndarray, eps=1e-12) -> np.ndarray:
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
    # Rows:
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
    vx: int; vy: int; vw: int; vh: int
    fb_w: int; fb_h: int
    sx: float; sy: float
    win_w: int; win_h: int


def read_viewport_info(window) -> ViewportInfo:
    vx, vy, vw, vh = glGetIntegerv(GL_VIEWPORT)
    fb_w, fb_h = glfw.get_framebuffer_size(window)
    sx, sy = glfw.get_window_content_scale(window)
    win_w, win_h = glfw.get_window_size(window)
    return ViewportInfo(int(vx), int(vy), int(vw), int(vh),
                        int(fb_w), int(fb_h),
                        float(sx), float(sy),
                        int(win_w), int(win_h))


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
    def __init__(self, target=(0.0, 0.0, 0.0), up=(0.,0.,1.),distance = 10.0,ortho_half_height = 5.0,yaw= math.radians(35.0),pitch= math.radians(30.0)):
        self.target = np.array(target, dtype=np.float32)
        self.distance = distance
        self.yaw = yaw
        self.pitch = pitch
        self.up_world = np.array(up, dtype=np.float32)  # Z-up CAD
        # Ortho zoom (world units half-extent vertically)
        self.ortho_half_height = ortho_half_height
        self.near = 0.1
        self.far = 10000.0
        self._lock_orbit = False
    def lock_orbit(self,lock:bool):
        self._lock_orbit = lock

    def eye(self) -> np.ndarray:
        # Spherical around target with Z-up
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        # right = (1,0,0) in world; forward roughly towards -Y at yaw=0
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
        # Sensitivity in radians per pixel
        s = 2.0 * math.pi / max(v.vw, v.vh)
        self.yaw -= dx_pixels * s
        self.pitch -= dy_pixels * s
        self.pitch = max(-math.radians(89.0), min(math.radians(89.0), self.pitch))

    def pan(self, dx_pixels: float, dy_pixels: float, v: ViewportInfo):
        # Convert pixels to world units based on current ortho scale
        aspect = v.vw / max(1, v.vh)
        h = self.ortho_half_height
        w = h * aspect
        # pixels -> NDC -> world delta
        dx_ndc = (dx_pixels / max(1, v.vw)) * 2.0
        dy_ndc = (dy_pixels / max(1, v.vh)) * 2.0
        delta_world = np.array([dx_ndc * w, dy_ndc * h, 0.0], dtype=np.float32)

        # Pan in camera's screen basis: right & up from view matrix
        V = self.view_matrix()
        right = V[0, 0:3]  # because we use row-major and upload with transpose=True
        up = V[1, 0:3]
        move = right * (-delta_world[0]) + up * (delta_world[1])
        self.target += move

    def zoom_wheel(self, yoffset: float):
        # Scale the ortho window
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

def pixel_planes_world(M_world_to_clip_rowmajor_T: np.ndarray, u_ndc: float, v_ndc: float) -> Tuple[np.ndarray, np.ndarray]:
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


def bern_eval_scalar(ctrl: np.ndarray, u: float) -> float:
    """Scalar Bézier in Bernstein basis (De Casteljau). ctrl: (n+1,)"""
    a = ctrl.astype(np.float64).copy()
    n = a.shape[0] - 1
    for _ in range(n):
        a[:-1] = (1.0 - u) * a[:-1] + u * a[1:]
        a = a[:-1]
    return float(a[0])


def snap_curve_to_cursor(curve: RationalBezier,
                         M_cpu: np.ndarray,  # (P@V@M).T (row-major)
                         v: ViewportInfo,
                         cursor_ndc_xy: np.ndarray,
                         snap_px: float = 8.0) -> Optional[dict]:
    """
    Returns dict with hit info or None:
      { 'u': float, 'world': (x,y,z), 'ndc': (x,y,z), 'pixel': (x,y) }
    Strategy:
      1) Projective prefilter: intervals for r_x(u) and r_y(u) must cross a small band.
      2) Coarse sample along u to find minimal pixel distance to cursor.
      3) 1D refinement (golden-section) in a small neighborhood.
    """
    # 1) Prefilter
    u_ndc, v_ndc = float(cursor_ndc_xy[0]), float(cursor_ndc_xy[1])
    n_x, n_y = pixel_planes_world(M_cpu, u_ndc, v_ndc)
    rx_ctrl = curve.ctrl4 @ n_x  # scalar Bernstein coefficients
    ry_ctrl = curve.ctrl4 @ n_y

    # Band: allow a bit of slack before we do the more expensive sampling
    band = 1e-9
    if (rx_ctrl.max() < -band) or (rx_ctrl.min() > band):
        return None
    if (ry_ctrl.max() < -band) or (ry_ctrl.min() > band):
        return None

    # Helper: pixel distance from a world point to cursor
    def world_to_ndc(p_world: np.ndarray) -> np.ndarray:
        P_eu = np.array([p_world[0], p_world[1], p_world[2], 1.0], dtype=np.float64)
        clip = M_cpu.T @ P_eu  # because M_cpu is (P@V@M).T
        if abs(clip[3]) < 1e-30:
            return np.array([np.nan, np.nan, np.nan])
        return clip[:3] / clip[3]

    def px_distance_from_u(u: float) -> Tuple[float, np.ndarray, np.ndarray]:
        Pw = curve.eval_world(u)
        ndc = world_to_ndc(Pw)
        if np.any(np.isnan(ndc)):
            return float('inf'), Pw, ndc
        # NDC -> pixel
        px = ndc_to_pixels(ndc[:2], v)
        cur_px = ndc_to_pixels(cursor_ndc_xy, v)
        dist = np.linalg.norm(px - cur_px)
        return float(dist), Pw, ndc

    # 2) Coarse sample
    samples = 64
    best = (float('inf'), 0.0, np.zeros(3), np.zeros(3))  # (dist, u, world, ndc)
    for i in range(samples + 1):
        u = i / samples
        d, Pw, ndc = px_distance_from_u(u)
        if d < best[0]:
            best = (d, u, Pw, ndc)

    if best[0] > snap_px:
        return None  # nothing close enough visually

    # 3) 1D refinement (golden section on pixel distance)
    def refine(u0: float, h: float = 1.0 / samples, iters: int = 20) -> Tuple[float, np.ndarray, np.ndarray, float]:
        a = max(0.0, u0 - 2*h)
        b = min(1.0, u0 + 2*h)
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
        else:
            return d, Pw_d, ndc_d, fd

    u_ref, Pw_ref, ndc_ref, dist_ref = refine(best[1])
    if dist_ref > snap_px:
        return None

    pix_ref = ndc_to_pixels(ndc_ref[:2], v)
    return {
        "u": float(u_ref),
        "world": tuple(Pw_ref.tolist()),
        "ndc": tuple(ndc_ref.tolist()),
        "pixel": tuple(pix_ref.tolist()),
        "dist_px": float(dist_ref),
    }


# =========================
# GL program
# =========================

VERT_SRC = """
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uProjection;
uniform mat4 uView;
uniform mat4 uModel;
void main(){
    gl_Position = uProjection * uView * uModel * vec4(aPos, 1.0);
}
"""

FRAG_SRC = """
#version 330 core
uniform vec4 uColor;
out vec4 FragColor;
void main(){
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




from dataclasses import dataclass
from functools import update_wrapper, partial
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Literal,
)

import numpy as np


# ---------------------------------------------------------------------------
# Shape pattern primitives
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Small example
# ---------------------------------------------------------------------------


class Viewer:
    def add(self, obj, *args,**kwargs):

        if isinstance(obj,RationalBezier):
            return self._add_curve(obj,*args,**kwargs)
        elif isinstance(obj,NURBSCurveTuple):
            return self._add_nurbs_curve(obj,*args,**kwargs)
        elif isinstance(obj,np.ndarray):
            if len(obj.shape)==2 and 'rational' in kwargs:
                return self.add_bern_curve(obj,*args,**kwargs)
            elif len(obj.shape )==1 and obj.shape[0]==3:
                return self.add_point3d(obj,*args,**kwargs)
            elif len(obj.shape )==1 and obj.shape[0]==2:
                return self.add_point2d(obj,*args,**kwargs)
            else:
                raise ValueError("Unsupported shape {obj.shape}")
        elif isinstance(obj,(tuple,list)):
            if len(obj)==2 :
                return self.add_point2d(obj,*args,**kwargs)
            elif len(obj)==3 :
                return self.add_point3d(obj,*args,**kwargs)
            else:
                raise ValueError(f"Unknown type: {obj}")



    def __init__(self, width=1200, height=800, camera=None):
        if not glfw.init():
            raise RuntimeError("GLFW init failed")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        # macOS retina framebuffer
        glfw.window_hint(glfw.COCOA_RETINA_FRAMEBUFFER, glfw.TRUE)

        self.window = glfw.create_window(width, height, "Snap Viewer", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("GLFW window creation failed")
        glfw.make_context_current(self.window)

        self.program = make_program()
        self.loc_uP = glGetUniformLocation(self.program, "uProjection")
        self.loc_uV = glGetUniformLocation(self.program, "uView")
        self.loc_uM = glGetUniformLocation(self.program, "uModel")
        self.loc_uColor = glGetUniformLocation(self.program, "uColor")

        glEnable(GL_DEPTH_TEST)
        self._rebuild_viewport()

        # Camera & input
        self.cam = OrbitCamera() if camera is None else camera
        self.dragging_orbit = False
        self.dragging_pan = False
        self.last_cursor_px = np.array([0.0, 0.0], dtype=np.float64)

        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_mouse_button_callback(self.window, self._on_mouse)
        glfw.set_scroll_callback(self.window, self._on_scroll)
        glfw.set_framebuffer_size_callback(self.window, self._on_resize)

        # Scene
        self.model = np.eye(4, dtype=np.float32)
        self.curves: List[RationalBezier] = []
        self.snap_px = 30.0
        self.snap_hit: Optional[dict] = None
        self.points=[]
        # GL buffers (recreated when curves change)
        self.lines = []  # list of (vao, vbo, nverts, color)


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

    def _add_curve(self, curve: RationalBezier, color=(1.0, 1.0, 1.0, 1.0), samples=128):
        self.curves.append(curve)
        l=len(self.curves) - 1
        # Build GL line strip
        pts = curve.polyline(samples=samples).astype(np.float32)
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, pts.nbytes, pts, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        self.lines.append((vao, vbo, pts.shape[0], np.array(color, dtype=np.float32)))
        return l

    def add_bern_curve(self, arr,*args, rational:bool=False,**kwargs):
        if not rational:

            A = np.zeros((arr.shape[0], 4))
            for i in range(arr.shape[1] ):
                A[..., i] = arr[..., i]
            A[..., -1] = 1



            curve = RationalBezier(A
                                   )
        else:

            if arr.shape[1]!=4:
                A = np.zeros((arr.shape[0], 4))
                for i in range(arr.shape[1]-1):
                    A[..., i] = arr[..., i]
                A[..., -1]=arr[..., -1]

            else:
                A=arr
            curve=RationalBezier(A)
        return self._add_curve(curve,*args,**kwargs)

    def add_point3d(self, arr, color=(0.8, 0.8, 0.8, 1.0),size_px=9):
        return self.points.append((np.array(arr,dtype=np.float32),color,size_px))


    def add_point2d(self, arr, color=(0.8, 0.8, 0.8, 1.0), size_px=9):
        return self.points.append((np.array((*arr,0), dtype=np.float32), color, size_px))


    def _build_demo_scene(self):
        # Two rational cubic Bézier curves (projective control points: [w*x, w*y, w*z, w])
        # Curve 1 (planar)
        P = np.array([
            [ -4.0, -2.0, 0.0, 1.0 ],
            [ -1.0,  3.0, 0.0, 0.7 ],
            [  2.0, -3.0, 0.0, 1.2 ],
            [  4.0,  2.0, 0.0, 1.0 ],
        ], dtype=np.float64)
        ctrl4 = np.column_stack((P[:, :3] * P[:, 3:4], P[:, 3:4]))  # [w*x,w*y,w*z,w]
        self._add_curve(RationalBezier(ctrl4), color=(1.0, 1.0, 1.0, 1.0))

        # Curve 2 (lifted)
        Q = np.array([
            [ -3.0,  3.5,  1.0, 1.0 ],
            [ -1.0, -1.0,  2.0, 0.8 ],
            [  1.0,  1.0, -2.0, 1.3 ],
            [  3.0, -2.5,  1.0, 1.0 ],
        ], dtype=np.float64)
        ctrl4b = np.column_stack((Q[:, :3] * Q[:, 3:4], Q[:, 3:4]))
        self._add_curve(RationalBezier(ctrl4b), color=(0.7, 0.9, 1.0, 1.0))

    # ---------- Render & snap ----------
    def _add_nurbs_curve(self, curve: NURBSCurveTuple, color=(1.0, 1.0, 1.0, 1.0)):
        beziers=decompose_curve(curve)

        return tuple(self.add(to_homogeneous_1d(bezier.control_points, bezier.weights), rational=True, color=color)        for bezier in beziers)


    def _upload_matrices(self, P_row: np.ndarray, V_row: np.ndarray, M_row: np.ndarray):
        """
        We build matrices row-major and ask GL to transpose them on upload.
        The shader then sees column-major matrices consistent with GLSL's math.
        CPU composite to match GLSL is: M_cpu = (P_row @ V_row @ M_row).T
        """
        glUseProgram(self.program)
        glUniformMatrix4fv(self.loc_uP, 1, GL_TRUE, P_row)
        glUniformMatrix4fv(self.loc_uV, 1, GL_TRUE, V_row)
        glUniformMatrix4fv(self.loc_uM, 1, GL_TRUE, M_row)

    def _draw_lines(self):
        for (vao, vbo, nverts, color) in self.lines:
            glBindVertexArray(vao)
            glUseProgram(self.program)
            glUniform4fv(self.loc_uColor, 1, color)
            glDrawArrays(GL_LINE_STRIP, 0, nverts)

    def _draw_point(self, pos_world: np.ndarray, size_px=7.0, color=(1.0, 1.0, 0.0, 1.0)):
        # Build a tiny VBO on the fly
        pts = np.array(pos_world, dtype=np.float32).reshape(1, 3)
        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, pts.nbytes, pts, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

        # Draw as points
        glUseProgram(self.program)
        glPointSize(max(1.0, float(size_px)))
        glUniform4fv(self.loc_uColor, 1, np.array(color, dtype=np.float32))
        glDrawArrays(GL_POINTS, 0, 1)

        glDeleteBuffers(1, [vbo])
        glDeleteVertexArrays(1, [vao])

    def _compute_snap(self, vinfo: ViewportInfo, M_cpu: np.ndarray):
        # Read cursor (GLFW points) -> NDC
        cursor_pt = glfw.get_cursor_pos(self.window)
        cursor_ndc = glfw_cursor_to_ndc(cursor_pt, vinfo)
        # Pick best among curves
        best = None
        for cv in self.curves:
            hit = snap_curve_to_cursor(cv, M_cpu, vinfo, cursor_ndc, snap_px=self.snap_px)
            if hit is None:
                continue
            if best is None or hit["dist_px"] < best["dist_px"]:
                best = hit
        self.snap_hit = best

    def run(self):
        last = time.time()
        while not glfw.window_should_close(self.window):
            glfw.poll_events()

            # Prepare transforms
            vinfo = read_viewport_info(self.window)
            aspect = vinfo.vw / max(1, vinfo.vh)
            P_row = self.cam.projection_matrix(aspect)
            V_row = self.cam.view_matrix()
            M_row = self.model
            # CPU composite matching GLSL column-major:
            M_cpu = (P_row @ V_row @ M_row).T

            # Upload matrices
            self._upload_matrices(P_row, V_row, M_row)

            # Snap
            self._compute_snap(vinfo, M_cpu)

            # Draw
            glClearColor(0.07, 0.07, 0.08, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self._draw_lines()
            for point_arr,color,size_px in self.points:
                self._draw_point(point_arr,
                                 size_px=size_px, color=color)

            # Snap marker
            if self.snap_hit is not None:
                self._draw_point(np.array(self.snap_hit["world"], dtype=np.float32),
                                 size_px=9.0, color=(1.0, 1.0, 0.0, 1.0))

            glfw.swap_buffers(self.window)

        # Cleanup
        for (vao, vbo, _, _) in self.lines:
            glDeleteBuffers(1, [vbo])
            glDeleteVertexArrays(1, [vao])
        glfw.terminate()


if __name__ == "__main__":
    Viewer().run()
