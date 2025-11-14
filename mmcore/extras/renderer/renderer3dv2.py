from __future__ import annotations


import threading
import time

import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
import pyrr
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import platform
from mmcore.geom._nurbs_eval import NURBSCurveTuple, _tuple_to_nurbs, _nurbs_to_tuple, NURBSSurfaceTuple
from mmcore.geom.nurbs import NURBSCurve, NURBSSurface, decompose_surface, greville_abscissae, decompose_curve

from mmcore.geom.nurbs_iso import extract_surface_boundaries, extract_isocurve
from mmcore.numeric.approx import adaptive_curve_sampler
from mmcore.numeric.closest_point import nurbs_surface_closest_point
from mmcore.numeric.sbern import bern_to_nurbs_bezier
from mmcore.topo.mesh.tess import tessellate_surface, surface_to_mesh

DEFAULT_BACKGROUND_COLOR = 158 / 256, 162 / 256, 169 / 256, 1.0
DEFAULT_DARK_BACKGROUND_COLOR = 20 / 256, 20 / 256, 20 / 256, 1.0

# ==== Unified conversions (GLFW points <-> framebuffer pixels <-> NDC) ======
import numpy as np

class ViewportInfo:
    """Cache everything we need for a consistent mapping."""
    def __init__(self, window):
        # GL viewport in pixels (OpenGL origin = bottom-left)
        vx, vy, vw, vh = glGetIntegerv(GL_VIEWPORT)
        self.vx, self.vy, self.vw, self.vh = int(vx), int(vy), int(vw), int(vh)
        # Framebuffer (pixels)
        self.fb_w_px, self.fb_h_px = glfw.get_framebuffer_size(window)
        # GLFW window content scale (points -> pixels)
        sx, sy = glfw.get_window_content_scale(window)
        self.sx, self.sy = float(sx), float(sy)
        # (optional) window size in points
        self.win_w_pt, self.win_h_pt = glfw.get_window_size(window)

def points_to_pixels(xy_pt, v: ViewportInfo):
    """GLFW points (origin top-left) -> pixels (origin bottom-left)."""
    x_px = xy_pt[0] * v.sx
    y_px_top = xy_pt[1] * v.sy
    y_px = (v.vy + v.vh) - y_px_top  # flip Y once into GL space
    return np.array([x_px, y_px], float)

def pixels_to_points(xy_px, v: ViewportInfo):
    """Pixels (origin bottom-left) -> GLFW points (origin top-left)."""
    x_pt = xy_px[0] / v.sx
    y_pt_top = (v.vy + v.vh) - xy_px[1]
    y_pt = y_pt_top / v.sy
    return np.array([x_pt, y_pt], float)

def pixels_to_ndc(xy_px, v: ViewportInfo):
    """Pixels (origin bottom-left) -> NDC ([-1,1]^2)."""
    x_ndc = ((xy_px[0] - v.vx) / v.vw) * 2.0 - 1.0
    y_ndc = ((xy_px[1] - v.vy) / v.vh) * 2.0 - 1.0
    return np.array([x_ndc, y_ndc], float)

def ndc_to_pixels(ndc_xy, v: ViewportInfo):
    """NDC -> pixels (origin bottom-left)."""
    x_px = v.vx + (ndc_xy[0] + 1.0) * 0.5 * v.vw
    y_px = v.vy + (ndc_xy[1] + 1.0) * 0.5 * v.vh
    return np.array([x_px, y_px], float)

def glfw_cursor_to_ndc(cursor_xy_points, v: ViewportInfo):
    """GLFW points -> NDC."""
    return pixels_to_ndc(points_to_pixels(cursor_xy_points, v), v)

def create_isolines(u_vals, v_vals):
    """
    Create three lists of isolines for the domain defined by u_vals and v_vals:
      1) boundary_isolines:   (direction, parameter) at the min and max of each set
      2) param_isolines:      (direction, parameter) for each 'internal' parameter in the sets
      3) midpoint_isolines:   (direction, parameter) for midpoints of each interval,
                              skipping duplicates
    Returns
    -------
    boundary_isolines, param_isolines, midpoint_isolines : 3 lists of (dir, param) tuples
    """
    # --- 1) BOUNDARY ISOLINES ---
    # For u and v, the boundaries are just the first and last values in each list
    boundary_set = set()
    boundary_set.add(("u", u_vals[0]))
    boundary_set.add(("u", u_vals[-1]))
    boundary_set.add(("v", v_vals[0]))
    boundary_set.add(("v", v_vals[-1]))
    # --- 2) PARAMETER ISOLINES (INTERNAL) ---
    # These are all the values except the boundary in each list
    param_set = set()
    for val in u_vals[1:-1]:
        param_set.add(("u", val))
    for val in v_vals[1:-1]:
        param_set.add(("v", val))
    # --- 3) MIDPOINT ISOLINES ---
    # For each consecutive pair (a, b), take midpoint m = 0.5*(a + b),
    # but skip if that midpoint is exactly one of the existing lines
    midpoint_set = set()

    def add_midpoints(values, direction):
        for i in range(len(values) - 1):
            a = values[i]
            b = values[i + 1]
            m = 0.5 * (a + b)
            candidate = (direction, m)
            # Only add if not already in boundary_set or param_set
            if candidate not in boundary_set and candidate not in param_set:
                midpoint_set.add(candidate)

    add_midpoints(u_vals, "u")
    add_midpoints(v_vals, "v")
    # Convert each set to a sorted list (sorted by direction first, then parameter).
    # Sorting by direction ensures all ("u", ...) come before ("v", ...).
    # You can adjust sorting logic if you prefer a different order.
    boundary_isolines = sorted(boundary_set, key=lambda x: (x[0], x[1]))
    param_isolines = sorted(param_set, key=lambda x: (x[0], x[1]))
    midpoint_isolines = sorted(midpoint_set, key=lambda x: (x[0], x[1]))
    return boundary_isolines, param_isolines, midpoint_isolines


@dataclass(unsafe_hash=True)
class Point:
    position: np.ndarray  # 3D vector
    color: np.ndarray  # RGB vector
    size: float


@dataclass
class Wire:
    vertices: np.ndarray  # Nx3 array of vertices
    color: np.ndarray  # RGB vector
    thickness: float


@dataclass
class Mesh:
    vertices: np.ndarray  # Nx3 array of vertices
    triangles: np.ndarray  # Mx3 array of triangle indices
    color: np.ndarray  # RGBA vector (with alpha for transparency)
    wireframe_color: Optional[np.ndarray] = None  # RGB vector for wireframe, if None will use a darker version of color

def naive_creases(surf):
    from mmcore.geom.nurbs_iso import extract_surface_boundaries_tuple
    from mmcore.geom._nurbs_knots import decompose_surface
    srf=    _nurbs_to_tuple(surf)
    parts=decompose_surface(srf)

  

    edges = {}
    for i in range(len(parts)):
        for j in range(len(parts)):
            if i != j:
                key = min(i, j), max(i, j)
                if key not in edges:
                    bnd1 = extract_surface_boundaries_tuple(parts[i])
                    bnd2 = extract_surface_boundaries_tuple(parts[j])
                    for b1 in bnd1:
                        for b2 in bnd2:
                            if b1.control_points.shape == b2.control_points.shape:
                                if np.allclose(b1.control_points, b2.control_points):
                                    edges[key] = b1
                                    continue
    creases = []
    for (i, j), edge_curve in edges.items():
        tmin, tmax = edge_curve.interval()
        for pt in edge_curve.control_points:
            uv1, (coord1, evals1, _) = nurbs_surface_closest_point(
                parts[i], pt, angle_tol=0.052
            )
            uv2, (coord2, evals2, _) = nurbs_surface_closest_point(
                parts[j], pt, angle_tol=0.052
            )
            N1 = np.cross(evals1["Su"], evals1["Sv"])
            N1 /= np.linalg.norm(N1)
            N2 = np.cross(evals2["Su"], evals2["Sv"])
            N2 /= np.linalg.norm(N2)
            if (1 - np.abs(np.dot(N1, N2))) > 0.052:
                creases.append((i, j))
                break
    return [(_tuple_to_nurbs(edges[key]),key ) for key in creases]
def nurbs_surface_wireframe_view(surf: NURBSSurface):
    (u_min, u_max), (v_min, v_max) = surf.interval()

    u_iso = extract_isocurve(surf, (u_min + u_max) * 0.5, direction="u")
    v_iso = extract_isocurve(surf, (v_min + v_max) * 0.5, direction="v")
    
    boundaries = extract_surface_boundaries(surf)
    
    return boundaries, [u_iso, v_iso]+[crv for crv,_ in naive_creases(surf)], []


from numpy.typing import NDArray
from mmcore.geom._nurbs_eval import to_homogeneous_1d

@dataclass
class BoundingSphere:
    origin: field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32))
    radius: float = 0.0

    def compute_from_geometries(self, points=None, wires=None, meshes=None):
        """Compute bounding sphere from existing geometries"""
        all_points = []

        # Add all points
        if points:
            for point in points:
                all_points.append(point.position)

        # Add wire vertices
        if wires:
            for wire in wires:
                all_points.extend(wire.vertices)

        # Add mesh vertices
        if meshes:
            for mesh in meshes:
                all_points.extend(mesh.vertices)

        if not all_points:
            return

        # Compute center and radius
        all_points = np.array(all_points)
        self.origin = np.mean(all_points, axis=0)

        # Calculate radius as the max distance from any point to the center
        if len(all_points) > 0:
            distances = np.linalg.norm(all_points - self.origin, axis=1)
            self.radius = np.max(distances)


@dataclass
class Camera:
    pos: NDArray[np.float32] = field(default_factory=lambda: np.array([150.0, 150.0, 150.0], dtype=np.float32))
    target: NDArray[np.float32] = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32))
    up: NDArray[np.float32] = field(default_factory=lambda: np.array([0.0,0.0, 1.0], dtype=np.float32))
    zoom: float = 1.0
    near: float = 0.1
    far: float = 1000000.0
    is_panning: bool = False

    def position_from_bounding_sphere(self, sphere: BoundingSphere):
        """Position camera based on bounding sphere to ensure geometry is in view

        Similar to the JS code:
        const cameraOffset = new Vector3(0, radius * 1.5, radius * 2.5);
        const newCamPos = new Vector3().addVectors(center, cameraOffset);
        camera.position.copy(newCamPos);
        camera.lookAt(center);
        """
        if sphere.radius <= 0:
            return

        # Set target to sphere center
        self.target = np.array(sphere.origin, dtype=np.float32)

        # Define camera offset (similar to JS example)
        camera_offset = np.array([0, sphere.radius * 1.5, sphere.radius * 2.5], dtype=np.float32)

        # Position camera
        self.pos = np.array(sphere.origin + camera_offset, dtype=np.float32)

        # Adjust zoom based on radius (optional)
        self.zoom = max(1.0, sphere.radius * 1.5)


import multiprocessing as mp
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

from mmcore.numeric.bern import bernstein_eval_1d, bern_roots_1d

def pixel_planes_world(M_world_to_clip: np.ndarray, u_ndc: float, v_ndc: float):
    # x' - u*w' = 0,  y' - v*w' = 0 in clip space
    p_x = np.array([1.0, 0.0, 0.0, -u_ndc], dtype=float)
    p_y = np.array([0.0, 1.0, 0.0, -v_ndc], dtype=float)
    MT = M_world_to_clip.T              # <-- GPU-consistent pullback
    return MT @ p_x, MT @ p_y

# --- main: snap on a rational Bézier using your 1D Bernstein rooter ----------
def snap_bezier_rational_with_bern_roots(
    ctrl4, M_world_to_clip, u_ndc, v_ndc,
    bern_roots_1d=bern_roots_1d,
    eval_scalar=bernstein_eval_1d,
    eps_root=1e-6,
    cross_tol=1e-5,     # a bit more forgiving
    band=1e-8
):
    # Clip-space pixel planes: x' - u*w' = 0  and  y' - v*w' = 0
    p_x = np.array([1.0, 0.0, 0.0, -u_ndc], dtype=float)
    p_y = np.array([0.0, 1.0, 0.0, -v_ndc], dtype=float)

    # Transform control net to CLIP once (matches shader path exactly)
    clip_ctrl = (M_world_to_clip @ ctrl4.T).T  # (n+1, 4)

    # Scalar Bézier control values for residuals
    rx_ctrl = clip_ctrl @ p_x
    ry_ctrl = clip_ctrl @ p_y

    # Interval band cull (cheap and robust)
    if (rx_ctrl.max() < -band) or (rx_ctrl.min() > band): return False, None, None, float('inf')
    if (ry_ctrl.max() < -band) or (ry_ctrl.min() > band): return False, None, None, float('inf')

    roots_x = bern_roots_1d(rx_ctrl, eps=eps_root).roots
    roots_y = bern_roots_1d(ry_ctrl, eps=eps_root).roots

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

    u_star, score = min(candidates, key=lambda t: t[1])

    # Evaluate world point at u_star using homogeneous De Casteljau
    a = ctrl4.astype(float).copy()
    for _ in range(len(a) - 1):
        a = (1.0 - u_star) * a[:-1] + u_star * a[1:]
    Ch = a[0]
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
def cursor_from_curve_param_v2(ctrl4, u, M_world_to_clip, v: ViewportInfo, eps_w: float = 1e-30):
        # Evaluate homogeneous Bézier
        a = ctrl4.astype(float).copy()
        for _ in range(len(a) - 1):
            a = (1.0 - u) * a[:-1] + u * a[1:]
        Ch = a[0]  # [X, Y, Z, W]
        w = Ch[3]
        world = (Ch[:3] / w) if abs(w) > eps_w else np.array([np.nan] * 3)

        # ---- compute clip from EUCLIDEAN point with the GPU matrix ----
        P_eu = np.array([world[0], world[1], world[2], 1.0], dtype=float)
        clip = M_world_to_clip @ P_eu
        if abs(clip[3]) <= eps_w:
            ndc = np.array([np.nan] * 3)
            pixels = points = np.array([np.nan] * 2)
        else:
            ndc = clip[:3] / clip[3]
            pixels = ndc_to_pixels(ndc[:2], v)
            points = pixels_to_points(pixels, v)

        return {"points": tuple(points), "pixels": tuple(pixels),
                "ndc": tuple(ndc), "world": tuple(world), "clip": tuple(clip)}

def _gen_cpts_to_display(scalar_net, interval:list[tuple[float,float]]=None):
   
    def get_interv(n)->tuple[float,float]:
        if interval is None:
            return (0., 1.)
        else:
            return interval[n]
    Pts = np.zeros((*scalar_net.shape, scalar_net.ndim + 1))
    
    mgr = np.mgrid[*(slice(*get_interv(i), complex(scalar_net.shape[i])) for i in range(scalar_net.ndim))]
  
    Pts[..., -1] = scalar_net
    Pts[..., :-1] = np.moveaxis(mgr, 0, -1)
    
    return Pts

class CADRenderer:

    def __init__(self, width=800, height=600, background_color=DEFAULT_DARK_BACKGROUND_COLOR, camera:  Camera = None, sampling_tol:float=1e-2, free_axis:bool=False):
        self._needs_to_update = True
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        if camera is None:
            camera = Camera()
        self._background_color = background_color
        self.camera_pos = camera.pos.copy()
        self.camera_target = camera.target.copy()
        self.camera_up = camera.up.copy()
        self.zoom = camera.zoom
        self.near = camera.near
        self.far = camera.far
        self.free_axis = free_axis
        self.is_panning = camera.is_panning
        self.auto_position_camera = True
        self.bsf = BoundingSphere(camera.target.copy(), 0.0)
        self.sampling_tol = sampling_tol
        self._projection=None
        self._platform = platform.system()
        # GLFW & OpenGL config
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        if self._platform=="Darwin":
            glfw.window_hint(glfw.COCOA_RETINA_FRAMEBUFFER, True)
  
        # Camera settings
        if camera is None:
            camera = Camera()
        self.camera_pos = camera.pos
        self.camera_target = camera.target
        self.camera_up = camera.up
        self.zoom = camera.zoom
        self.is_panning = camera.is_panning
        self.near = camera.near
        self.far = camera.far
        
        # Mouse interaction
        self.is_dragging = False
        self.last_mouse_pos = np.array([0.0, 0.0])
        self.snap_distance = 0.001
        self.framebuffer_size=None
        self.width = width
        self.height = height
        self.window = None
        # Geometry storage is already initialized above
        self.default_vao=None
        self.default_vao_color=None
        self._stop_mouse_event=0
        # Geometry lists
        self.points: List[Point] = []
        self._temporal_points: List[Point] = []
        self.wires: List[Wire] = []
        self.meshes: List[Mesh] = []
        self._hcurves=[]
        self._snap_cache=dict()
        self._snap_mode=True
        self._warping_cursor = False
        #self._run_thread=threading.Thread(target=self._run, name="renderer3dv2", daemon=True)
    
    def create_window(self):
        
        if self.window is not None and not glfw.window_should_close(self.window):
            return
        from mmcore import __version__
        self.window = glfw.create_window(self.width, self.height, f"mmcore@{__version__}", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        glfw.make_context_current(self.window)
        
        #print("OpenGL version:", glGetString(GL_VERSION).decode())
        #print("GLSL version:", glGetString(GL_SHADING_LANGUAGE_VERSION).decode())
        # Setup callbacks
        self.setup_callbacks()
        
        # Shaders
        self.setup_shaders()
        self._needs_to_update=True
        # GL state
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(3, GL_POLYGON_OFFSET_UNITS)
        # Enable program point size so gl_PointSize is honored
        glEnable(GL_PROGRAM_POINT_SIZE)
        
        # Default VAO
        self.default_vao = glGenVertexArrays(1)
        glBindVertexArray(self.default_vao)
        self.framebuffer_size = glfw.get_framebuffer_size(self.window)
        glViewport(0, 0, *self.framebuffer_size)
        self._should_terminate=False
        
    def stop(self):
        glfw.destroy_window(self.window)
        
        
    def setup_callbacks(self):
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self._mouse_move_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)
        glfw.set_framebuffer_size_callback(self.window, self._framebuffer_size_callback)
        self.setup_focus_callback(window=self.window)
        glfw.set_window_close_callback(self.window, self._window_should_close_callback)
        glfw.set_window_size_callback(self.window, self._on_window_resize)

    def _framebuffer_size_callback(self, window, width, height):
        # print("mouse_move",window, width, height)
        glViewport(0, 0, width, height)
   
        
        self.framebuffer_size = (width, height)
        self._needs_to_update=True

    def _mouse_button_callback(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            if mods & glfw.MOD_SHIFT:
                self.is_panning = action == glfw.PRESS
            else:
                self.is_dragging = action == glfw.PRESS
        if button == glfw.MOUSE_BUTTON_RIGHT:
            self.is_panning = action == glfw.PRESS

        if self.is_dragging or self.is_panning:
            x_pt, y_pt = glfw.get_cursor_pos(window)
            v = ViewportInfo(window)
            self.last_mouse_pos = points_to_pixels((x_pt, y_pt), v)

        self._needs_to_update = True

    def setup_shaders(self):
        vertex_shader_source = """
        #version 410
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec4 color;
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform float uPointSize;
        out vec4 vertex_color;
        void main() {
            gl_Position = projection * view * model * vec4(position, 1.0);
            gl_PointSize = uPointSize;
            vertex_color = color;
        }
        """
        fragment_shader_source = """
        #version 410
        in vec4 vertex_color;
        out vec4 FragColor;
        void main() {
            FragColor = vertex_color;
        }
        """
        vs = shaders.compileShader(vertex_shader_source, GL_VERTEX_SHADER)
        fs = shaders.compileShader(fragment_shader_source, GL_FRAGMENT_SHADER)
        prog = glCreateProgram()
        glAttachShader(prog, vs)
        glAttachShader(prog, fs)
        glLinkProgram(prog)
        if not glGetProgramiv(prog, GL_LINK_STATUS):
            raise RuntimeError(glGetProgramInfoLog(prog))
        glValidateProgram(prog)
        glDeleteShader(vs)
        glDeleteShader(fs)
        self.shader_program = prog


    def update_camera_position(self):
        """Update camera position based on scene geometry"""
        if not self.auto_position_camera:
            return

        # Compute bounding sphere from all geometry
        self.bsf.compute_from_geometries(self.points, self.wires, self.meshes)

        # Use Camera class method to position camera from bounding sphere
        if self.bsf.radius > 0:
            camera_data = Camera(
                pos=self.camera_pos, target=self.camera_target, up=self.camera_up, zoom=self.zoom, near=self.near, far=self.far
            )
            camera_data.position_from_bounding_sphere(self.bsf)

            # Update renderer camera properties
            self.camera_pos = camera_data.pos
            self.camera_target = camera_data.target
            self.zoom = camera_data.zoom
    def check(self):
        v = ViewportInfo(self.window)
        print("viewport(px):", (v.vx, v.vy, v.vw, v.vh))
        print("framebuffer(px):", (v.fb_w_px, v.fb_h_px))
        print("content_scale:", (v.sx, v.sy))
        # Roundtrip check
        pt_dbg = np.array([200.0, 300.0])
        rt = pixels_to_points(points_to_pixels(pt_dbg, v), v)

        print("roundtrip error (points):", np.linalg.norm(rt - pt_dbg))
    def add_mesh(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        color: np.ndarray = np.array([0.5, 0.5, 0.5, 0.5]),
        wireframe_color: Optional[np.ndarray] = np.array([0.0, 0.0, 0.0]),
    ):
        """Add a mesh to the scene"""
        # Ensure vertices are float32
        vertices = np.array(vertices, dtype=np.float32)

        # Ensure triangles are uint32
        triangles = np.array(triangles, dtype=np.uint32)

        # If color doesn't have alpha, add 0.5 alpha
        if len(color) == 3:
            color = np.append(color, 0.5)
        color = np.array(color, dtype=np.float32)

        # If wireframe color is provided, ensure it's RGB
        if wireframe_color is not None:
            wireframe_color = np.array(wireframe_color[:3], dtype=np.float32)

        # Add mesh to the scene
        self.meshes.append(Mesh(vertices, triangles, color, wireframe_color))
        self.update_camera_position()
        import threading
    def add_point(self, position, color=np.array([1.0, 1.0, 1.0]), size=5.0, temporary:bool=False):
        pos = np.array(position, dtype=np.float32)
        col = np.array(color, dtype=np.float32)
        pt=Point(pos, col, size)

        if temporary:
            self._temporal_points.append(pt)
        else:
            self.points.append(pt)

        return pt
        

    def add_wire(self, vertices: np.ndarray, color: np.ndarray = np.array([1.0, 1.0, 1.0]), thickness: float = 1.0):
        """Add a wire (curve) to the scene"""
        vxs=np.array(vertices)
        
        if vxs.shape[-1]<3:
            z=np.zeros((len(vxs),3), dtype=np.float32)
            z[...,:vxs.shape[-1]]=vxs
            vxs=z
        self.wires.append(Wire(vxs, color, thickness))
        self.update_camera_position()
        
    def _on_window_resize(self, window,width, height):
        self.width=width
        self.height=height
        self._needs_to_update = True

    def _mouse_move_callback(self, window, x_pt, y_pt):
        v = ViewportInfo(window)
        if self._warping_cursor:
            self.last_mouse_pos = points_to_pixels((x_pt, y_pt), v)
            self._warping_cursor = False
            return

        current_pos_px = points_to_pixels((x_pt, y_pt), v)

        if self.is_dragging or self.is_panning:
            delta = current_pos_px - self.last_mouse_pos

            fb_w, fb_h = v.vw, v.vh
            aspect = fb_w / max(fb_h, 1)
            if self.is_panning:
                world_delta_x = (delta[0] / max(fb_w, 1)) * self.zoom * 2 * aspect
                world_delta_y = -(delta[1] / max(fb_h, 1)) * self.zoom * 2
                pan_vector = self.camera_right * world_delta_x + self.camera_up * world_delta_y
                self.camera_pos -= pan_vector
                self.camera_target -= pan_vector

            elif self.is_dragging and not self.free_axis:
                sensitivity = 0.005
                rotation_x = pyrr.matrix44.create_from_axis_rotation(self.camera_up, delta[0] * sensitivity,
                                                                     dtype=np.float32)
                rotation_y = pyrr.matrix44.create_from_axis_rotation(self.camera_right, delta[1] * sensitivity,
                                                                     dtype=np.float32)
                camera_to_target = self.camera_pos - self.camera_target

                camera_to_target = np.dot(rotation_x, np.append(camera_to_target, 1.0))[:3]
                camera_to_target = np.dot(rotation_y, np.append(camera_to_target, 1.0))[:3]
                self.camera_pos = self.camera_target + camera_to_target

            elif self.is_dragging and self.free_axis:
                sensitivity = 0.005
                camera_to_target = self.camera_pos - self.camera_target
                if np.linalg.norm(camera_to_target) > 1e-8:
                    yaw_angle = delta[0] * sensitivity
                    pitch_angle = delta[1] * sensitivity
                    if abs(yaw_angle) > 1e-8 and np.linalg.norm(self.camera_up) > 1e-8:
                        yaw_axis = self.camera_up / np.linalg.norm(self.camera_up)
                        rotation_yaw = pyrr.matrix44.create_from_axis_rotation(yaw_axis, yaw_angle, dtype=np.float32)
                        camera_to_target = np.dot(rotation_yaw, np.append(camera_to_target, 1.0))[:3]
                    temp_pos = self.camera_target + camera_to_target
                    forward = self.camera_target - temp_pos
                    if np.linalg.norm(forward) < 1e-8:
                        forward = self.camera_target - self.camera_pos
                    forward /= np.linalg.norm(forward)
                    right_axis = np.cross(forward, self.camera_up)
                    if np.linalg.norm(right_axis) < 1e-8:
                        right_axis = self.camera_right
                    right_axis /= np.linalg.norm(right_axis)
                    if abs(pitch_angle) > 1e-8:
                        rotation_pitch = pyrr.matrix44.create_from_axis_rotation(right_axis, pitch_angle,
                                                                                 dtype=np.float32)
                        camera_to_target = np.dot(rotation_pitch, np.append(camera_to_target, 1.0))[:3]
                        self.camera_up = np.dot(rotation_pitch, np.append(self.camera_up, 1.0))[:3]

                self.camera_pos = self.camera_target + camera_to_target
                forward = self.camera_target - self.camera_pos
                if np.linalg.norm(forward) > 1e-8 and np.linalg.norm(self.camera_up) > 1e-8:
                    forward /= np.linalg.norm(forward)
                    right = np.cross(forward, self.camera_up)
                    if np.linalg.norm(right) > 1e-8:
                        right /= np.linalg.norm(right)
                        self.camera_up = np.cross(right, forward)
                        self.camera_up /= np.linalg.norm(self.camera_up)

            self.last_mouse_pos = current_pos_px

        else:
            snap_pts = self.perform_snap()
            if snap_pts is not None:
                self._warping_cursor = True
                glfw.set_cursor_pos(window, float(snap_pts[0]), float(snap_pts[1]))
                current_pos_px = points_to_pixels(snap_pts, v)
            self.last_mouse_pos = current_pos_px

        self._needs_to_update = True

    def perform_snap(self):
        vinfo = ViewportInfo(self.window)

        # Clamp input cursor to the window content rect
        cursor_xy_points = glfw.get_cursor_pos(self.window)

        ndc_xy = glfw_cursor_to_ndc(cursor_xy_points, vinfo)
        u_ndc, v_ndc = float(ndc_xy[0]), float(ndc_xy[1])

        M = self.M_gpu  # projection @ view @ model

        best = None
        for i, hcurve in enumerate(self._hcurves):
            ok, u_star, Pw, score = snap_bezier_rational_with_bern_roots(
                hcurve, M, u_ndc, v_ndc, bern_roots_1d=bern_roots_1d,
                cross_tol=self.snap_distance  # try 1e-5 .. 3e-5 if needed
            )
            if not ok:
                continue

            res = cursor_from_curve_param_v2(hcurve, u_star, M, vinfo)
            pts = np.array(res["points"], float)
            if np.any(np.isnan(pts)):
                continue

            if (best is None) or (score < best[0]):
                best = (score, pts, res["world"], u_star, i)

        if best:
            score, pts, world, u_star, idx = best
            self.add_point(world, np.array([1., 1., 0.]), temporary=True)
            return pts

        return None

    def _scroll_callback(self, window, xoffset, yoffset):
        # Modify zoom for orthographic projection
        #print("scroll", window, xoffset, yoffset)
        zoom_factor = 0.1
        self.zoom *= 1.0 - yoffset * zoom_factor
        self.zoom = np.clip(self.zoom, self.near, self.far)
    
        self._needs_to_update=True
    @property
    def camera_right(self):
        # Get the camera's right vector
        forward =  self.camera_target-self.camera_pos
        forward = forward / np.linalg.norm(forward)

        right = np.cross( forward,self.camera_up)
        

        return right / np.linalg.norm(right)

    def render_mesh(self, mesh: Mesh):
        """Render a single mesh with transparency"""
        # Create and bind VAO
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)

        # VBO: positions
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, mesh.vertices.nbytes, mesh.vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        # Colors (per-vertex)
        colors = np.tile(mesh.color, (len(mesh.vertices), 1)).astype(np.float32)
        cbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, cbo)
        glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)

        # Create and bind element buffer object (EBO)
        ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.triangles.nbytes, mesh.triangles, GL_STATIC_DRAW)

        # Disable depth writes for transparent surface
        glDepthMask(GL_FALSE)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glDrawElements(GL_TRIANGLES, mesh.triangles.size * 3, GL_UNSIGNED_INT, None)
        glDepthMask(GL_TRUE)

        # If wireframe is requested, draw wireframe on top
        if mesh.wireframe_color is not None:
            wf = np.zeros((len(mesh.vertices), 4), dtype=np.float32)
            wf[:, :3] = mesh.wireframe_color
            wf[:, 3] = 1.0
            glBindBuffer(GL_ARRAY_BUFFER, cbo)
            glBufferData(GL_ARRAY_BUFFER, wf.nbytes, wf, GL_STATIC_DRAW)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glLineWidth(1.0)
            glDrawElements(GL_TRIANGLES, mesh.triangles.size * 3, GL_UNSIGNED_INT, None)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        # Cleanup
        glDeleteBuffers(1, [vbo, cbo, ebo])
        glDeleteVertexArrays(1, [vao])
    @property
    def projection(self):
        return self._projection
    @projection.setter
    def projection(self, value):
      
        self._snap_cache.clear()
        self._projection = value
        
    def render(self):
        if self._needs_to_update:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glClearColor(*self._background_color)

            w, h = self.framebuffer_size
            aspect = w / h

            # Build NumPy-side matrices (row-major math is fine)
            self.projection = pyrr.matrix44.create_orthogonal_projection(
                -self.zoom * aspect, self.zoom * aspect,
                -self.zoom, self.zoom, self.near, self.far, dtype=np.float32
            )
            self.view = pyrr.matrix44.create_look_at(self.camera_pos, self.camera_target, self.camera_up,
                                                     dtype=np.float32)
            self.model = pyrr.matrix44.create_identity(dtype=np.float32)

            # ---- UPLOAD TRANSPOSED ----
            glUseProgram(self.shader_program)
            glUniformMatrix4fv(glGetUniformLocation(self.shader_program, "projection"), 1, GL_TRUE, self.projection)
            glUniformMatrix4fv(glGetUniformLocation(self.shader_program, "view"), 1, GL_TRUE, self.view)
            glUniformMatrix4fv(glGetUniformLocation(self.shader_program, "model"), 1, GL_TRUE, self.model)

            # CPU composite must mirror what the shader actually uses
            #self.M_gpu = self.projection @ self.view @ self.model
            self.M_gpu = self.projection.T @ self.view.T @ self.model.T
            # ---- COMPOSITE USED BY GPU (and by the CPU snap code) ----
            # IMPORTANT: this is the *exact* matrix the shader uses
            print("P:", self.projection.T)
            print("V:", self.view.T)
            print("M:", self.model.T)
            print("M_gpu = P @ V @ M:", self.M_gpu )
            print("|P|,|V|,|M|:", np.linalg.norm(self.projection), np.linalg.norm(self.view),
                  np.linalg.norm(self.model))
            print("Using M_gpu = P @ V @ M,  ||M_gpu||:", np.linalg.norm(self.M_gpu))



            # Draw meshes first (transparent)
            for m in self.meshes:
                self.render_mesh(m)
            # Then points
            for p in self.points:
                self.render_point(p)
            # Then wires
            for w in self.wires:
                self.render_wire(w)
            for p in self._temporal_points:
                self.render_point(p)
            self._needs_to_update=False
        else:
            time.sleep(1/60)
        
        
    def render_point(self, point: Point):
        # Pass size via uniform instead of glPointSize
        size_loc = glGetUniformLocation(self.shader_program, "uPointSize")
        glUniform1f(size_loc, point.size * 2.0)  # retina scaling

        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)

        # Position VBO
        p_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, p_vbo)
        glBufferData(GL_ARRAY_BUFFER, point.position.nbytes, point.position, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        # Color VBO
        rgba = np.zeros(4, dtype=np.float32)
        rgba[:3] = point.color
        rgba[3] = 1.0
        c_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, c_vbo)
        glBufferData(GL_ARRAY_BUFFER, rgba.nbytes, rgba, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)

        # Draw
        glDrawArrays(GL_POINTS, 0, 1)

        # Cleanup
        glDeleteBuffers(1, [p_vbo, c_vbo])
        glDeleteVertexArrays(1, [vao])

    def render_wire(self, wire: Wire):
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)

        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, wire.vertices.nbytes, wire.vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        cols = np.zeros((len(wire.vertices), 4), dtype=np.float32)
        cols[:, :3] = wire.color
        cols[:, 3] = 1.0
        c_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, c_vbo)
        glBufferData(GL_ARRAY_BUFFER, cols.nbytes, cols, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)

        glDrawArrays(GL_LINE_STRIP, 0, len(wire.vertices))
        # Determine the supported line-width range:
        line_range = glGetFloatv(GL_SMOOTH_LINE_WIDTH_RANGE)
        # Clamp your requested width into [min, max]:
        desired = wire.thickness * 2.0  # if you really want retina scaling
        width = max(line_range[0], min(desired, line_range[1]))
        glLineWidth(width)

        glDrawArrays(GL_LINE_STRIP, 0, len(wire.vertices))

        glDeleteBuffers(1, [vbo, c_vbo])
        glDeleteVertexArrays(1, [vao])
        
    def _focus_callback(self, window,focus,*args):
        #print("focus",window,focus,*args)
        self._needs_to_update=bool(focus)
        if focus==1:
            self._needs_to_update=True
       
        else:
            
            self._needs_to_update = False


    def setup_focus_callback(self, window):
        glfw.set_window_focus_callback(window, self._focus_callback)
        
        
    def _window_should_close_callback(self, window,*args,**kwargs):
        self._needs_to_update=False
        #print(
        #    '_window_should_close_callback',window,*args,**kwargs
        #)
        #glfw.set_window_should_close(window, should)
        self._should_terminate=True

    
        
    def _run(self):
        self._should_terminate=False
        ftime=1/60
        
        while not  self._should_terminate:
            
            try:
                
                if self._should_terminate:
                    print('terminating')
                    #self._needs_to_update=False
                    break
                if glfw.window_should_close(self.window):
                    
                    print('window_should_close')
                    #self._needs_to_update = False
                    break
                glfw.poll_events()
                if  self._snap_mode and all((not self.projection is None, not self.last_mouse_pos is None)):
                    
                    
                    
                 
                    snap = self.perform_snap()

                        #glfw.set_cursor_pos(self.window, snap[0],snap[1])


                self.render()
                
             
                

                glfw.swap_buffers(self.window)
                
                glfw.wait_events()
                
            except KeyboardInterrupt:
                print('KeyboardInterrupt')
                #self._needs_to_update = False
             
                break
        #print('breaking')
        #glfw.destroy_window(self.window)
    def close(self):
        #print('clos')
       
        self._needs_to_update = False
        self._should_terminate = True
        
        
        #glfw.destroy_window(self.window)
    def destroy(self):
        #print('destroy')
        self._needs_to_update = False
        self._should_terminate = True
        glfw.destroy_window(self.window)
    
        del self.window
        self.window = None
    
        
    def terminate(self):
        #for i in range(4):
        #    if not self._run_thread.is_alive():
        #        break
        self._needs_to_update=False
        self._should_terminate = True
        glfw.terminate()
        
        
        
                
    def run(self):
        self._should_terminate = False
        self.create_window()
        """Main application loop"""
        self._run()
        self.destroy()


    def add_nurbs_curve(self, crv: NURBSCurve|NURBSCurveTuple, color=(0.8, 0.8, 0.8), thickness=1.0, **kwargs):
       
        if isinstance(crv, NURBSCurve):
            crv = _nurbs_to_tuple(crv)
        from mmcore.geom._nurbs_knots import decompose_curve
        
        for d in decompose_curve(crv):
            hpoints = np.c_[d.control_points,d.weights[..., np.newaxis]]
            self._hcurves.append(hpoints)
        #print(crv)
        params,du_list,evals,s_list=adaptive_curve_sampler(crv,self.sampling_tol,max_param_step_fraction=24)
        res = np.array([i['C']for i in evals], dtype=np.float32)
        # print(res)
        
        self.add_wire(np.asarray(res, dtype=np.float32), color=np.array(color, dtype=np.float32), thickness=thickness)  # Green
        

    def add_nurbs_surface_mesh(self, surf: NURBSSurface|NURBSSurfaceTuple, color=(0.5, 0.5, 0.5, 0.5), wireframe_color=(0.0, 0.0, 0.0)):
        """Add a NURBS surface as a transparent mesh with wireframe"""
        # Tessellate the surface
        tessellation = surface_to_mesh(surf,  0.01)

        # Extract mesh data
        
        vertices = tessellation["position"]
        triangles = tessellation["faces"]

        # Add mesh to the scene
        self.add_mesh(vertices, triangles, color=color, wireframe_color=wireframe_color)

        return tessellation
    
    def add_nurbs_surface(
        self,
        surf: NURBSSurface|NURBSSurfaceTuple,
        color=(1.0, 1.0, 1.0),
        thickness=1.0,
        render_as_mesh=True,
        surface_color=(0.5, 0.5, 0.9, 0.05),
        draw_isolies: bool = True,
    ):
        """Add a NURBS surface to the scene

        Args:
            surf: The NURBS surface to add
            color: Color for wireframe curves
            thickness: Thickness for wireframe curves
            render_as_mesh: Whether to render as transparent mesh (default: True)
            surface_color: Color for surface mesh (RGBA with alpha) if render_as_mesh is True
        """
        # Add wireframe representation
        if isinstance(surf, NURBSSurfaceTuple):
            surf=_tuple_to_nurbs(surf)
        boundaries, isolines, mid_iso = nurbs_surface_wireframe_view(surf)
        if draw_isolies:
            for iso in isolines:
                self.add_nurbs_curve(iso, (np.array(color[:3]) * 0.5).tolist(), thickness)
        for b in boundaries:
            self.add_nurbs_curve(b, color[:3], thickness)

        # If requested, add mesh representation
        if render_as_mesh:
            self.add_nurbs_surface_mesh(surf, color=surface_color, wireframe_color=None)
    
  
        
        
    def add_nurbs(self, geom, *args, **kwargs):
        if isinstance(geom, (NURBSCurve,NURBSCurveTuple)):
            self.add_nurbs_curve(geom, *args, **kwargs)
        elif isinstance(geom, (NURBSSurface,NURBSSurfaceTuple)):
            self.add_nurbs_surface(geom, *args, **kwargs)
        else:
            raise ValueError(f"Unsupported geometry type: {type(geom).__name__}")
        
    def add_geometry(self, geometry, color=(1.0, 1.0, 1.0), thickness: float = 1.0, **kwargs):
        """Add geometry to the scene and update camera position

        Args:
            geometry: The geometry to add (NURBSCurve or NURBSSurface)
            color: Color for wireframe or curves
            thickness: Thickness for wireframe or curves
            **kwargs: Additional parameters:
                - render_as_mesh: Whether to render surfaces as transparent mesh (default: True)
                - surface_color: Color for surface mesh (RGBA with alpha) if render_as_mesh is True
                - u_count: Number of u divisions for surface tessellation
                - v_count: Number of v divisions for surface tessellation
        """
        dispatch = {
            NURBSCurve: self.add_nurbs_curve,
            NURBSSurface: self.add_nurbs_surface,
        }
        fun = dispatch.get(type(geometry))
        if fun is None:
            raise KeyError(f"{type(geometry).__name__} is not supported")
        else:
            if isinstance(geometry, NURBSSurface):
                # Pass additional parameters for surface rendering
                fun(geometry, color, thickness, **kwargs)
            else:
                fun(geometry, color, thickness)

        # Camera will be automatically updated by the lower-level methods

    def set_auto_camera_positioning(self, enabled=True):
        """Enable or disable automatic camera positioning"""
        self.auto_position_camera = enabled
        if enabled:
            # Update camera position based on current geometry
            self.update_camera_position()
    def add_scalar_bern(self, bern, color=(1.0, 1.0, 1.0), thickness=1.0,weights=None,interval=None,**kwargs):
        
            bern = _gen_cpts_to_display(np.squeeze(bern),interval=interval)
        
            if weights is not None:
                
                return self.add_nurbs(bern_to_nurbs_bezier(np.c_[bern*weights[...,np.newaxis],weights[...,np.newaxis]],rational=True,interval=interval), color, thickness,**kwargs)
            
            return self.add_nurbs(bern_to_nurbs_bezier(bern, rational=False,interval=interval), color=color,
                                 thickness=thickness, **kwargs)
        
    
    def add_bezier(self, bezier,color=(1.0, 1.0, 1.0), thickness=1.0, *, scalar=False,rational=False, interval:tuple=None,**kwargs):
        if scalar:
            if rational:
                return self.add_scalar_bern(bezier[...,0],weights=bezier[...,1],color=color,thickness=thickness,interval=interval,**kwargs)
            else:
                
                return self.add_scalar_bern(np.squeeze(bezier),color=color,thickness=thickness,interval=interval,**kwargs)
            
   
            
        return self.add_nurbs(      bern_to_nurbs_bezier(bezier, rational=rational,interval=interval), color, thickness,**kwargs)
     
        
     
        
    def add_rational_bezier(self, bezier, color=(1.0, 1.0, 1.0), thickness=1.0, **kwargs):
        self.add_bezier(bezier, rational=True, color=color, thickness=thickness, **kwargs)

import numpy as np


if __name__ == "__main__":
    # Example usage
    viewer = CADRenderer(background_color=DEFAULT_DARK_BACKGROUND_COLOR)
    from mmcore._test_data import ssx as ssx_data

    from mmcore.numeric.intersection.ssx import surface_ppi

    # Get the test surfaces
    s1, s2 = ssx_data[2]

    # Add the surfaces with transparency
    viewer.add_geometry(
        s1, color=(0.2, 0.2, 0.2), thickness=1.5, render_as_mesh=True, surface_color=(0.3, 0.7, 0.9, 0.5)
    )  # Blue transparent

    viewer.add_geometry(
        s2, color=(0.2, 0.2, 0.2), thickness=1.5, render_as_mesh=True, surface_color=(0.9, 0.5, 0.3, 0.5)
    )  # Orange transparent

    # Get the intersection curves
    cc = surface_ppi(*ssx_data[2])

    # Add intersection curves with white color
    for c in cc[0]:
        viewer.add_wire(np.array(c, np.float32), color=np.array((1.0, 1.0, 1.0), np.float32), thickness=2.0)

    # Run the viewer
    viewer.run()
