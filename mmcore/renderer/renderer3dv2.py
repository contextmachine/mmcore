import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
import pyrr
from dataclasses import dataclass,field
from typing import List, Tuple

from mmcore.geom.bvh import BoundingBox
from mmcore.geom.nurbs import NURBSCurve, NURBSSurface, decompose_surface, greville_abscissae
from mmcore.numeric.vectors import scalar_unit,gram_schmidt

from mmcore.numeric.intersection.ssx.boundary_intersection import extract_isocurve

from mmcore.numeric.intersection.ssx.boundary_intersection import (
    extract_surface_boundaries,
    find_boundary_intersections,
    sort_boundary_intersections,
    IntersectionPoint
)

DEFAULT_BACKGROUND_COLOR = 158 / 256, 162 / 256, 169 / 256, 1.
DEFAULT_DARK_BACKGROUND_COLOR = 0.05, 0.05, 0.05, 1.


@dataclass
class Point:
    position: np.ndarray  # 3D vector
    color: np.ndarray  # RGB vector
    size: float


@dataclass
class Wire:
    vertices: np.ndarray  # Nx3 array of vertices
    color: np.ndarray  # RGB vector
    thickness: float


def nurbs_surface_wireframe_view(surf: NURBSSurface):
    (u_min, u_max), (v_min, v_max) = surf.interval()

    u_iso = extract_isocurve(surf, (u_min + u_max) * 0.5, direction='u')
    v_iso = extract_isocurve(surf, (v_min + v_max) * 0.5, direction='v')
    boundaries = extract_surface_boundaries(surf)
    return boundaries, (u_iso, v_iso)
from numpy.typing import NDArray
@dataclass
class BoundingSphere:
    origin:field(default_factory=lambda : np.array([0.,0.,0.], dtype=np.float32))
    radius:float = 0.
@dataclass
class Camera:
    pos:NDArray[np.float32]=field(default_factory=lambda : np.array([150.0,150.0, 150.0], dtype=np.float32))
    target: NDArray[np.float32]=field(default_factory=lambda : np.array([0.0,0.0, 0.0], dtype=np.float32))
    up: NDArray[np.float32]=field(default_factory=lambda : np.array([0.0, 1.0, 0.0], dtype=np.float32))
    zoom:float=1.
    near:float = 0.01
    far:float = 1000000.0
    is_panning:bool = False
import multiprocessing as mp
class CADRenderer:
    def __init__(self, width=800, height=600, background_color=DEFAULT_DARK_BACKGROUND_COLOR, camera:Camera=None):

        # Initialize window
        self._background_color = background_color
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        if camera is None:
            camera=Camera()
        self.bsf=BoundingSphere(camera.target,0.)
        # Configure GLFW for macOS compatibility
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.COCOA_RETINA_FRAMEBUFFER, True)

        # Create window
        self.window = glfw.create_window(width, height, "CAD Viewer", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)

        # Print OpenGL version info
        print("OpenGL version:", glGetString(GL_VERSION).decode())
        print("GLSL version:", glGetString(GL_SHADING_LANGUAGE_VERSION).decode())

        # Camera settings
        if camera is None:
            camera=Camera()
        self.camera_pos = camera.pos
        self.camera_target = camera.target
        self.camera_up = camera.up
        self.zoom =camera.zoom
        self.is_panning = camera.is_panning
        self.near = camera.near
        self.far = camera.far

        # Mouse interaction
        self.is_dragging = False
        self.last_mouse_pos = np.array([0.0, 0.0])
        self.snap_distance = 0.1

        # Geometry storage
        self.points: List[Point] = []
        self.wires: List[Wire] = []

        # Setup callbacks
        self.setup_callbacks()

        # Initialize shaders
        self.setup_shaders()

        # Enable depth testing
        glEnable(GL_DEPTH_TEST)

        # Create and bind a default VAO
        self.default_vao = glGenVertexArrays(1)
        glBindVertexArray(self.default_vao)
        # For macOS Retina displays
        self.framebuffer_size = glfw.get_framebuffer_size(self.window)
        glViewport(0, 0, self.framebuffer_size[0], self.framebuffer_size[1])

    def setup_callbacks(self):
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self._mouse_move_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)
        glfw.set_framebuffer_size_callback(self.window, self._framebuffer_size_callback)

    def _framebuffer_size_callback(self, window, width, height):
        glViewport(0, 0, width, height)
        self.framebuffer_size = (width, height)

    def _mouse_button_callback(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            # Check if CMD (Control on macOS) is pressed
            if mods & glfw.MOD_SHIFT:
                print("Left click + SHIFT")
                self.is_panning = action == glfw.PRESS
            else:
                self.is_dragging = action == glfw.PRESS
        if button == glfw.MOUSE_BUTTON_RIGHT:
            print("Right click")
            self.is_panning = action == glfw.PRESS

        if self.is_dragging or self.is_panning:
            x, y = glfw.get_cursor_pos(window)
            # Scale cursor position for Retina displays
            fb_width, fb_height = self.framebuffer_size
            win_width, win_height = glfw.get_window_size(window)
            x *= fb_width / win_width
            y *= fb_height / win_height
            self.last_mouse_pos = np.array([x, y])

    def setup_shaders(self):
        # macOS compatible vertex shader
        vertex_shader_source = """
          #version 410
          layout (location = 0) in vec3 position;
          layout (location = 1) in vec3 color;
          uniform mat4 model;
          uniform mat4 view;
          uniform mat4 projection;
          out vec3 vertex_color;
          void main() {
              gl_Position = projection * view * model * vec4(position, 1.0);
              vertex_color = color;
          }
          """

        # macOS compatible fragment shader
        fragment_shader_source = """
          #version 410
          in vec3 vertex_color;
          out vec4 FragColor;
          void main() {
              FragColor = vec4(vertex_color, 1.0);
          }
          """

        try:
            # Compile shaders
            vertex_shader = shaders.compileShader(vertex_shader_source, GL_VERTEX_SHADER)
            fragment_shader = shaders.compileShader(fragment_shader_source, GL_FRAGMENT_SHADER)

            # Create program and attach shaders
            self.shader_program = glCreateProgram()
            glAttachShader(self.shader_program, vertex_shader)
            glAttachShader(self.shader_program, fragment_shader)

            # Link program
            glLinkProgram(self.shader_program)

            # Check for linking errors
            if not glGetProgramiv(self.shader_program, GL_LINK_STATUS):
                info_log = glGetProgramInfoLog(self.shader_program)
                raise RuntimeError(f"Error linking program: {info_log}")

            # Clean up shaders
            glDeleteShader(vertex_shader)
            glDeleteShader(fragment_shader)

            # Create a dummy VAO for validation
            dummy_vao = glGenVertexArrays(1)
            glBindVertexArray(dummy_vao)

            # Now validate the program
            glValidateProgram(self.shader_program)
            if not glGetProgramiv(self.shader_program, GL_VALIDATE_STATUS):
                info_log = glGetProgramInfoLog(self.shader_program)
                print(f"Warning - Program validation: {info_log}")

            # Clean up dummy VAO
            glBindVertexArray(0)
            glDeleteVertexArrays(1, [dummy_vao])

        except Exception as e:
            print(f"Shader setup error: {e}")
            raise


    def add_point(self, position: np.ndarray, color: np.ndarray = np.array([1.0, 1.0, 1.0]), size: float = 5.0):
        """Add a point to the scene"""
        self.points.append(Point(position, color, size))
        self.camera_target=self.bsf.origin=(self.camera_target+position)/2

        nrms = np.linalg.norm([p.position -self.camera_target for p in self.points - self.camera_target], axis=1)
        i = np.argmax(nrms)
        if nrms[i] > self.bsf.radius:
            self.bsf.radius=nrms[i]
        view_vec = scalar_unit(np.array(self.camera_pos - self.camera_target,dtype=float))

        self.camera_pos = np.asarray(self.camera_target + view_vec *     self.bsf.radius * 2,dtype=np.float32)

    def add_wire(self, vertices: np.ndarray, color: np.ndarray = np.array([1.0, 1.0, 1.0]), thickness: float = 1.0):
        """Add a wire (curve) to the scene"""
        self.wires.append(Wire(vertices, color, thickness))
        self.camera_target = np.average([self.camera_target,vertices[0],vertices[1]], axis=0)

        self.camera_target = self.bsf.origin=np.average([self.camera_target,vertices[0],vertices[1]], axis=0)
        p = []
        nrms=np.linalg.norm(
            [(w.vertices[1]+w.vertices[0])/2 - self.camera_target for w in self.wires], axis=1)



        i = np.argmax(nrms)
        if nrms[i] > self.bsf.radius:
            self.bsf.radius = nrms[i]
        view_vec = scalar_unit(np.array(self.camera_pos - self.camera_target, dtype=float))

        self.camera_pos = np.asarray(self.camera_target +view_vec * self.bsf.radius * 2, dtype=np.float32)


    def _mouse_move_callback(self, window, x, y):
        # Scale cursor position for Retina displays
        fb_width, fb_height = self.framebuffer_size
        win_width, win_height = glfw.get_window_size(window)
        x *= fb_width / win_width
        y *= fb_height / win_height

        current_pos = np.array([x, y])

        if self.is_dragging or self.is_panning:
            delta = current_pos - self.last_mouse_pos

            if self.is_panning:
                # Pan the camera
                # Convert screen delta to world space delta
                aspect = fb_width / fb_height
                world_delta_x = (delta[0] / fb_width) * self.zoom * 2 * aspect
                world_delta_y = -(delta[1] / fb_height) * self.zoom * 2

                # Move camera and target together to pan
                pan_vector = (
                        self.camera_right * world_delta_x +
                        self.camera_up * world_delta_y
                )
                self.camera_pos -= pan_vector
                self.camera_target -= pan_vector

            elif self.is_dragging:
                # Rotate camera around target
                sensitivity = 0.005
                rotation_x = pyrr.matrix44.create_from_y_rotation(delta[0] * sensitivity)
                rotation_y = pyrr.matrix44.create_from_x_rotation(delta[1] * sensitivity)

                # Apply rotations
                camera_to_target = self.camera_pos - self.camera_target
                camera_to_target = np.dot(rotation_x, np.append(camera_to_target, 1.0))[:3]
                camera_to_target = np.dot(rotation_y, np.append(camera_to_target, 1.0))[:3]
                self.camera_pos = self.camera_target + camera_to_target

            self.last_mouse_pos = current_pos

    def _scroll_callback(self, window, xoffset, yoffset):
        # Modify zoom for orthographic projection
        zoom_factor = 0.1
        self.zoom *= (1.0 - yoffset * zoom_factor)
        self.zoom = np.clip(self.zoom, self.near, self.far)

    @property
    def camera_right(self):
        # Get the camera's right vector
        forward = self.camera_target - self.camera_pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, self.camera_up)

        return right / np.linalg.norm(right)

    def render(self):
        """Main render function"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(*self._background_color)

        # Get current framebuffer size
        width, height = self.framebuffer_size
        aspect = width / height

        # Create orthographic projection matrix
        # Note: zoom controls the visible area size
        self.projection = pyrr.matrix44.create_orthogonal_projection(
            left=-self.zoom * aspect,
            right=self.zoom * aspect,
            bottom=-self.zoom,
            top=self.zoom,
            near=self.near,
            far=self.far,
            dtype=np.float32
        )

        # Create view matrix
        self.view = pyrr.matrix44.create_look_at(
            self.camera_pos,
            self.camera_target,
            self.camera_up,
            dtype=np.float32
        )
        self.model = pyrr.matrix44.create_identity(dtype=np.float32)

        # Use shader program and set uniforms
        glUseProgram(self.shader_program)

        # Set matrices in shader
        glUniformMatrix4fv(
            glGetUniformLocation(self.shader_program, "projection"),
            1, GL_FALSE, self.projection
        )
        glUniformMatrix4fv(
            glGetUniformLocation(self.shader_program, "view"),
            1, GL_FALSE, self.view
        )
        glUniformMatrix4fv(
            glGetUniformLocation(self.shader_program, "model"),
            1, GL_FALSE, self.model
        )
        if len(self.points)>100:
            # Render points
            with mp.Pool(8) as pool:
                pool.map(self.render_point, self.points)
        else:
            [self.render_point(p) for p in self.points]
        if len(self.wires) > 100:
            with mp.Pool(8) as pool:
                pool.map(self.render_point, self.points)
        else:
            # Render wires
            for wire in self.wires:
                self.render_wire(wire)

    def render_point(self, points: Point):
        """Render a single point"""
        glPointSize(point.size * 2)  # Multiply by 2 for Retina displays

        # Create and bind VAO
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)

        # Create and bind VBO for position
        position_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, position_vbo)
        glBufferData(GL_ARRAY_BUFFER, point.position.nbytes, point.position, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        # Create and bind VBO for color
        color_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, color_vbo)
        glBufferData(GL_ARRAY_BUFFER, point.color.nbytes, point.color, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)

        # Draw point
        glDrawArrays(GL_POINTS, 0, 1)

        # Cleanup
        glDeleteBuffers(1, [position_vbo, color_vbo])
        glDeleteVertexArrays(1, [vao])

    def render_wire(self, wire: Wire):
        """Render a single wire"""
        glLineWidth(wire.thickness)  # Multiply by 2 for Retina displays

        # Create and bind VAO
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)


        # Create and bind VBO for vertices
        vertex_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_vbo)
        glBufferData(GL_ARRAY_BUFFER, wire.vertices.nbytes, wire.vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        # Create array of colors (one for each vertex)
        colors = np.tile(wire.color, (len(wire.vertices), 1))

        # Create and bind VBO for colors
        color_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, color_vbo)
        glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)

        # Draw wire
        glDrawArrays(GL_LINE_STRIP, 0, len(wire.vertices))

        # Cleanup
        glDeleteBuffers(1, [vertex_vbo, color_vbo])
        glDeleteVertexArrays(1, [vao])



    def run(self):
        """Main application loop"""
        while not glfw.window_should_close(self.window):
            self.render()
            glfw.swap_buffers(self.window)
            glfw.poll_events()

        glfw.terminate()

    def add_nurbs_curve(self, crv: NURBSCurve, color=(0., 1., 1.), thickness=1.0, **kwargs):
        res = np.array(
            crv.evaluate_multi(np.linspace(*crv.interval(), len(crv.knots) * 5)), dtype=np.float32)
        #print(res)
        self.add_wire(res,color=np.array(color, dtype=np.float32), thickness=thickness)  # Green


    def add_nurbs_surface(self, surf: NURBSSurface, color=(0., 0., 0.), thickness=1.0):
        boundaries, isolines = nurbs_surface_wireframe_view(surf)

        for iso in isolines:
            self.add_nurbs_curve(iso, (np.array(color) * 0.5).tolist(), thickness)
        for b in boundaries:
            self.add_nurbs_curve(b, color, thickness)

    def add_geometry(self, geometry, color=(1., 1., 1.), thickness: float = 1.0):
        dispatch = {

            NURBSCurve: self.add_nurbs_curve,
            NURBSSurface: self.add_nurbs_surface,
        }
        fun = dispatch.get(type(geometry))
        if fun is None:
            raise KeyError(f"{type(geometry).__name__} is not supported")
        else:
            fun(geometry, color, thickness)


if __name__ == "__main__":
    # Example usage
    viewer = CADRenderer(background_color=DEFAULT_DARK_BACKGROUND_COLOR)
    from mmcore._test_data import ssx as ssx_data

    from mmcore.numeric.intersection.ssx import surface_ppi

    # Add a point at origin

    np.average(np.array(ssx_data[2][0].control_points_flat))
    s1, s2 = ssx_data[2]

    cc = surface_ppi(*ssx_data[2])
    print(cc[0])
    for c in cc[0]:
        print('\nwire\n', c, '\n')
        viewer.add_wire(np.array(c, np.float32), color=np.array((1., 1., 1.), np.float32), thickness=1.)

    for i in ssx_data[2]:
        viewer.add_geometry(i, color=(0.6, 0.6, 0.6), thickness=1.)
    #
    ## Add a simple wire (triangle)
    # wire_vertices = np.array([
    #    [0.0, 0.0, 0.0],
    #    [1.0, 1.0, 0.0],
    #    [2.0, 0.0, 0.0]
    # ], dtype=np.float32)
    #
    # viewer.add_wire(
    #    vertices=wire_vertices,
    #    color=np.array([0.0, 1.0, 0.0], dtype=np.float32),  # Green
    #    thickness=1.0
    # )

    # Run the viewer
    viewer.run()