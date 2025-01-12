import glfw
import numpy as np

from OpenGL.GL import shaders
import pyrr
from dataclasses import dataclass
from typing import List, Tuple, Any

from mmcore.geom.nurbs import NURBSCurve, NURBSSurface, decompose_surface, greville_abscissae


from mmcore.numeric.intersection.ssx.boundary_intersection import extract_isocurve

from mmcore.numeric.intersection.ssx.boundary_intersection import (
    extract_surface_boundaries,
    find_boundary_intersections,
    sort_boundary_intersections,
    IntersectionPoint
)
from mmcore.topo.mesh.tess import tessellate_surface

DEFAULT_BACKGROUND_COLOR = 158 / 256, 162 / 256, 169 / 256, 1.
DEFAULT_DARK_BACKGROUND_COLOR = 0.05, 0.05, 0.05, 1.

def nurbs_surface_wireframe_view(surf:NURBSSurface):
    (u_min,u_max),(v_min,v_max)=surf.interval()

    u_iso=extract_isocurve(surf,(u_min+u_max)*0.5,direction='u')
    v_iso=extract_isocurve(surf,(v_min+ v_max) * 0.5,direction='v')
    boundaries=extract_surface_boundaries(surf)
    return boundaries,(u_iso,v_iso)

@dataclass
class Point:
    position: np.ndarray  # 3D vector
    color: np.ndarray  # RGB vector
    size: float


@dataclass
class Wire:
    vertices: np.ndarray  # Nx3 array of vertices
    color: np.ndarray  # RGB vector
    thickness: float  # Thickness in world units


@dataclass
class Surface:
    vertices: np.ndarray  # Nx3 array of vertices
    normals: np.ndarray  # Nx3 array of normals
    color: np.ndarray  # RGBA vector
    material: dict  # Material properties
    vao: int  # Vertex Array Object
    vbo_vertices: int  # Vertex Buffer Object for vertices
    vbo_normals: int  # Vertex Buffer Object for normals
    owner:Any
@dataclass
class Light:
    direction: np.ndarray  # 3D vector
    color: np.ndarray  # RGB vector
    intensity: float


class CADRenderer:
    def __init__(self, width=800, height=600, background_color=DEFAULT_DARK_BACKGROUND_COLOR):
        # Initialize window
        self._background_color = background_color
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        # Configure GLFW for macOS compatibility and Geometry Shader support
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
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
        self.camera_pos = np.array([150.0, 150.0, 150.0], dtype=np.float32)
        self.camera_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.zoom = 1.0  # Controls orthographic size
        self.is_panning = False
        self.near = 0.1
        self.far = 100000.0

        # Mouse interaction
        self.is_dragging = False
        self.last_mouse_pos = np.array([0.0, 0.0])
        self.snap_distance = 0.1

        # Geometry storage
        self.points: List[Point] = []
        self.wires: List[Wire] = []
        self.surfaces: List[Surface] = []

        # Lighting
        self.ambient_light = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        self.main_light = Light(
            direction=np.array([1000.0, 1000.0, 1000.0], dtype=np.float32),
            color=np.array([0.8, 0.8, 0.8], dtype=np.float32),
            intensity=1
        )

        # Setup callbacks
        self.setup_callbacks()

        # Initialize shaders
        self.setup_shaders()

        # Enable depth testing and blending for transparency
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

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
            # Check if SHIFT is pressed for panning
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
        # Vertex shader for surfaces with lighting
        vertex_shader_source = """
        #version 410 core
        layout (location = 0) in vec3 position;
        layout (location = 1) in vec3 normal;
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        out vec3 frag_pos;
        out vec3 frag_normal;
        void main() {
            frag_pos = vec3(model * vec4(position, 1.0));
            frag_normal = normal;
            gl_Position = projection * view * vec4(frag_pos, 1.0);
        }
        """

        # Fragment shader for surfaces with lighting and material properties
        fragment_shader_source = """
        #version 410 core
        struct Material {
            vec3 ambient;
            vec3 diffuse;
            vec3 specular;
            float shininess;
        };

        struct Light {
            vec3 direction;
            vec3 color;
            float intensity;
        };

        in vec3 frag_pos;
        in vec3 frag_normal;
        out vec4 FragColor;

        uniform vec3 view_pos;
        uniform Material material;
        uniform Light main_light;
        uniform vec3 ambient_light;
        uniform float alpha;

        void main() {
            // Ambient
            vec3 ambient = ambient_light * material.ambient;

            // Diffuse
            vec3 norm = normalize(frag_normal);
            vec3 light_dir = normalize(-main_light.direction);
            float diff = max(dot(norm, light_dir), 0.0);
            vec3 diffuse = main_light.intensity * diff * material.diffuse * main_light.color;

            // Specular
            vec3 view_dir = normalize(view_pos - frag_pos);
            vec3 reflect_dir = reflect(-light_dir, norm);
            float spec = pow(max(dot(view_dir, reflect_dir), 0.0), material.shininess);
            vec3 specular = main_light.intensity * spec * material.specular * main_light.color;

            vec3 result = ambient + diffuse + specular;
            FragColor = vec4(result, alpha);
        }
        """

        # Wireframe Vertex Shader
        wire_vertex_shader_source = """
        #version 410 core
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 color;

        out VS_OUT {
            vec3 color;
            vec3 position;
        } vs_out;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;

        void main()
        {
            vs_out.color = color;
            vs_out.position = vec3(model * vec4(position, 1.0));
            gl_Position = projection * view * vec4(vs_out.position, 1.0);
        }
        """

        # Wireframe Geometry Shader with declared projection and view
        wire_geometry_shader_source = """
        #version 410 core
        layout(lines) in;
        layout(triangle_strip, max_vertices = 4) out;

        in VS_OUT {
            vec3 color;
            vec3 position;
        } gs_in[];

        out GS_OUT {
            vec3 color;
        } gs_out;

        uniform float lineThickness; // Thickness in world units
        uniform mat4 projection;
        uniform mat4 view;

        void main()
        {
            vec3 p0 = gs_in[0].position;
            vec3 p1 = gs_in[1].position;

            // Calculate the direction of the line
            vec3 lineDir = normalize(p1 - p0);

            // Calculate a vector perpendicular to the line and the view direction
            vec3 up = vec3(0.0, 1.0, 0.0);
            vec3 perpendicular = normalize(cross(lineDir, up));

            // If lineDir is parallel to up, choose another perpendicular vector
            if(length(perpendicular) < 0.001)
            {
                up = vec3(1.0, 0.0, 0.0);
                perpendicular = normalize(cross(lineDir, up));
            }

            // Scale the perpendicular vector by half the thickness
            vec3 offset = perpendicular * (lineThickness / 2.0);

            // Define the four vertices of the quad
            vec3 v0 = p0 + offset;
            vec3 v1 = p0 - offset;
            vec3 v2 = p1 + offset;
            vec3 v3 = p1 - offset;

            // Emit the quad vertices
            gs_out.color = gs_in[0].color;
            gl_Position = projection * view * vec4(v0, 1.0);
            EmitVertex();

            gs_out.color = gs_in[0].color;
            gl_Position = projection * view * vec4(v1, 1.0);
            EmitVertex();

            gs_out.color = gs_in[1].color;
            gl_Position = projection * view * vec4(v2, 1.0);
            EmitVertex();

            gs_out.color = gs_in[1].color;
            gl_Position = projection * view * vec4(v3, 1.0);
            EmitVertex();

            EndPrimitive();
        }
        """

        # Wireframe Fragment Shader
        wire_fragment_shader_source = """
        #version 410 core
        in GS_OUT {
            vec3 color;
        } gs_out;

        out vec4 FragColor;

        void main()
        {
            FragColor = vec4(gs_out.color, 1.0);
        }
        """

        try:
            # Compile shaders for surface rendering
            vertex_shader = shaders.compileShader(vertex_shader_source, GL_VERTEX_SHADER)
            fragment_shader = shaders.compileShader(fragment_shader_source, GL_FRAGMENT_SHADER)

            self.surface_shader_program = glCreateProgram()
            glAttachShader(self.surface_shader_program, vertex_shader)
            glAttachShader(self.surface_shader_program, fragment_shader)
            glLinkProgram(self.surface_shader_program)

            # Check for linking errors
            if not glGetProgramiv(self.surface_shader_program, GL_LINK_STATUS):
                info_log = glGetProgramInfoLog(self.surface_shader_program)
                raise RuntimeError(f"Error linking surface shader program: {info_log.decode()}")

            glDeleteShader(vertex_shader)
            glDeleteShader(fragment_shader)

            # Compile shaders for wireframe rendering
            wire_vertex_shader = shaders.compileShader(wire_vertex_shader_source, GL_VERTEX_SHADER)
            wire_geometry_shader = shaders.compileShader(wire_geometry_shader_source, GL_GEOMETRY_SHADER)
            wire_fragment_shader = shaders.compileShader(wire_fragment_shader_source, GL_FRAGMENT_SHADER)

            self.wireframe_shader_program = glCreateProgram()
            glAttachShader(self.wireframe_shader_program, wire_vertex_shader)
            glAttachShader(self.wireframe_shader_program, wire_geometry_shader)
            glAttachShader(self.wireframe_shader_program, wire_fragment_shader)
            glLinkProgram(self.wireframe_shader_program)

            # Check for linking errors
            if not glGetProgramiv(self.wireframe_shader_program, GL_LINK_STATUS):
                info_log = glGetProgramInfoLog(self.wireframe_shader_program)
                raise RuntimeError(f"Error linking wireframe shader program: {info_log.decode()}")

            glDeleteShader(wire_vertex_shader)
            glDeleteShader(wire_geometry_shader)
            glDeleteShader(wire_fragment_shader)

        except Exception as e:
            print(f"Shader setup error: {e}")
            raise

    def add_point(self, position: np.ndarray, color: np.ndarray = np.array([1.0, 1.0, 1.0]), size: float = 5.0):
        """Add a point to the scene"""
        self.points.append(Point(position, color, size))

    def add_wire(self, vertices: np.ndarray, color: np.ndarray = np.array([1.0, 1.0, 1.0]), thickness: float = 0.05):
        """Add a wire (curve) to the scene with specified thickness"""
        self.wires.append(Wire(vertices, color, thickness))

    def add_surface(self, surface: NURBSSurface, color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
                    material: dict = None):
        """
        Add a NURBS surface to the scene with customizable color and material properties.
        :param surface: NURBSSurface instance
        :param color: RGBA tuple
        :param material: Dictionary with material properties
        :param resolution: Number of samples per parametric direction
        """
        if material is None:
            material = {
                'ambient': np.array([1.0, 1.0, 1.0], dtype=np.float32),
                'diffuse': np.array([1.0, 1.0, 1.0], dtype=np.float32),
                'specular': np.array([1.0, 1.0, 1.0], dtype=np.float32),
                'shininess': 32.0
            }

        # Sample the surface
        num_rows = len(np.unique(surface.knots_u)) * 5
        num_cols = len(np.unique(surface.knots_v)) * 5
        u_range,v_range=surface.interval()
        u_vals = np.linspace(*u_range, num_rows)
        v_vals = np.linspace(*v_range, num_cols)
        u_grid, v_grid = np.meshgrid( u_vals,v_vals,indexing='ij')

        params = np.stack([ u_grid,v_grid], axis=2)
        print(params.shape)
        vertices = []
        normals=[]
        print(num_rows,num_cols)
        for i in range(num_rows- 1):
            if i % 2 == 0:
                # Even row: left to right
                for j in range(num_cols):
                    vertices.append(surface.evaluate(params[i][j]))
                    vertices.append(surface.evaluate(params[i + 1][j]))
                    normals.append(surface.normal(params[i][j]))
                    normals.append(surface.normal(params[i + 1][j]))
            else:
                # Odd row: right to left
                for j in range(num_cols - 1, -1, -1):
                    vertices.append(surface.evaluate(params[i][j]))
                    vertices.append(surface.evaluate(params[i + 1][j]))
                    normals.append(surface.normal(params[i][j]))
                    normals.append(surface.normal(params[i + 1][j]))
            # Add degenerate triangles if not the last row
            if i < num_rows - 2:
                vertices.append(surface.evaluate(params[i + 1][num_cols - 1]))
                vertices.append(surface.evaluate(params[i + 1][0]))
                normals.append(surface.normal(params[i + 1][num_cols - 1]))
                normals.append(surface.normal(params[i + 1][0]))



        evaluated_points = np.array(vertices, dtype=np.float32)
        evaluated_normals = np.array(normals, dtype=np.float32)

        # Create VAO and VBOs
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)

        # VBO for vertices
        vbo_vertices = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices)
        glBufferData(GL_ARRAY_BUFFER, evaluated_points.nbytes, evaluated_points, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        # VBO for normals
        vbo_normals = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_normals)
        glBufferData(GL_ARRAY_BUFFER, evaluated_normals.nbytes, evaluated_normals, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)

        glBindVertexArray(0)

        # Store the surface
        self.surfaces.append(Surface(
            vertices=evaluated_points,
            normals=evaluated_normals,
            color=np.array(color, dtype=np.float32),
            material=material,
            vao=vao,
            vbo_vertices=vbo_vertices,
            vbo_normals=vbo_normals,
            owner=surface
        ))

    def add_nurbs_curve(self, crv: NURBSCurve, color=(0., 1., 1.), thickness=0.05, **kwargs):
        res = np.array(
            crv.evaluate_multi(np.linspace(*crv.interval(), len(crv.knots) * 5)), dtype=np.float32)
        self.add_wire(
            res,
            color=np.array(color, dtype=np.float32), thickness=thickness)  # Green

    def add_nurbs_surface_wireframe(self, surf: NURBSSurface, color=(0., 0., 0.), thickness=0.05):
        boundaries, isolines = nurbs_surface_wireframe_view(surf)

        for iso in isolines:
            self.add_nurbs_curve(iso, (np.array(color) * 0.5).tolist(), thickness)
        for b in boundaries:
            self.add_nurbs_curve(b, color, thickness)

    def add_geometry(self, geometry, color=(1., 1., 1., 1.0), thickness: float = 0.05):
        dispatch = {
            NURBSCurve: self.add_nurbs_curve,
            NURBSSurface: self.add_nurbs_surface,
        }
        fun = dispatch.get(type(geometry))
        if fun is None:
            raise KeyError(f"{type(geometry).__name__} is not supported")
        else:
            fun(geometry, color, thickness)

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
        projection = pyrr.matrix44.create_orthogonal_projection(
            left=-self.zoom * aspect,
            right=self.zoom * aspect,
            bottom=-self.zoom,
            top=self.zoom,
            near=self.near,
            far=self.far,
            dtype=np.float32
        )

        # Create view matrix
        view = pyrr.matrix44.create_look_at(
            self.camera_pos,
            self.camera_target,
            self.camera_up,
            dtype=np.float32
        )
        model = pyrr.matrix44.create_identity(dtype=np.float32)

        # Render surfaces with lighting
        self.render_surfaces(model, view, projection)

        # Render wireframes (edges) with thickness
        self.render_wireframes(model, view, projection)

        # Render points
        for point in self.points:
            self.render_point(point)

    def render_surfaces(self, model, view, projection):
        """Render all surfaces with lighting"""
        glUseProgram(self.surface_shader_program)

        # Set uniform matrices
        glUniformMatrix4fv(
            glGetUniformLocation(self.surface_shader_program, "projection"),
            1, GL_FALSE, projection
        )
        glUniformMatrix4fv(
            glGetUniformLocation(self.surface_shader_program, "view"),
            1, GL_FALSE, view
        )
        glUniformMatrix4fv(
            glGetUniformLocation(self.surface_shader_program, "model"),
            1, GL_FALSE, model
        )

        # Set lighting uniforms
        glUniform3fv(glGetUniformLocation(self.surface_shader_program, "view_pos"), 1, self.camera_pos)
        glUniform3fv(glGetUniformLocation(self.surface_shader_program, "ambient_light"), 1, self.ambient_light)
        glUniform3fv(glGetUniformLocation(self.surface_shader_program, "main_light.direction"), 1, self.main_light.direction)
        glUniform3fv(glGetUniformLocation(self.surface_shader_program, "main_light.color"), 1, self.main_light.color)
        glUniform1f(glGetUniformLocation(self.surface_shader_program, "main_light.intensity"), self.main_light.intensity)

        for surface in self.surfaces:
            # Set material properties
            glUniform3fv(glGetUniformLocation(self.surface_shader_program, "material.ambient"), 1, surface.material['ambient'])
            glUniform3fv(glGetUniformLocation(self.surface_shader_program, "material.diffuse"), 1, surface.material['diffuse'])
            glUniform3fv(glGetUniformLocation(self.surface_shader_program, "material.specular"), 1, surface.material['specular'])
            glUniform1f(glGetUniformLocation(self.surface_shader_program, "material.shininess"), surface.material['shininess'])
            glUniform1f(glGetUniformLocation(self.surface_shader_program, "alpha"), surface.color[3])

            # Bind VAO and draw
            glBindVertexArray(surface.vao)
            # Assuming the surface is rendered as a grid of triangles
            # You might need to adjust the drawing mode and count based on how you sampled the surface
            # Here, using GL_TRIANGLES for simplicity
            triangulate_result=tess=tessellate_surface(surface)
            glDrawArrays(GL_TRIANGLE_STRIP, 0, triangulate_result['position'])
            glDrawArrays(GL_INDEX_ARRAY, 0,triangulate_result['triangles'])
            glBindVertexArray(0)


        glUseProgram(0)

    def render_wireframes(self, model, view, projection):
        """Render wireframes (edges) over surfaces with thickness using Geometry Shader"""
        glUseProgram(self.wireframe_shader_program)

        # Set uniform matrices
        glUniformMatrix4fv(
            glGetUniformLocation(self.wireframe_shader_program, "projection"),
            1, GL_FALSE, projection
        )
        glUniformMatrix4fv(
            glGetUniformLocation(self.wireframe_shader_program, "view"),
            1, GL_FALSE, view
        )
        glUniformMatrix4fv(
            glGetUniformLocation(self.wireframe_shader_program, "model"),
            1, GL_FALSE, model
        )

        # Draw each wire
        for wire in self.wires:
            # Set line thickness for this wire
            glUniform1f(glGetUniformLocation(self.wireframe_shader_program, "lineThickness"), wire.thickness)

            # Create and bind VAO for wire
            vao = glGenVertexArrays(1)
            glBindVertexArray(vao)

            # VBO for wire vertices
            vbo_vertices = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices)
            glBufferData(GL_ARRAY_BUFFER, wire.vertices.nbytes, wire.vertices, GL_STATIC_DRAW)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
            glEnableVertexAttribArray(0)

            # VBO for wire colors
            vbo_colors = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo_colors)
            colors = np.tile(wire.color, (len(wire.vertices), 1)).astype(np.float32)
            glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors.flatten(), GL_STATIC_DRAW)
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
            glEnableVertexAttribArray(1)

            # Draw lines as GL_LINES (each pair of vertices forms a line)
            glDrawArrays(GL_LINE_STRIP, 0, len(wire.vertices))

            # Cleanup
            glDeleteBuffers(1, [vbo_vertices, vbo_colors])
            glDeleteVertexArrays(1, [vao])

        glUseProgram(0)

    def render_point(self, point: Point):
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

    def run(self):
        """Main application loop"""
        while not glfw.window_should_close(self.window):
            self.render()
            glfw.swap_buffers(self.window)
            glfw.poll_events()

        glfw.terminate()

    def add_nurbs_surface(self, surf: NURBSSurface, color=(0., 0., 0., 1.0), thickness=0.05, resolution=50):
        """Add a NURBS surface to the scene with customizable color and material properties"""
        self.add_surface(surf, color=color)
        self.add_nurbs_surface_wireframe(surf, color, thickness)
if __name__ == "__main__":
    # Example usage
    viewer = CADRenderer(background_color=DEFAULT_BACKGROUND_COLOR)
    from mmcore._test_data import ssx as ssx_data

    from mmcore.numeric.intersection.ssx import surface_ppi

    # Add intersection curves as wires
    cc = surface_ppi(*ssx_data[2])
    for c in cc[0]:
        viewer.add_wire(
            np.array(c, np.float32),
            color=np.array((1., 1., 0.1), np.float32),
            thickness=0.5  # Example thickness
        )

    # Add NURBS surfaces with material properties
    for surf in ssx_data[2]:
        material = {
            'ambient': np.array([1.5,1.5,1.5], dtype=np.float32),
            'diffuse': np.array([0.4, 0.4, 0.41], dtype=np.float32),
            'specular': np.array([0.05, 0.05, 0.05], dtype=np.float32),
            'shininess': 0.1
        }
        bnds,iso=nurbs_surface_wireframe_view(surf)
        for s in bnds:
            viewer.add_nurbs_curve(s,np.array((0.05, 0.05,0.05), np.float32),thickness=0.3 )
        for s in iso:
            viewer.add_nurbs_curve(s, np.array((0.05, 0.05,0.05), np.float32), thickness=0.14)
        viewer.add_surface(surf,color=(0.8, 0.8, 0.8,1),material=material)
        #viewer.add_nurbs_surface(surf, color=(0.6, 0.6, 0.6, 0.8), thickness=0.2,material=)

    # Run the viewer
    viewer.run()
