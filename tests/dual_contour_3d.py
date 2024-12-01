"""Provides a function for performing 3D Dual Countouring

# Both marching cube and dual contouring are adaptive, i.e. they select
# the vertex that best describes the underlying function. But for illustrative purposes
# you can turn this off, and simply select the midpoint vertex.
ADAPTIVE = True

# In dual contouring, if true, crudely force the selected vertex to belong in the cell
CLIP = False
# In dual contouring, if true, apply boundaries to the minimization process finding the vertex for each cell
BOUNDARY = True
# In dual contouring, if true, apply extra penalties to encourage the vertex to stay within the cell
BIAS = True
# Strength of the above bias, relative to 1.0 strength for the input gradients
BIAS_STRENGTH = 0.03

# Default bounds to evaluate over
XMIN = -2.
XMAX = 2.
YMIN = -2.
YMAX = 2.
ZMIN = -2.
ZMAX = 2.

# Size of single cell in grid
CELL_SIZE = 0.05

# Small value used to avoid floating point issues.
EPS = 1e-5
"""

import numpy as np

import numpy
import numpy.linalg
import math


class Edge:
    def __init__(self, v1, v2):
        self.v1 = v1
        self.v2 = v2

    def swap(self, swap=True):
        if swap:
            return Edge(self.v2, self.v1)
        else:
            return Edge(self.v1, self.v2)

def adapt(v0, v1, adaptive,cell_size):
    """v0 and v1 are numbers of opposite sign. This returns how far you need to interpolate from v0 to v1 to get to 0."""
    assert (v1 > 0) != (v0 > 0), "v0 and v1 do not have opposite sign"
    if adaptive:
        return (0 - v0) / (v1 - v0) * cell_size
    else:
        return 0.5 * cell_size

def frange(start, stop, step=1):
    """Like range, but works for floats"""
    v = start
    while v < stop:
        yield v
        v += step



def norm(x,y,z):
    return math.sqrt(x**2+y**2+z**2)
class V2:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def norm(self):
        return math.sqrt(self.x * self.x + self.y * self.y)
    def normalize(self):
        d = math.sqrt(self.x*self.x+self.y*self.y)
        return V2(self.x / d, self.y / d)

class V3:
    """A vector in 3D space"""
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    def norm(self):
        return math.sqrt(self.x*self.x+self.y*self.y+self.z*self.z)

    def normalize(self):
        d = self.norm()
        return V3(self.x / d, self.y / d, self.z / d)

    def astuple(self):
        return self.x, self.y, self.z
class Tri:
    """A 3d triangle"""
    def __init__(self, v1, v2, v3):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3

    def map(self, f):
        return Tri(f(self.v1), f(self.v2), f(self.v3))


class Quad:
    """A 3d quadrilateral (polygon with 4 vertices)"""
    def __init__(self, v1, v2, v3, v4):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.v4 = v4

    def map(self, f):
        return Quad(f(self.v1), f(self.v2), f(self.v3), f(self.v4))

    def swap(self, swap=True):
        if swap:
            return Quad(self.v4, self.v3, self.v2, self.v1)
        else:
            return Quad(self.v1, self.v2, self.v3, self.v4)


class Mesh:
    """A collection of vertices, and faces between those vertices."""
    def __init__(self, verts=None, faces=None):
        self.verts = verts or []
        self.faces = faces or []

    def extend(self, other):
        l = len(self.verts)
        f = lambda v: v + l
        self.verts.extend(other.verts)
        self.faces.extend(face.map(f) for face in other.faces)

    def __add__(self, other):
        r = Mesh()
        r.extend(self)
        r.extend(other)
        return r

    def translate(self, offset):
        new_verts = [V3(v.x + offset.x, v.y + offset.y, v.z + offset.z) for v in self.verts]
        return Mesh(new_verts, self.faces)


def make_obj(f, mesh):
    """Crude export to Wavefront mesh format"""
    for v in mesh.verts:
        f.write("v {} {} {}\n".format(v.x, v.y, v.z))
    for face in mesh.faces:
        if isinstance(face, Quad):
            f.write("f {} {} {} {}\n".format(face.v1, face.v2, face.v3, face.v4))
        if isinstance(face, Tri):
            f.write("f {} {} {}\n".format(face.v1, face.v2, face.v3))






class QEF:
    """Represents and solves the quadratic error function"""
    def __init__(self, A, b, fixed_values):
        self.A = A
        self.b = b
        self.fixed_values = fixed_values

    def evaluate(self, x):
        """Evaluates the function at a given point.
        This is what the solve method is trying to minimize.
        NB: Doesn't work with fixed axes."""
        x = numpy.array(x)
        return numpy.linalg.norm(numpy.matmul(self.A, x) - self.b)

    def eval_with_pos(self, x):
        """Evaluates the QEF at a position, returning the same format solve does."""
        return self.evaluate(x), x

    @staticmethod
    def make_2d(positions, normals):
        """Returns a QEF that measures the the error from a bunch of normals, each emanating from given positions"""
        A = numpy.array(normals)
        b = [v[0] * n[0] + v[1] * n[1] for v, n in zip(positions, normals)]
        fixed_values = [None] * A.shape[1]
        return QEF(A, b, fixed_values)

    @staticmethod
    def make_3d(positions, normals):
        """Returns a QEF that measures the the error from a bunch of normals, each emanating from given positions"""
        A = numpy.array(normals)
        b = [v[0] * n[0] + v[1] * n[1] + v[2] * n[2] for v, n in zip(positions, normals)]
        fixed_values = [None] * A.shape[1]
        return QEF(A, b, fixed_values)

    def fix_axis(self, axis, value):
        """Returns a new QEF that gives the same values as the old one, only with the position along the given axis
        constrained to be value."""
        # Pre-evaluate the fixed axis, adjusting b
        b = self.b[:] - self.A[:, axis] * value
        # Remove that axis from a
        A = numpy.delete(self.A, axis, 1)
        fixed_values = self.fixed_values[:]
        fixed_values[axis] = value
        return QEF(A, b, fixed_values)

    def solve(self):
        """Finds the point that minimizes the error of this QEF,
        and returns a tuple of the error squared and the point itself"""
        #print(self.A,self.b)
        result, residual, rank, s = numpy.linalg.lstsq(self.A, self.b, rcond=None)
        if len(residual) == 0:
            residual = self.evaluate(result)
        else:
            residual = residual[0]
        # Result only contains the solution for the unfixed axis,
        # we need to add back all the ones we previously fixed.
        position = []
        i = 0
        for value in self.fixed_values:
            if value is None:
                position.append(result[i])
                i += 1
            else:
                position.append(value)
        return residual, position


def solve_qef_2d(x, y, positions, normals, cell_size, clip,boundary, bias,bias_strength):
    # The error term we are trying to minimize is sum( dot(x-v[i], n[i]) ^ 2)
    # This should be minimized over the unit square with top left point (x, y)

    # In other words, minimize || A * x - b || ^2 where A and b are a matrix and vector
    # derived from v and n
    # The heavy lifting is done by the QEF class, but this function includes some important
    # tricks to cope with edge cases

    # This is demonstration code and isn't optimized, there are many good C++ implementations
    # out there if you need speed.



    if bias:
        # Add extra normals that add extra error the further we go
        # from the cell, this encourages the final result to be
        # inside the cell
        # These normals are shorter than the input normals
        # as that makes the bias weaker,  we want them to only
        # really be important when the input is ambiguous

        # Take a simple average of positions as the point we will
        # pull towards.
        mass_point = numpy.mean(positions, axis=0)

        normals.append([bias_strength, 0])
        positions.append(mass_point)
        normals.append([0, bias_strength])
        positions.append(mass_point)

    qef = QEF.make_2d(positions, normals)

    residual, v = qef.solve()

    if boundary:
        def inside(r):
            return x <= r[1][0] <= x + cell_size and y <= r[1][1] <= y + cell_size

        # It's entirely possible that the best solution to the qef is not actually
        # inside the cell.
        if not inside((residual, v)):
            # If so, we constrain the the qef to the horizontal and vertical
            # lines bordering the cell, and find the best point of those
            r1 = qef.fix_axis(0, x + 0).solve()
            r2 = qef.fix_axis(0, x + cell_size).solve()
            r3 = qef.fix_axis(1, y + 0).solve()
            r4 = qef.fix_axis(1, y + cell_size).solve()

            rs = list(filter(inside, [r1, r2, r3, r4]))

            if len(rs) == 0:
                # It's still possible that those lines (which are infinite)
                # cause solutions outside the box. So finally, we evaluate which corner
                # of the cell looks best
                r1 = qef.eval_with_pos((x + 0, y + 0))
                r2 = qef.eval_with_pos((x + 0, y + cell_size))
                r3 = qef.eval_with_pos((x + cell_size, y + 0))
                r4 = qef.eval_with_pos((x + cell_size, y + cell_size))

                rs = list(filter(inside, [r1, r2, r3, r4]))

            # Pick the best of the available options
            residual, v = min(rs)

    if clip:
        # Crudely force v to be inside the cell
        v[0] = numpy.clip(v[0], x, x + cell_size)
        v[1] = numpy.clip(v[1], y, y + cell_size)

    return V2(v[0], v[1])


def solve_qef_3d(x, y, z, positions, normals, cell_size,clip, boundary, bias,bias_strength):
    # The error term we are trying to minimize is sum( dot(x-v[i], n[i]) ^ 2)
    # This should be minimized over the unit square with top left point (x, y)

    # In other words, minimize || A * x - b || ^2 where A and b are a matrix and vector
    # derived from v and n
    # The heavy lifting is done by the QEF class, but this function includes some important
    # tricks to cope with edge cases

    # This is demonstration code and isn't optimized, there are many good C++ implementations
    # out there if you need speed.



    if bias:
        # Add extra normals that add extra error the further we go
        # from the cell, this encourages the final result to be
        # inside the cell
        # These normals are shorter than the input normals
        # as that makes the bias weaker,  we want them to only
        # really be important when the input is ambiguous

        # Take a simple average of positions as the point we will
        # pull towards.
        mass_point = numpy.mean(positions, axis=0)

        normals.append([bias_strength, 0, 0])
        positions.append(mass_point)
        normals.append([0, bias_strength, 0])
        positions.append(mass_point)
        normals.append([0, 0, bias_strength])
        positions.append(mass_point)

    qef = QEF.make_3d(positions, normals)

    residual, v = qef.solve()

    if boundary:
        def inside(r):
            return x <= r[1][0] <= x + cell_size and y <= r[1][1] <= y + cell_size and z <= r[1][2] <= z + cell_size

        # It's entirely possible that the best solution to the qef is not actually
        # inside the cell.
        if not inside((residual, v)):
            # If so, we constrain the the qef to the 6
            # planes bordering the cell, and find the best point of those
            r1 = qef.fix_axis(0, x + 0).solve()
            r2 = qef.fix_axis(0, x + cell_size).solve()
            r3 = qef.fix_axis(1, y + 0).solve()
            r4 = qef.fix_axis(1, y + cell_size).solve()
            r5 = qef.fix_axis(2, z + 0).solve()
            r6 = qef.fix_axis(2, z + cell_size).solve()

            rs = list(filter(inside, [r1, r2, r3, r4, r5, r6]))

            if len(rs) == 0:
                # It's still possible that those planes (which are infinite)
                # cause solutions outside the box.
                # So now try the 12 lines bordering the cell
                r1  = qef.fix_axis(1, y + 0).fix_axis(0, x + 0).solve()
                r2  = qef.fix_axis(1, y + cell_size).fix_axis(0, x + 0).solve()
                r3  = qef.fix_axis(1, y + 0).fix_axis(0, x + cell_size).solve()
                r4  = qef.fix_axis(1, y + cell_size).fix_axis(0, x + cell_size).solve()
                r5  = qef.fix_axis(2, z + 0).fix_axis(0, x + 0).solve()
                r6  = qef.fix_axis(2, z + cell_size).fix_axis(0, x + 0).solve()
                r7  = qef.fix_axis(2, z + 0).fix_axis(0, x + cell_size).solve()
                r8  = qef.fix_axis(2, z + cell_size).fix_axis(0, x + cell_size).solve()
                r9  = qef.fix_axis(2, z + 0).fix_axis(1, y + 0).solve()
                r10 = qef.fix_axis(2, z + cell_size).fix_axis(1, y + 0).solve()
                r11 = qef.fix_axis(2, z + 0).fix_axis(1, y + cell_size).solve()
                r12 = qef.fix_axis(2, z + cell_size).fix_axis(1, y + cell_size).solve()

                rs = list(filter(inside, [r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12]))

            if len(rs) == 0:
                # So finally, we evaluate which corner
                # of the cell looks best
                r1 = qef.eval_with_pos((x + 0, y + 0, z + 0))
                r2 = qef.eval_with_pos((x + 0, y + 0, z + cell_size))
                r3 = qef.eval_with_pos((x + 0, y + cell_size, z + 0))
                r4 = qef.eval_with_pos((x + 0, y + cell_size, z + cell_size))
                r5 = qef.eval_with_pos((x + cell_size, y + 0, z + 0))
                r6 = qef.eval_with_pos((x + cell_size, y + 0, z + cell_size))
                r7 = qef.eval_with_pos((x + cell_size, y + cell_size, z + 0))
                r8 = qef.eval_with_pos((x + cell_size, y + cell_size, z + cell_size))

                rs = list(filter(inside, [r1, r2, r3, r4, r5, r6, r7, r8]))

            # Pick the best of the available options
            residual, v = min(rs)

    if clip:
        # Crudely force v to be inside the cell
        v[0] = numpy.clip(v[0], x, x + cell_size)
        v[1] = numpy.clip(v[1], y, y + cell_size)
        v[2] = numpy.clip(v[2], z, z + cell_size)

    return V3(v[0], v[1], v[2])

def dual_contour_3d_find_best_vertex(f, f_normal, x, y, z, cell_size, adaptive,clip,  boundary, bias,bias_strength):
    if not adaptive:
        return V3(x+0.5*cell_size, y+0.5*cell_size, z+0.5*cell_size)

    # Evaluate f at each corner
    v = np.empty((2, 2, 2))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0,1):
                v[dx, dy, dz] = f(x + dx * cell_size, y + dy * cell_size, z + dz * cell_size)

    # For each edge, identify where there is a sign change.
    # There are 4 edges along each of the three axes
    changes = []
    for dx in (0, 1):
        for dy in (0, 1):
            if (v[dx, dy, 0] > 0) != (v[dx, dy, 1] > 0):
                changes.append((x + dx * cell_size,
                                y + dy * cell_size,
                                z + adapt(v[dx, dy, 0],v[dx, dy, 1], adaptive, cell_size)))

    for dx in (0, 1):
        for dz in (0, 1):
            if (v[dx, 0, dz] > 0) != (v[dx, 1, dz] > 0):
                changes.append((x + dx * cell_size,
                                y + adapt(v[dx, 0, dz], v[dx, 1, dz],adaptive, cell_size),
                                z + dz * cell_size))

    for dy in (0, 1):
        for dz in (0, 1):
            if (v[0, dy, dz] > 0) != (v[1, dy, dz] > 0):
                changes.append((x + adapt(v[0, dy, dz], v[1, dy, dz],adaptive, cell_size),
                                y + dy * cell_size,
                                z + dz * cell_size))

    if len(changes) <= 1:
        return None

    # For each sign change location v[i], we find the normal n[i].
    # The error term we are trying to minimize is sum( dot(x-v[i], n[i]) ^ 2)

    # In other words, minimize || A * x - b || ^2 where A and b are a matrix and vector
    # derived from v and n

    normals = []
    for v in changes:
        n = f_normal(v[0], v[1], v[2])
        normals.append([n.x, n.y, n.z])

    return solve_qef_3d(x, y, z, changes, normals, cell_size, clip,  boundary, bias,bias_strength)



def dual_contour_3d(f, f_normal, xmin, xmax, ymin, ymax, zmin, zmax,cell_size, adaptive, clip,  boundary, bias,bias_strength):
    """Iterates over a cells of size one between the specified range, and evaluates f and f_normal to produce
        a boundary by Dual Contouring. Returns a Mesh object."""
    # For each cell, find the the best vertex for fitting f
    vert_array = []
    vert_indices = {}
    if f_normal is None:
        f_normal=normal_from_function(f)
    for ix, x in enumerate(frange(xmin, xmax, cell_size)):
        for iy, y in enumerate(frange(ymin, ymax, cell_size)):
            for iz, z in enumerate(frange(zmin, zmax, cell_size)):
                vert = dual_contour_3d_find_best_vertex(f, f_normal, x, y, z,cell_size, adaptive, clip, boundary, bias, bias_strength)
                if vert is None:
                    continue
                vert_array.append(vert)
                vert_indices[ix, iy, iz] = len(vert_array)

    # For each cell edge, emit an face between the center of the adjacent cells if it is a sign changing edge
    faces = []
    for ix, x in enumerate(frange(xmin, xmax, cell_size)):
        for iy, y in enumerate(frange(ymin, ymax, cell_size)):
            for iz, z in enumerate(frange(zmin, zmax, cell_size)):
                if x > xmin and y > ymin:
                    solid1 = f(x, y, z + 0) > 0
                    solid2 = f(x, y, z + cell_size) > 0
                    if solid1 != solid2:
                        faces.append(Quad(
                            vert_indices[(ix - 1, iy - 1, iz)],
                            vert_indices[(ix - 0, iy - 1, iz)],
                            vert_indices[(ix - 0, iy - 0, iz)],
                            vert_indices[(ix - 1, iy - 0, iz)],
                        ).swap(solid2))
                if x > xmin and z > zmin:
                    solid1 = f(x, y + 0, z) > 0
                    solid2 = f(x, y + cell_size, z) > 0
                    if solid1 != solid2:
                        faces.append(Quad(
                            vert_indices[(ix - 1, iy, iz - 1)],
                            vert_indices[(ix - 0, iy, iz - 1)],
                            vert_indices[(ix - 0, iy, iz - 0)],
                            vert_indices[(ix - 1, iy, iz - 0)],
                        ).swap(solid1))
                if y > ymin and z > zmin:
                    solid1 = f(x + 0, y, z) > 0
                    solid2 = f(x + cell_size, y, z) > 0
                    if solid1 != solid2:
                        faces.append(Quad(
                            vert_indices[(ix, iy - 1, iz - 1)],
                            vert_indices[(ix, iy - 0, iz - 1)],
                            vert_indices[(ix, iy - 0, iz - 0)],
                            vert_indices[(ix, iy - 1, iz - 0)],
                        ).swap(solid2))

    return Mesh(vert_array, faces)





class Rosen:
    def __init__(self, a, b):
        self.a = a
        self.b = b


    def __call__(self, x: float, y: float, z: float) -> float:

        return z - (((self.a - x) ** 2 ) + self.b * ((y - (x ** 2)) **2) )


class Sphere:
    def __init__(self, r=1, origin=V3(0, 0, 0)):
        self.origin =origin
        self.r=r
    def __call__(self, x, y, z):
        x0,y0,z0=self.origin.x,self.origin.y,self.origin.z

        return V3(x - x0, y - y0, z - y0).norm() - self.r
    def normal(self,x, y, z):
        return V3(self.origin.x-x,self.origin.y-y, z-self.origin.z-z).normalize()


class CylinderXY:
    def __init__(self, r=1,  origin=V3(0, 0, 0)):
        self.origin =origin
        self.r=r
    def __call__(self, x, y, z):
        x0,y0,z0=self.origin.x,self.origin.y,self.origin.z

        return V2(x - x0, y - y0).norm() - self.r
    def normal(self,x, y, z):
        return V3(self.origin.x-x,self.origin.y-y, z).normalize()
class CylinderXZ:
    def __init__(self, r=1,  origin=V3(0, 0, 0)):
        self.origin =origin
        self.r=r
    def __call__(self, x, y, z):
        x0,y0,z0=self.origin.x,self.origin.y,self.origin.z

        return V2(x - x0,  z - z0).norm() - self.r
    def normal(self,x, y, z):
        return V3(self.origin.x-x,y,self.origin.z-z).normalize()


def invert(f):
    return lambda x,y,z: -f(x,y,z)
def sphere(x,y,z):

    return 2.5-  math.sqrt(x ** 2 + y ** 2 + z ** 2 )

def translate(fn,x,y,z):
    return lambda a,b,c: fn(x+a,y+b,z+c)

def translate_sphere(a,b,c):
    def sph(x,y,z):
        return a+x ** 2 + b+y ** 2 + c+z ** 2 - 2.5
    return sph
def union(*fns):
    return lambda x,y,z: np.sort(
        [fn(x,y,z) for fn in fns], 0)[0]

def intersect(*fns):
    return lambda x,y,z: sorted([fn(x,y,z) for fn in fns])[0]

def subtract(fn1, fn2):
    return intersect(fn1, lambda *args: -fn2(*args))


def circle_function(x, y, z):
    return math.sqrt(x*x + y*y + z*z)- 2.5


def circle_normal(x, y, z):
    l = math.sqrt(x*x + y*y + z*z)
    return V3(x / l, y / l, z / l)

def intersect_function(x, y, z):
    y -= 0.3
    x -= 0.5
    x = abs(x)
    return min(x - y, x + y)

def normal_from_function(f, d=0.0001):
    """Given a sufficiently smooth 3d function, f, returns a function approximating of the gradient of f.
    d controls the scale, smaller values are a more accurate approximation."""
    def norm(x, y, z):
        return V3(
            (f(x + d, y, z) - f(x - d, y, z)) / 2 / d,
            (f(x, y + d, z) - f(x, y - d, z)) / 2 / d,
            (f(x, y, z + d) - f(x, y, z - d)) / 2 / d
        ).normalize()
    return norm



if __name__ == "__main__":

    sph1=Sphere(2.5,V3(1.,1.,1.))
    sph2 = Sphere(2.5, V3(0.,0.,0))

    c1=CylinderXY(2.5,V3(0.,0.,0.))
    c2 = CylinderXZ(1.0, V3(0., 0., 0.))
    c3 = subtract(CylinderXY(1.5,V3(0.,0.,0.)), c1)
    tor = intersect(c3,  c2)
    mesh = dual_contour_3d( tor, normal_from_function(tor),   xmin=-3.0, xmax=3.0,
        ymin=-3.0, ymax=3.0,
        zmin=-3.0, zmax=3.0,
        cell_size=0.05,
        adaptive=True,
        clip=False,
        boundary=True,
        bias=True,
        bias_strength=0.03)
    with open("output5.obj", "w") as fs:
        make_obj(fs, mesh)
