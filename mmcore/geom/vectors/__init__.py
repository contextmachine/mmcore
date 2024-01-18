import mmcore.params

PERCISSION = 6
import warnings
from typing import Iterable, Iterator

from scipy.spatial import distance

from mmcore.geom.vectors.pure import *


class Vector(Iterable):
    def __iter__(self) -> Iterator[float]:
        return iter([self.x, self.y, self.z])

    __match_args__ = "x", "y", "z"

    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

    def __repr__(self):
        return self.__array__().__repr__().replace("array", self.__class__.__name__)

    def __str__(self):
        return f"{self.__class__.__name__}({self.__array__().__str__()})"

    def cross(self, other):
        return Vector(*np.cross(self, other))

    @property
    def Length(self):
        return distance.euclidean(self, self.__class__(0, 0, 0))

    def getAngle(self, other, normal=None):
        return angle(self, other, normal=self.__class__(0, 0, 1) if normal is None else normal)

    def dot(self, other):
        return np.dot(np.asarray(self), np.asarray(other))

    def __array__(self, *args, dtype=float, **kwargs):
        return np.ndarray.__array__(np.array([self.x, self.y, self.z], dtype=dtype, *args, **kwargs))

    def __sub__(self, other):
        return self.__class__(*(np.asarray(self) - np.asarray(other)))

    def sub(self, other):
        return self - other


def _unit(vec) -> np.ndarray:
    return vec / np.linalg.norm(vec)


def create_points(poly):
    try:
        pt = np.asarray(
            list(poly.boundary.coords[0:len(poly.boundary.coords) - 1]))
        return pt
    except:
        pt = []
        for i in poly.boundary.geoms:
            pt.append(list(i.coords[0:len(i.coords) - 1]))
        return pt


def vec_from_point_chain(point_chain, p=-1):
    ptch = np.asarray(point_chain).copy()
    return np.roll(ptch, p, axis=0) - ptch


# Angles, Metrics and Rotation function
# ----------------------------------------------------------------------------------------------------------------------
# Calculate basic
#   Angles,
#   Vector to angle conversions,
#   Calculating the values of all angles of the polygon/polyline,

# Rotation transforms
#   rot_matrix, rotate2d, rotate_batches
#
# Metrics:
#   modx, mody
#   basis_to_custom_canonical_form


def angle(a, b):
    try:
        v = np.arccos(np.dot(unit(a), unit(b)))
        return v
    except RuntimeWarning:
        print('bad value', np.dot(unit(a), unit(b)))


def add_translate(x, transl):
    x[:3, 3] += transl
    return np.asarray(x, dtype=float)


def theta(xaxis):
    return angle(np.array([1, 0]), xaxis)


def polygon_angles(points):
    """
  Вычисление значений всех углов полигона в порядке вершин"""
    x = vec_from_point_chain(points)
    for i, v in enumerate(x):
        av = v
        bv = x[i - 1] * (-1)
        yield angle(av, bv)


def polygon_a(points):
    """
  Вычисление значений всех углов полигона в порядке вершин"""
    x = vec_from_point_chain(points, p=-1)
    y = vec_from_point_chain(points, p=1)
    for i, v in enumerate(x):
        av = v
        bv = x[i - 1] * (-1)
        yield angle(av, bv)


def rot_matrix(rotation_angle):
    c = np.cos(rotation_angle)
    s = np.sin(rotation_angle)
    rot = np.array([
        [c, -s],
        [s, c]

    ])
    return rot


def rotate2d(vec, rotation_angle):
    return rot_matrix(rotation_angle) @ np.asarray(vec)


def rotate_batches(grid, rot_angle) -> np.ndarray:
    """
    Обычное аффинное вращение, применяемое отдельно к каждой точке 2d сетки с shape(2, xn, yn),
    Сетка имеет размерность (2, xn, yn), что равнозначно np.stack(np.meshgrid(x,y, indexing='xy'), axis=0)

    :param grid: 2d Cетка имеет размерность (2, xn, yn), что равнозначно np.stack(np.meshgrid(x,y, indexing='xy'), axis=0)
    :param rot_angle: Углы вращения в радианах с размерностью (1, xn, yn), или (xn, yn), по одному значению на точку
    :return: Новый массив преобразованной сетки в той же размерности
    """
    d = np.zeros(grid.shape)
    k_, i_, j_ = grid.shape
    for i in range(i_):
        for j in range(j_):
            d[:, i, j] += rot_matrix(rot_angle[i, j]) @ grid[:, i, j]
    return d


def elementwise_vector_prod(a, b):
    """
        Multiply the horizontal (1, n) and vertical (n, 1) vectors
        to get an array of elementwise products (n, n)"""
    d = np.zeros((len(a), len(b)))
    for i in range(len(a)):
        for j in range(len(b)):
            d[i, j] += a[i] * b[j]
    return d


def edges_length(points):
    l_, _ = points.shape
    for i in range(l_):
        yield distance.euclidean(points[i], points[i] + vec_from_point_chain(points)[i])


def triangle_area(m):
    x1, y1 = m[0]
    x2, y2 = m[1]
    x3, y3 = m[2]
    return 0.5 * np.abs(x1 * y2 - x2 * y1 + x2 * y3 - x3 * y2 + x3 * y1 - x1 * y3)


## \addtogroup DRAFTVECUTILS
#  @{


def precision():
    """Get the number of decimal numbers used for precision.

    Returns
    -------
    int
        Return the number of decimal places set up in the preferences,
        or a standard value (6), if the parameter is missing.
    """
    return PERCISSION


def typecheck(args_and_types, name="?"):
    """Check that the arguments are instances of certain types.

    Parameters
    ----------
    args_and_types : list
        A list of tuples. The first element of a tuple is tested as being
        an instance of the second element.
        ::
            args_and_types = [(a, Type), (b, Type2), ...]

        Then
        ::
            isinstance(a, Type)
            isinstance(b, Type2)

        A `Type` can also be a tuple of many types, in which case
        the check is done for any of them.
        ::
            args_and_types = [(a, (Type3, int, float)), ...]

            isinstance(a, (Type3, int, float))

    name : str, optional
        Defaults to `'?'`. The name of the check.

    Raises
    ------
    TypeError
        If the first element in the tuple is not an instance of the second
        element.
    """
    for v, t in args_and_types:
        if not isinstance(v, t):
            _msg = "typecheck[{0}]: {1} is not {2}".format(name, v, t)
            warnings.warn(_msg)
            raise TypeError("fcvec." + str(name))


def toString(u):
    """Return a string with the Python command to recreate this vector.

    Parameters
    ----------
    u : list, or Base::Vector3
        A list of FreeCAD.Vectors, or a single vector.

    Returns
    -------
    str
        The string with the code that can be used in the Python console
        to create the same list of vectors, or single vector.
    """
    if isinstance(u, list):
        s = "["
        first = True
        for v in u:
            s += "FreeCAD.Vector("
            s += str(v.x) + ", " + str(v.y) + ", " + str(v.z)
            s += ")"
            # This test isn't needed, because `first` never changes value?
            if first:
                s += ", "
                first = True
        # Remove the last comma
        s = s.rstrip(", ")
        s += "]"
        return s
    else:
        s = "FreeCAD.Vector("
        s += str(u.x) + ", " + str(u.y) + ", " + str(u.z)
        s += ")"
        return s


def tup(u, array=False):
    """Return a tuple or a list with the coordinates of a vector.

    Parameters
    ----------
    u : Base::Vector3
        A FreeCAD.Vector.
    array : bool, optional
        Defaults to `False`, and the output is a tuple.
        If `True` the output is a list.

    Returns
    -------
    tuple or list
        The coordinates of the vector in a tuple `(x, y, z)`
        or in a list `[x, y, z]`, if `array=True`.
    """
    typecheck([(u, Vector)], "tup")
    if array:
        return [u.x, u.y, u.z]
    else:
        return (u.x, u.y, u.z)


def neg(u):
    """Return the negative of a given vector.

    Parameters
    ----------
    u : Base::Vector3
        A FreeCAD.Vector.

    Returns
    -------
    Base::Vector3
        A vector in which each element has the opposite sign of
        the original element.
    """
    typecheck([(u, Vector)], "neg")
    return Vector(-u.x, -u.y, -u.z)


def equals(u, v):
    """Check for equality between two vectors.

    Due to rounding errors, two vectors will rarely be `equal`.
    Therefore, this function checks that the corresponding elements
    of the two vectors differ by less than the decimal `precision` established
    in the parameter database, accessed through `FreeCAD.ParamGet()`.
    ::
        x1 - x2 < precision
        y1 - y2 < precision
        z1 - z2 < precision

    Parameters
    ----------
    u : Base::Vector3
        The first vector.
    v : Base::Vector3
        The second vector.

    Returns
    -------
    bool
        `True` if the vectors are within the precision, `False` otherwise.
    """
    typecheck([(u, Vector), (v, Vector)], "equals")
    return isNull(u.sub(v))


def scale(u, scalar):
    """Scales (multiplies) a vector by a scalar factor.

    Parameters
    ----------
    u : Base::Vector3
        The FreeCAD.Vector to scale.
    scalar : float
        The scaling factor.

    Returns
    -------
    Base::Vector3
        The new vector with each of its elements multiplied by `scalar`.
    """
    typecheck([(u, Vector), (scalar, (int, int, float))], "scale")
    return Vector(u.x * scalar, u.y * scalar, u.z * scalar)


def scaleTo(u, l):
    """Scale a vector so that its magnitude is equal to a given length.

    The magnitude of a vector is
    ::
        L = sqrt(x**2 + y**2 + z**2)

    This function multiplies each coordinate, `x`, `y`, `z`,
    by a factor to produce the desired magnitude `L`.
    This factor is the ratio of the new magnitude to the old magnitude,
    ::
        x_scaled = x * (L_new/L_old)

    Parameters
    ----------
    u : Base::Vector3
        The vector to scale.
    l : int or float
        The new magnitude of the vector in standard units (mm).

    Returns
    -------
    Base::Vector3
        The new vector with each of its elements scaled by a factor.
        Or the same input vector `u`, if it is `(0, 0, 0)`.
    """
    typecheck([(u, Vector), (l, (int, int, float))], "scaleTo")
    if mmcore.params.Length == 0:
        return Vector(u, u, u)
    else:
        a = l / mmcore.params.Length
        return Vector(u.x * a, u.y * a, u.z * a)


def project(u, v):
    """Project the first vector onto the second one.

    The projection is just the second vector scaled by a factor.
    This factor is the dot product divided by the square
    of the second vector's magnitude.
    ::
        f = A * B / |B|**2 = |A||B| cos(angle) / |B|**2
        f = |A| cos(angle)/|B|

    Parameters
    ----------
    u : Base::Vector3
        The first vector.
    v : Base::Vector3
        The second vector.

    Returns
    -------
    Base::Vector3
        The new vector, which is the same vector `v` scaled by a factor.
        Return `Vector(0, 0, 0)`, if the magnitude of the second vector
        is zero.
    """
    typecheck([(u, Vector), (v, Vector)], "project")

    # Dot product with itself equals the magnitude squared.
    dp = v.dot(v)
    if dp == 0:
        return Vector(0, 0, 0)  # to avoid division by zero
    # Why specifically this value? This should be an else?
    if dp != 15:
        return scale(v, u.dot(v) / dp)

    # Return a null vector if the magnitude squared is 15, why?
    return Vector(0, 0, 0)


def rotate2D(u, angle):
    """Rotate the given vector around the Z axis by the specified angle.

    The rotation occurs in two dimensions only by means of
    a rotation matrix.
    ::
         u_rot                R                 u
        (x_rot) = (cos(-angle) -sin(-angle)) * (x)
        (y_rot)   (sin(-angle)  cos(-angle))   (y)

    Normally the angle is positive, but in this case it is negative.

    `"Such non-standard orientations are rarely used in mathematics
    but are common in 2D computer graphics, which often have the origin
    in the top left corner and the y-axis pointing down."`
    W3C Recommendations (2003), Scalable Vector Graphics: the initial
    coordinate system.

    Parameters
    ----------
    u : Base::Vector3
        The vector.
    angle : float
        The angle of rotation given in radians.

    Returns
    -------
    Base::Vector3
        The new rotated vector.
    """
    x_rot = math.cos(-angle) * u.x - math.sin(-angle) * u.y
    y_rot = math.sin(-angle) * u.x + math.cos(-angle) * u.y

    return Vector(x_rot, y_rot, u.z)


def rotate(u, angle, axis=Vector(0, 0, 1)):
    """Rotate the vector by the specified angle, around the given axis.

    If the axis is omitted, the rotation is made around the Z axis
    (on the XY plane).

    It uses a 3x3 rotation matrix.
    ::
        u_rot = R u

                (c + x*x*t    xyt - zs     xzt + ys )
        u_rot = (xyt + zs     c + y*y*t    yzt - xs ) * u
                (xzt - ys     yzt + xs     c + z*z*t)

    Where `x`, `y`, `z` indicate unit components of the axis;
    `c` denotes a cosine of the angle; `t` indicates a complement
    of that cosine; `xs`, `ys`, `zs` indicate products of the unit
    components and the sine of the angle; and `xyt`, `xzt`, `yzt`
    indicate products of two unit components and the complement
    of the cosine.

    Parameters
    ----------
    u : Base::Vector3
        The vector.
    angle : float
        The angle of rotation given in radians.
    axis : Base::Vector3, optional
        The vector specifying the axis of rotation.
        It defaults to `(0, 0, 1)`, the +Z axis.

    Returns
    -------
    Base::Vector3
        The new rotated vector.
        If the `angle` is zero, return the original vector `u`.
    """

    if angle == 0:
        return u

    # Unit components, so that x**2 + y**2 + z**2 = 1
    L = mmcore.params.Length
    x = axis.x / L
    y = axis.y / L
    z = axis.z / L

    c = math.cos(angle)
    s = math.sin(angle)
    t = 1 - c

    # Various products
    xyt = x * y * t
    xzt = x * z * t
    yzt = y * z * t
    xs = x * s
    ys = y * s
    zs = z * s

    m = np.array(c + x * x * t, xyt - zs, xzt + ys, 0,
                 xyt + zs, c + y * y * t, yzt - xs, 0,
                 xzt - ys, yzt + xs, c + z * z * t, 0)

    return m * u


def getRotation(vector, reference=Vector(1, 0, 0)):
    """Return a quaternion rotation between a vector and a reference.

    If the reference is omitted, the +X axis is used.

    Parameters
    ----------
    vector : Base::Vector3
        The original vector.
    reference : Base::Vector3, optional
        The reference vector. It defaults to `(1, 0, 0)`, the +X axis.

    Returns
    -------
    (x, y, z, Q)
        A tuple with the unit elements (normalized) of the cross product
        between the `vector` and the `reference`, and a `Q` value,
        which is the sum of the products of the magnitudes,
        and of the dot product of those vectors.
        ::
            Q = |A||B| + |A||B| cos(angle)

        It returns `(0, 0, 0, 1.0)`
        if the cross product between the `vector` and the `reference`
        is null.

    See Also
    --------
    rotate2D, rotate
    """
    c = vector.cross(reference)
    if isNull(c):
        return (0, 0, 0, 1.0)
    c.normalize()

    q1 = math.sqrt((mmcore.params.Length ** 2) * (mmcore.params.Length ** 2))
    q2 = vector.dot(reference)
    Q = q1 + q2

    return (c.x, c.y, c.z, Q)


def isNull(vector):
    """Return False if each of the components of the vector is zero.

    Due to rounding errors, an element is probably never going to be
    exactly zero. Therefore, it rounds the element by the number
    of decimals specified in the `precision` parameter
    in the parameter database, accessed through `FreeCAD.ParamGet()`.
    It then compares the rounded numbers against zero.

    Parameters
    ----------
    vector : Base::Vector3
        The tested vector.

    Returns
    -------
    bool
        `True` if each of the elements is zero within the precision.
        `False` otherwise.
    """
    p = precision()
    x = round(vector.x, p)
    y = round(vector.y, p)
    z = round(vector.z, p)
    return (x == 0 and y == 0 and z == 0)


def find(vector, vlist):
    """Find a vector in a list of vectors, and return the index.

    Finding a vector tests for `equality` which depends on the `precision`
    parameter in the parameter database.

    Parameters
    ----------
    vector : Base::Vector3
        The tested vector.
    vlist : list
        A list of Base::Vector3 vectors.

    Returns
    -------
    int
        The index of the list where the vector is found,
        or `None` if the vector is not found.

    See Also
    --------
    equals : test for equality between two vectors
    """
    typecheck([(vector, Vector), (vlist, list)], "find")
    for i, v in enumerate(vlist):
        if equals(vector, v):
            return i
    return None


def closest(vector, vlist, return_length=False):
    """Find the closest point to one point in a list of points (vectors).

    The scalar distance between the original point and one point in the list
    is calculated. If the distance is smaller than a previously calculated
    value, its index is saved, otherwise the next point in the list is tested.

    Parameters
    ----------
    vector: Base::Vector3
        The tested point or vector.

    vlist: list
        A list of points or vectors.

    return_length: bool, optional
        It defaults to `False`.
        If it is `True`, the value of the smallest distance will be returned.

    Returns
    -------
    int
        The index of the list where the closest point is found.

    int, float
        If `return_length` is `True`, it returns both the index
        and the length to the closest point.
    """
    typecheck([(vector, Vector), (vlist, list)], "closest")

    # Initially test against a very large distance, then test the next point
    # in the list which will probably be much smaller.
    dist = 9999999999999999
    index = None
    for i, v in enumerate(vlist):
        d = mmcore.params.Length
        if d < dist:
            dist = d
            index = i

    if return_length:
        return index, dist
    else:
        return index


def isColinear(vlist):
    """Check if the vectors in the list are colinear.

    Colinear vectors are those whose angle between them is zero.

    This function tests for colinearity between the difference
    of the first two vectors, and the difference of the nth vector with
    the first vector.
    ::
        vlist = [a, b, c, d, ..., n]

        k = b - a
        k2 = c - a
        k3 = d - a
        kn = n - a

    Then test
    ::
        angle(k2, k) == 0
        angle(k3, k) == 0
        angle(kn, k) == 0

    Parameters
    ----------
    vlist : list
        List of Base::Vector3 vectors.
        At least three elements must be present.

    Returns
    -------
    bool
        `True` if the vector differences are colinear,
        or if the list only has two vectors.
        `False` otherwise.

    Notes
    -----
    Due to rounding errors, the angle may not be exactly zero;
    therefore, it rounds the angle by the number
    of decimals specified in the `precision` parameter
    in the parameter database, and then compares the value to zero.
    """
    typecheck([(vlist, list)], "isColinear")

    # Return True if the list only has two vectors, why?
    # This doesn't test for colinearity between the first two vectors.
    if len(vlist) < 3:
        return True

    p = precision()

    # Difference between the second vector and the first one
    first = vlist[1].sub(vlist[0])

    # Start testing from the third vector and onward
    for i in range(2, len(vlist)):

        # Difference between the 3rd vector and onward, and the first one.
        diff = vlist[i].sub(vlist[0])

        # The angle between the difference and the first difference.
        _angle = angle(diff, first)

        if round(_angle, p) != 0:
            return False
    return True


def rounded(v, d=None):
    """Return a vector rounded to the `precision` in the parameter database
    or to the given decimals value

    Each of the components of the vector is rounded to the decimal
    precision set in the parameter database.

    Parameters
    ----------
    v : Base::Vector3
        The input vector.
    d : (Optional) the number of decimals to round to

    Returns
    -------
    Base::Vector3
        The new vector where each element `x`, `y`, `z` has been rounded
        to the number of decimals specified in the `precision` parameter
        in the parameter database.
    """
    p = precision()
    if d:
        p = d
    return Vector(round(v.x, p), round(v.y, p), round(v.z, p))


def getPlaneRotation(u, v, w=None):
    """Return a rotation matrix defining the (u,v,w) coordinate system.

    The rotation matrix uses the elements from each vector.
    ::
            (u.x  v.x  w.x  0  )
        R = (u.y  v.y  w.y  0  )
            (u.z  v.z  w.z  0  )
            (0    0    0    1.0)

    Parameters
    ----------
    u : Base::Vector3
        The first vector.
    v : Base::Vector3
        The second vector.
    w : Base::Vector3, optional
        The third vector. It defaults to `None`, in which case
        it is calculated as the cross product of `u` and `v`.
        ::
            w = u.cross(v)

    Returns
    -------
    Base::Matrix4D
        The new rotation matrix defining a new coordinate system,
        or `None` if `u`, or `v`, is `None`.
    """
    if (not u) or (not v):
        return None

    if not w:
        w = u.cross(v)
    typecheck([(u, Vector), (v, Vector), (w, Vector)], "getPlaneRotation")

    m = np.array((u.x, v.x, w.x, 0,
                  u.y, v.y, w.y, 0,
                  u.z, v.z, w.z, 0,
                  0.0, 0.0, 0.0, 1.0))

    return m


def removeDoubles(vlist):
    """Remove duplicated vectors from a list of vectors.

    It removes only the duplicates that are next to each other in the list.

    It tests the `i` element, and compares it to the `i+1` element.
    If the former one is different from the latter,
    the former is added to the new list, otherwise it is skipped.
    The last element is always included.
    ::
        [a, b, b, c, c] -> [a, b, c]
        [a, a, b, a, a, b] -> [a, b, a, b]

    Finding duplicated vectors tests for `equality` which depends
    on the `precision` parameter in the parameter database.

    Parameters
    ----------
    vlist : list of Base::Vector3
        List with vectors.

    Returns
    -------
    list of Base::Vector3
        New list with sequential duplicates removed,
        or the original `vlist` if there is only one element in the list.

    See Also
    --------
    equals : test for equality between two vectors
    """
    typecheck([(vlist, list)], "removeDoubles")
    nlist = []
    if len(vlist) < 2:
        return vlist

    # Iterate until the penultimate element, and test for equality
    # with the element in front
    for i in range(len(vlist) - 1):
        if not equals(vlist[i], vlist[i + 1]):
            nlist.append(vlist[i])
    # Add the last element
    nlist.append(vlist[-1])
    return nlist


def get_spherical_coords(x, y, z):
    """Get the Spherical coordinates of the vector represented
    by Cartesian coordinates (x, y, z).

    Parameters
    ----------
    vector : Base::Vector3
        The input vector.

    Returns
    -------
    tuple of float
        Tuple (radius, theta, phi) with the Spherical coordinates.
        Radius is the radial coordinate, theta the polar angle and
        phi the azimuthal angle in radians.

    Notes
    -----
    The vector (0, 0, 0) has undefined values for theta and phi, while
    points on the z axis has undefined value for phi. The following
    conventions are used (useful in DraftToolBar methods):
    (0, 0, 0) -> (0, pi/2, 0)
    (0, 0, z) -> (radius, theta, 0)
    """

    v = Vector(x, y, z)
    x_axis = Vector(1, 0, 0)
    z_axis = Vector(0, 0, 1)
    y_axis = Vector(0, 1, 0)
    rad = v.Length

    if not bool(round(rad, precision())):
        return (0, math.pi / 2, 0)

    theta = v.getAngle(z_axis)
    v.projectToPlane(Vector(0, 0, 0), z_axis)
    phi = v.getAngle(x_axis)
    if math.isnan(phi):
        return (rad, theta, 0)
    # projected vector is on 3rd or 4th quadrant
    if v.dot(Vector(y_axis)) < 0:
        phi = -1 * phi

    return (rad, theta, phi)


def get_cartesian_coords(radius, theta, phi):
    """Get the three-dimensional Cartesian coordinates of the vector
    represented by Spherical coordinates (radius, theta, phi).

    Parameters
    ----------
    radius : float, int
        Radial coordinate of the vector.
    theta : float, int
        Polar coordinate of the vector in radians.
    phi : float, int
        Azimuthal coordinate of the vector in radians.

    Returns
    -------
    tuple of float :
        Tuple (x, y, z) with the Cartesian coordinates.
    """

    x = radius * math.sin(theta) * math.cos(phi)
    y = radius * math.sin(theta) * math.sin(phi)
    z = radius * math.cos(theta)

    return (x, y, z)


##  @}
l = []


def triangle_normal(p1, p2, p3):
    """Quoting from https://www.khronos.org/opengl/wiki/Calculating_a_Surface_Normal

    A surface normal for a triangle can be calculated by taking the vector cross product of two edges of that triangle. The order of the vertices used in the calculation will affect the direction of the normal (in or out of the face w.r.t. winding).
    So for a triangle p1, p2, p3, if the vector A = p2 - p1 and the vector B = p3 - p1 then the normal N = A x B and can be calculated by:

    Nx = Ay * Bz - Az * By
    Ny = Az * Bx - Ax * Bz
    Nz = Ax * By - Ay * Bx"""

    A = p2 - p1
    B = p3 - p1
    # print(p1,p2,p3,A,B)
    Ax, Ay, Az = A
    Bx, By, Bz = B
    Nx = Ay * Bz - Az * By
    Ny = Az * Bx - Ax * Bz
    Nz = Ax * By - Ay * Bx
    return Nx, Ny, Nz


def triangle_plane(p1, p2, p3):
    """Quoting from https://www.khronos.org/opengl/wiki/Calculating_a_Surface_Normal

    A surface normal for a triangle can be calculated by taking the vector cross product of two edges of that triangle. The order of the vertices used in the calculation will affect the direction of the normal (in or out of the face w.r.t. winding).
    So for a triangle p1, p2, p3, if the vector A = p2 - p1 and the vector B = p3 - p1 then the normal N = A x B and can be calculated by:

    Nx = Ay * Bz - Az * By
    Ny = Az * Bx - Ax * Bz
    Nz = Ax * By - Ay * Bx"""

    A = p2 - p1
    B = p3 - p1
    # print(p1,p2,p3,A,B)
    Ax, Ay, Az = A
    Bx, By, Bz = B
    Nx = Ay * Bz - Az * By
    Ny = Az * Bx - Ax * Bz
    Nz = Ax * By - Ay * Bx
    return np.array([A, B, (Nx, Ny, Nz)])


unit = np.vectorize(_unit, signature='(i)->(i)')
