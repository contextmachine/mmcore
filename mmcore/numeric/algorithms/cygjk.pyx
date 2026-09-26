# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: embedsignature=True
# cython: infer_types=False
# cython: initializedcheck=False
# distutils: language = c++

cimport cython
from libc.math cimport fabs, isfinite, sqrt
from operator import index as _index

cimport mmcore.numeric.algorithms.cygjk
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(False)
def gjk(double[:,:] v1, double[:,:] v2, double tol=1e-6, int max_iter=25):
    cdef vector[Vec3[double]] vf1=vector[Vec3[double]](v1.shape[0])
    cdef vector[Vec3[double]] vf2=vector[Vec3[double]](v2.shape[0])
    cdef int i;

    for i in range(v1.shape[0]):
        vf1[i][0]=v1[i][0]
        vf1[i][1]=v1[i][1]
        vf1[i][2]=v1[i][2]
    for i in range(v2.shape[0]):
        vf2[i][0]=v2[i][0]
        vf2[i][1]=v2[i][1]
        vf2[i][2]=v2[i][2]

    cdef bool result= gjk_collision_detection(vf1, vf2, tol, max_iter)


    return result


def gjk_separating_axis(const double[:, :] v1, const double[:, :] v2,
                        double tol=1e-12, max_iter=25):
    """Propose a unit separating direction, or return ``None``.

    The direction comes from GJK's first negative support test.  It is not
    a separation certificate: callers must project every original control
    point onto it and verify their model-space padding and roundoff margin.
    ``tol`` is a numerical search parameter, not a geometric padding.
    Searches reporting contact/overlap or exhausting iterations return ``None``.

    Inputs must be finite, nonempty float64 buffers of shape ``(n, 3)``;
    read-only and strided buffers are supported.  Uniform internal scaling
    prevents overflow without changing the direction's coordinate frame.
    ``max_iter`` must be a positive integer, keeping the search bounded.
    """
    if v1 is None or v2 is None:
        raise ValueError("vertex buffers must be nonempty arrays of shape (n, 3)")
    if v1.shape[0] == 0 or v2.shape[0] == 0 or v1.shape[1] != 3 or v2.shape[1] != 3:
        raise ValueError("vertex buffers must be nonempty arrays of shape (n, 3)")
    if not isfinite(tol) or tol < 0.0:
        raise ValueError("tol must be finite and nonnegative")
    requested_iterations = _index(max_iter)
    if requested_iterations <= 0:
        raise ValueError("max_iter must be a positive integer")
    cdef size_t iterations = requested_iterations
    cdef vector[Vec3[double]] first = vector[Vec3[double]](v1.shape[0])
    cdef vector[Vec3[double]] second = vector[Vec3[double]](v2.shape[0])
    cdef Py_ssize_t i, coordinate
    cdef double value, scale = 0.0
    cdef Vec3[double] direction
    for i in range(v1.shape[0]):
        for coordinate in range(3):
            value = v1[i, coordinate]
            if not isfinite(value):
                raise ValueError("vertex coordinates must be finite")
            first[i][coordinate] = value
            scale = max(scale, fabs(value))
    for i in range(v2.shape[0]):
        for coordinate in range(3):
            value = v2[i, coordinate]
            if not isfinite(value):
                raise ValueError("vertex coordinates must be finite")
            second[i][coordinate] = value
            scale = max(scale, fabs(value))
    if scale == 0.0:
        return None
    for i in range(v1.shape[0]):
        for coordinate in range(3):
            first[i][coordinate] /= scale
    for i in range(v2.shape[0]):
        for coordinate in range(3):
            second[i][coordinate] /= scale
    if gjk_collision_detection_with_axis(first, second, tol, iterations, &direction):
        return None
    scale = 0.0
    for coordinate in range(3):
        if not isfinite(direction[coordinate]):
            return None
        scale = max(scale, fabs(direction[coordinate]))
    if scale == 0.0:
        return None
    for coordinate in range(3):
        direction[coordinate] /= scale
    scale = sqrt(direction[0] * direction[0] + direction[1] * direction[1]
                 + direction[2] * direction[2])
    return (direction[0] / scale, direction[1] / scale, direction[2] / scale)
