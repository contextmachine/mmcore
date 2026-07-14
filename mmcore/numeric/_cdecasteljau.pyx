# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: language_level=3
cimport cython
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free

cnp.import_array()

ctypedef double f64

# Maximum doubles for stack-allocated working buffer.
# Covers degree <= 32 with nval <= 4 => 33*4 = 132.
cdef int _STACK_MAX = 136


cdef inline void _split_fiber(
    const f64* data,    # source fiber
    int m,              # number of control points (degree+1)
    Py_ssize_t in_stride,  # stride (in doubles) between consecutive input points
    int nval,           # value components per point
    f64 t,
    f64* left,          # output left
    Py_ssize_t out_stride,  # stride (in doubles) between consecutive output points
    f64* right,         # output right (same out_stride)
) noexcept nogil:
    """
    De Casteljau split on one fiber of m control points.

    Input:  data[i * in_stride + v]  for i=0..m-1, v=0..nval-1
    Output: left[i * out_stride + v], right[i * out_stride + v]
    """
    cdef f64 stack_buf[136]
    cdef f64* buf
    cdef int total = m * nval
    cdef int use_heap = total > _STACK_MAX
    cdef f64 omt = 1.0 - t
    cdef int i, k, v
    cdef int mi

    if use_heap:
        buf = <f64*>malloc(total * sizeof(f64))
    else:
        buf = stack_buf

    # Copy fiber into contiguous working buffer: buf[i*nval + v]
    for i in range(m):
        for v in range(nval):
            buf[i * nval + v] = data[i * in_stride + v]

    # Level 0
    for v in range(nval):
        left[v] = buf[v]
        right[(m - 1) * out_stride + v] = buf[(m - 1) * nval + v]

    # De Casteljau pyramid
    for k in range(1, m):
        for i in range(m - k):
            for v in range(nval):
                buf[i * nval + v] = omt * buf[i * nval + v] + t * buf[(i + 1) * nval + v]
        for v in range(nval):
            left[k * out_stride + v] = buf[v]
        mi = m - 1 - k
        for v in range(nval):
            right[mi * out_stride + v] = buf[mi * nval + v]

    if use_heap:
        free(buf)


# ---------------------------------------------------------------------------
# 1D: curve — shape (m, nval)
# ---------------------------------------------------------------------------

cpdef tuple c_de_casteljau_split_1d(f64[:, ::1] ctrl, f64 t):
    """Split a 1D Bernstein curve at parameter t."""
    cdef int m = ctrl.shape[0]
    cdef int nval = ctrl.shape[1]

    left_arr = np.empty((m, nval), dtype=np.float64)
    right_arr = np.empty((m, nval), dtype=np.float64)

    cdef f64[:, ::1] lv = left_arr
    cdef f64[:, ::1] rv = right_arr

    _split_fiber(&ctrl[0, 0], m, nval, nval, t, &lv[0, 0], nval, &rv[0, 0])
    return (left_arr, right_arr)


# ---------------------------------------------------------------------------
# 2D: surface — shape (nu, nv, nval)
# ---------------------------------------------------------------------------

cpdef tuple c_de_casteljau_split_2d(f64[:, :, ::1] ctrl, int axis, f64 t):
    """Split a 2D Bernstein surface along `axis` at parameter t."""
    cdef int nu = ctrl.shape[0]
    cdef int nv = ctrl.shape[1]
    cdef int nval = ctrl.shape[2]

    left_arr = np.empty((nu, nv, nval), dtype=np.float64)
    right_arr = np.empty((nu, nv, nval), dtype=np.float64)

    cdef f64[:, :, ::1] lv = left_arr
    cdef f64[:, :, ::1] rv = right_arr

    cdef int i, j
    cdef Py_ssize_t in_stride, out_stride

    if axis == 0:
        # Fibers along axis 0: ctrl[:, j, :]
        # in_stride  between ctrl[i,j,:] and ctrl[i+1,j,:] = nv * nval
        # out_stride between lv[i,j,:] and lv[i+1,j,:]     = nv * nval
        in_stride = nv * nval
        out_stride = nv * nval
        for j in range(nv):
            _split_fiber(
                &ctrl[0, j, 0], nu, in_stride, nval, t,
                &lv[0, j, 0], out_stride, &rv[0, j, 0],
            )
    else:
        # axis == 1: fibers along axis 1: ctrl[i, :, :]
        # in_stride  = nval (contiguous along axis 1)
        # out_stride = nval
        in_stride = nval
        out_stride = nval
        for i in range(nu):
            _split_fiber(
                &ctrl[i, 0, 0], nv, in_stride, nval, t,
                &lv[i, 0, 0], out_stride, &rv[i, 0, 0],
            )

    return (left_arr, right_arr)


# ---------------------------------------------------------------------------
# 3D: trivariate — shape (na, nb, nc, nval)
# ---------------------------------------------------------------------------

cpdef tuple c_de_casteljau_split_3d(f64[:, :, :, ::1] ctrl, int axis, f64 t):
    """Split a 3D Bernstein trivariate along `axis` at parameter t."""
    cdef int na = ctrl.shape[0]
    cdef int nb = ctrl.shape[1]
    cdef int nc = ctrl.shape[2]
    cdef int nval = ctrl.shape[3]

    left_arr = np.empty((na, nb, nc, nval), dtype=np.float64)
    right_arr = np.empty((na, nb, nc, nval), dtype=np.float64)

    cdef f64[:, :, :, ::1] lv = left_arr
    cdef f64[:, :, :, ::1] rv = right_arr

    cdef int i, j, k
    cdef Py_ssize_t in_stride, out_stride

    if axis == 0:
        # Fibers along axis 0: ctrl[:, j, k, :]
        in_stride = nb * nc * nval
        out_stride = nb * nc * nval
        for j in range(nb):
            for k in range(nc):
                _split_fiber(
                    &ctrl[0, j, k, 0], na, in_stride, nval, t,
                    &lv[0, j, k, 0], out_stride, &rv[0, j, k, 0],
                )
    elif axis == 1:
        # Fibers along axis 1: ctrl[i, :, k, :]
        in_stride = nc * nval
        out_stride = nc * nval
        for i in range(na):
            for k in range(nc):
                _split_fiber(
                    &ctrl[i, 0, k, 0], nb, in_stride, nval, t,
                    &lv[i, 0, k, 0], out_stride, &rv[i, 0, k, 0],
                )
    else:
        # axis == 2: fibers along axis 2: ctrl[i, j, :, :]
        in_stride = nval
        out_stride = nval
        for i in range(na):
            for j in range(nb):
                _split_fiber(
                    &ctrl[i, j, 0, 0], nc, in_stride, nval, t,
                    &lv[i, j, 0, 0], out_stride, &rv[i, j, 0, 0],
                )

    return (left_arr, right_arr)


# ---------------------------------------------------------------------------
# General dispatcher
# ---------------------------------------------------------------------------

cpdef tuple c_de_casteljau_split_nd(object ctrl_obj, int axis, f64 t):
    """
    Cython-accelerated de Casteljau split for 1D/2D/3D parametric grids.

    ctrl must be a C-contiguous float64 numpy array with shape:
      (m, nval)              for ndim==2 (1D parametric)
      (nu, nv, nval)         for ndim==3 (2D parametric)
      (na, nb, nc, nval)     for ndim==4 (3D parametric)

    Returns (left, right) numpy arrays of same shape.
    """
    cdef int ndim = ctrl_obj.ndim
    cdef int param_ndim = ndim - 1

    # Normalize negative axis
    if axis < 0:
        axis += param_ndim
    if axis < 0 or axis >= param_ndim:
        raise ValueError(f"axis {axis} out of range for {param_ndim} parametric dimensions")

    if ndim == 2:
        return c_de_casteljau_split_1d(ctrl_obj, t)
    elif ndim == 3:
        return c_de_casteljau_split_2d(ctrl_obj, axis, t)
    elif ndim == 4:
        return c_de_casteljau_split_3d(ctrl_obj, axis, t)
    else:
        raise ValueError(f"c_de_casteljau_split_nd: unsupported ndim={ndim}")
