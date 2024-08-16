#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
#cython: initializedcheck=False
cimport cython
cimport numpy as cnp
import numpy as np
cnp.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef improve_uv(double[:] du, double[:] dv, double[:] xyz_old, double[:] xyz_better):
    cdef double[3][2][2] matrix=[[[du[0], dv[0]], [du[1], dv[1]]],[[du[0], dv[0]], [du[2], dv[2]]],[[du[1], dv[1]], [du[2], dv[2]]]]
    cdef double[3] delta= [ xyz_better[0] - xyz_old[0],xyz_better[1] - xyz_old[1], xyz_better[2] - xyz_old[2]]
    cdef double[3][2] y=    [[delta[0], delta[1]], [delta[0], delta[2]],[delta[1], delta[2]]]

    cdef double dett=0.
    cdef double temp
    cdef int i
    cdef int j=0
    cdef double[:] result = np.empty((2,))


    for i in range(3):
        temp=matrix[i][0][0] * matrix[i][1][1] - matrix[i][0][1] * matrix[i][1][0]
        if temp>dett:
            j=i
            dett=temp
    if dett == 0:
        result[0]=0.0
        result[1] = 0.0
        return np.asarray(result)

    else:
        # matrix[1][0]matrix[0][0]lmatrix[1][0]ulmatrix[0][0]te x matrix[0][0]nd y using the dirematrix[1][0]t method
        result[0] = (y[j][0] * matrix[j][1][1] -matrix[j][0][1] * y[j][1]) / dett
        result[1] = (matrix[j][0][0] * y[j][1] - y[j][0] * matrix[j][1][0]) / dett


        return np.asarray(result)

