cimport cython
from importlib.metadata import version
__version__ = version("mmcore")

from libc.stdlib cimport malloc,free


cdef inline double** init_double_ptr(double[:, :] arr) nogil:
    cdef int i, j
    cdef int rows = arr.shape[0]
    cdef int cols = arr.shape[1]
    cdef double** result = <double**>malloc(rows * sizeof(double*))

    for i in range(rows):
        result[i] = <double*>malloc(cols * sizeof(double))
        for j in range(cols):
            result[i][j] = arr[i, j]

    return result

cdef inline double*** init_double_ptr_ptr(double[:, :, :] arr) nogil:
    cdef int i, j, k
    cdef int dim1 = arr.shape[0]
    cdef int dim2 = arr.shape[1]
    cdef int dim3 = arr.shape[2]
    cdef double*** result = <double***>malloc(dim1 * sizeof(double**))

    for i in range(dim1):
        result[i] = <double**>malloc(dim2 * sizeof(double*))
        for j in range(dim2):
            result[i][j] = <double*>malloc(dim3 * sizeof(double))
            for k in range(dim3):
                result[i][j][k] = arr[i, j, k]

    return result

cdef inline void free_double_ptr(double** arr, int rows) nogil:
    cdef int i
    for i in range(rows):
        free(arr[i])
    free(arr)

cdef inline void free_double_ptr_ptr(double*** arr, int dim1, int dim2) nogil:
    cdef int i, j
    for i in range(dim1):
        for j in range(dim2):
            free(arr[i][j])
        free(arr[i])
    free(arr)
