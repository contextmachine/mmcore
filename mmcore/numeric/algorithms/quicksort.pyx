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
cimport mmcore.numeric.algorithms.quicksort
cnp.import_array()



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void quicksort(double[:] arr)  noexcept nogil:
    cdef int low=0
    cdef int n=arr.shape[0]
    cdef int high=n-1
    cdef int i
    #cdef double* arr_c=<double*> malloc(sizeof(double)*n)

    quickSort(&arr[0],low,high)


    #free(arr_c)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int[:] argsort(double[:] arr) :
    cdef int low=0
    cdef int n=arr.shape[0]
    cdef int high=n-1
    cdef int i
    cdef int[:] indices=np.arange(n,dtype=np.intc)
    argSort(&arr[0], &indices[0],low,  high)
    return indices







