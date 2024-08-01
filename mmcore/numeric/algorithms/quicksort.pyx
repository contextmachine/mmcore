cimport cython
cimport numpy as cnp
import numpy as np
cnp.import_array()

cdef extern from "_quicksort.c" nogil:
    void quickSort(double* arr, int low, int high)
    void argSort(double* arr, int* indices, int low, int high)
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






