
cimport cython
cimport numpy as cnp
import numpy as np
cimport mmcore.numeric.algorithms.quicksort
from cvxpy import reshape
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





@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:] unique(double[:] arr, double[:] result=None) noexcept nogil:
    cdef int new_len=0;
    cdef bint success
    if result is None:
        with gil:
            result=arr.copy()
    success=uniqueSortedByRef(&arr[0],arr.shape[0], &new_len,&result[0])
    if success==0:
        with gil:
            print('error')


    return result[:new_len]









