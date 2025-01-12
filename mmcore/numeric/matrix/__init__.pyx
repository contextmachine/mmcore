cimport cython
cimport mmcore.numeric.matrix
import numpy as np

cdef class Matrix:
    def __init__(self, int rows, int columns):
        self.m = create_matrix(rows, columns)

    def __getitem__(self, item):
        cdef double res = get_item(&self.m, item[0], item[1])
        return res
    def __setitem__(self, item, val):
        set_item(&self.m, item[0], item[1], <double> val)


cpdef double[:,:] inv(double[:,:] arr,double[:,:] result =None):
    cdef matrix m = matrix(NULL, 0,0);
    cdef matrix inversion = matrix(NULL, 0,0);
    m.data=&arr[0,0]
    m.rows=arr.shape[0]
    m.columns = arr.shape[1]


    set_matrix_from_array(&m,&arr[0,0])

    if result is None:
        result=np.zeros(arr.shape)
    inversion.data=&result[0,0]
    inversion.rows =result.shape[0]
    inversion.columns = result.shape[1]

    invert_matrix(&m,&inversion)
    
    return result