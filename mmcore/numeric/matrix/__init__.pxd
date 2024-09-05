cimport cython

cdef extern from "cmatrix.h" nogil:
    ctypedef struct matrix:
        double* data;
        int rows;
        int columns;

   
    double get_item(matrix* m, int i, int j)
    void set_item(matrix *m, int i, int j, double val)
    void   set_matrix_from_array(matrix* m, double* arr)
    void   set_matrix_from_array2d(matrix* m, double[:,:] arr)
    matrix create_matrix(int rows, int columns)
    void  free_matrix(matrix * m)
    void print_matrix(matrix* m)
    int LU_decomposition(matrix * A, int * P)
    void  LU_solve(matrix * A, int * P, double * b, double * x)
    int invert_matrix(matrix* A, matrix* inverse)

cdef class Matrix:
    cdef matrix m
