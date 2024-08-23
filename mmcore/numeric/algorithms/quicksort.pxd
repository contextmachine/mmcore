



cdef extern from "_quicksort.c" nogil:
    void quickSort(double* arr, int low, int high)
    void argSort(double* arr, int* indices, int low, int high)

cdef extern from "_unique.c" nogil:
    double* uniqueSortedEps(double* array, int size, double eps, int* new_size)
    double* uniqueSorted(double *arr, int n, int *new_len) 
cpdef void quicksort(double[:] arr) noexcept nogil
cpdef int[:] argsort(double[:] arr)