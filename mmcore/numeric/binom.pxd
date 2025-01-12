
cimport cython

cdef public double[31][31] binomial_coefficients


cdef public double binomial_coefficient(int i, int j) noexcept nogil

