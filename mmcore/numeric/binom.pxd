
cimport cython

cdef public double[31][31] binomial_coefficients

@cython.cdivision(True)
cdef public double binomial_coefficient(int i, int j) noexcept nogil

