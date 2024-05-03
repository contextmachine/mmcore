cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double cdeboor(double[:] knots, double t, int i, int k) noexcept nogil:
    """
    Calculating basis function with de Boor algorithm
    """
    # print(t,i,k)


    cdef double c1,c2;

    if k == 0:
        return 1.0 if knots[i] <= t <= knots[i + 1] else 0.0
    if knots[i + k] == knots[i]:
        c1 = 0.0
    else:
        c1 = (t  - knots[i]) / (knots[i + k] - knots[i]) * cdeboor(knots, t, i, k - 1)

    if knots[i + k + 1] == knots[i + 1]:
        c2 = 0.0
    else:
        c2 = ((knots[i + k + 1] - t) / (knots[i + k + 1] - knots[i + 1]) * cdeboor(knots,t, i + 1, k - 1))

    return c1 + c2


def deboor(double[:] knots, double t, int i, int k)   :
    cdef double d=cdeboor(knots,t,i,k)
    return d

