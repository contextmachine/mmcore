#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
#cython: initializedcheck=False


cimport cython
from mmcore cimport init_double_ptr,free_double_ptr
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc,free
cimport mmcore.geom.nurbs.algorithms
from mmcore.numeric.algorithms.quicksort cimport uniqueSorted
from libc.math cimport fabs, sqrt,fmin,fmax,pow

cnp.import_array()
cdef public double[31][31] binomial_coefficients=[[1.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0],
 [1.0,
  1.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0],
 [1.0,
  2.0,
  1.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0],
 [1.0,
  3.0,
  3.0,
  1.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0],
 [1.0,
  4.0,
  6.0,
  4.0,
  1.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0],
 [1.0,
  5.0,
  10.0,
  10.0,
  5.0,
  1.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0],
 [1.0,
  6.0,
  15.0,
  20.0,
  15.0,
  6.0,
  1.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0],
 [1.0,
  7.0,
  21.0,
  35.0,
  35.0,
  21.0,
  7.0,
  1.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0],
 [1.0,
  8.0,
  28.0,
  56.0,
  70.0,
  56.0,
  28.0,
  8.0,
  1.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0],
 [1.0,
  9.0,
  36.0,
  84.0,
  126.0,
  126.0,
  84.0,
  36.0,
  9.0,
  1.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0],
 [1.0,
  10.0,
  45.0,
  120.0,
  210.0,
  252.0,
  210.0,
  120.0,
  45.0,
  10.0,
  1.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0],
 [1.0,
  11.0,
  55.0,
  165.0,
  330.0,
  462.0,
  462.0,
  330.0,
  165.0,
  55.0,
  11.0,
  1.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0],
 [1.0,
  12.0,
  66.0,
  220.0,
  495.0,
  792.0,
  924.0,
  792.0,
  495.0,
  220.0,
  66.0,
  12.0,
  1.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0],
 [1.0,
  13.0,
  78.0,
  286.0,
  715.0,
  1287.0,
  1716.0,
  1716.0,
  1287.0,
  715.0,
  286.0,
  78.0,
  13.0,
  1.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0],
 [1.0,
  14.0,
  91.0,
  364.0,
  1001.0,
  2002.0,
  3003.0,
  3432.0,
  3003.0,
  2002.0,
  1001.0,
  364.0,
  91.0,
  14.0,
  1.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0],
 [1.0,
  15.0,
  105.0,
  455.0,
  1365.0,
  3003.0,
  5005.0,
  6435.0,
  6435.0,
  5005.0,
  3003.0,
  1365.0,
  455.0,
  105.0,
  15.0,
  1.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0],
 [1.0,
  16.0,
  120.0,
  560.0,
  1820.0,
  4368.0,
  8008.0,
  11440.0,
  12870.0,
  11440.0,
  8008.0,
  4368.0,
  1820.0,
  560.0,
  120.0,
  16.0,
  1.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0],
 [1.0,
  17.0,
  136.0,
  680.0,
  2380.0,
  6188.0,
  12376.0,
  19448.0,
  24310.0,
  24310.0,
  19448.0,
  12376.0,
  6188.0,
  2380.0,
  680.0,
  136.0,
  17.0,
  1.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0],
 [1.0,
  18.0,
  153.0,
  816.0,
  3060.0,
  8568.0,
  18564.0,
  31824.0,
  43758.0,
  48620.0,
  43758.0,
  31824.0,
  18564.0,
  8568.0,
  3060.0,
  816.0,
  153.0,
  18.0,
  1.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0],
 [1.0,
  19.0,
  171.0,
  969.0,
  3876.0,
  11628.0,
  27132.0,
  50388.0,
  75582.0,
  92378.0,
  92378.0,
  75582.0,
  50388.0,
  27132.0,
  11628.0,
  3876.0,
  969.0,
  171.0,
  19.0,
  1.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0],
 [1.0,
  20.0,
  190.0,
  1140.0,
  4845.0,
  15504.0,
  38760.0,
  77520.0,
  125970.0,
  167960.0,
  184756.0,
  167960.0,
  125970.0,
  77520.0,
  38760.0,
  15504.0,
  4845.0,
  1140.0,
  190.0,
  20.0,
  1.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0],
 [1.0,
  21.0,
  210.0,
  1330.0,
  5985.0,
  20349.0,
  54264.0,
  116280.0,
  203490.0,
  293930.0,
  352716.0,
  352716.0,
  293930.0,
  203490.0,
  116280.0,
  54264.0,
  20349.0,
  5985.0,
  1330.0,
  210.0,
  21.0,
  1.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0],
 [1.0,
  22.0,
  231.0,
  1540.0,
  7315.0,
  26334.0,
  74613.0,
  170544.0,
  319770.0,
  497420.0,
  646646.0,
  705432.0,
  646646.0,
  497420.0,
  319770.0,
  170544.0,
  74613.0,
  26334.0,
  7315.0,
  1540.0,
  231.0,
  22.0,
  1.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0],
 [1.0,
  23.0,
  253.0,
  1771.0,
  8855.0,
  33649.0,
  100947.0,
  245157.0,
  490314.0,
  817190.0,
  1144066.0,
  1352078.0,
  1352078.0,
  1144066.0,
  817190.0,
  490314.0,
  245157.0,
  100947.0,
  33649.0,
  8855.0,
  1771.0,
  253.0,
  23.0,
  1.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0],
 [1.0,
  24.0,
  276.0,
  2024.0,
  10626.0,
  42504.0,
  134596.0,
  346104.0,
  735471.0,
  1307504.0,
  1961256.0,
  2496144.0,
  2704156.0,
  2496144.0,
  1961256.0,
  1307504.0,
  735471.0,
  346104.0,
  134596.0,
  42504.0,
  10626.0,
  2024.0,
  276.0,
  24.0,
  1.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0],
 [1.0,
  25.0,
  300.0,
  2300.0,
  12650.0,
  53130.0,
  177100.0,
  480700.0,
  1081575.0,
  2042975.0,
  3268760.0,
  4457400.0,
  5200300.0,
  5200300.0,
  4457400.0,
  3268760.0,
  2042975.0,
  1081575.0,
  480700.0,
  177100.0,
  53130.0,
  12650.0,
  2300.0,
  300.0,
  25.0,
  1.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0],
 [1.0,
  26.0,
  325.0,
  2600.0,
  14950.0,
  65780.0,
  230230.0,
  657800.0,
  1562275.0,
  3124550.0,
  5311735.0,
  7726160.0,
  9657700.0,
  10400600.0,
  9657700.0,
  7726160.0,
  5311735.0,
  3124550.0,
  1562275.0,
  657800.0,
  230230.0,
  65780.0,
  14950.0,
  2600.0,
  325.0,
  26.0,
  1.0,
  0.0,
  0.0,
  0.0,
  0.0],
 [1.0,
  27.0,
  351.0,
  2925.0,
  17550.0,
  80730.0,
  296010.0,
  888030.0,
  2220075.0,
  4686825.0,
  8436285.0,
  13037895.0,
  17383860.0,
  20058300.0,
  20058300.0,
  17383860.0,
  13037895.0,
  8436285.0,
  4686825.0,
  2220075.0,
  888030.0,
  296010.0,
  80730.0,
  17550.0,
  2925.0,
  351.0,
  27.0,
  1.0,
  0.0,
  0.0,
  0.0],
 [1.0,
  28.0,
  378.0,
  3276.0,
  20475.0,
  98280.0,
  376740.0,
  1184040.0,
  3108105.0,
  6906900.0,
  13123110.0,
  21474180.0,
  30421755.0,
  37442160.0,
  40116600.0,
  37442160.0,
  30421755.0,
  21474180.0,
  13123110.0,
  6906900.0,
  3108105.0,
  1184040.0,
  376740.0,
  98280.0,
  20475.0,
  3276.0,
  378.0,
  28.0,
  1.0,
  0.0,
  0.0],
 [1.0,
  29.0,
  406.0,
  3654.0,
  23751.0,
  118755.0,
  475020.0,
  1560780.0,
  4292145.0,
  10015005.0,
  20030010.0,
  34597290.0,
  51895935.0,
  67863915.0,
  77558760.0,
  77558760.0,
  67863915.0,
  51895935.0,
  34597290.0,
  20030010.0,
  10015005.0,
  4292145.0,
  1560780.0,
  475020.0,
  118755.0,
  23751.0,
  3654.0,
  406.0,
  29.0,
  1.0,
  0.0],
 [1.0,
  30.0,
  435.0,
  4060.0,
  27405.0,
  142506.0,
  593775.0,
  2035800.0,
  5852925.0,
  14307150.0,
  30045015.0,
  54627300.0,
  86493225.0,
  119759850.0,
  145422675.0,
  155117520.0,
  145422675.0,
  119759850.0,
  86493225.0,
  54627300.0,
  30045015.0,
  14307150.0,
  5852925.0,
  2035800.0,
  593775.0,
  142506.0,
  27405.0,
  4060.0,
  435.0,
  30.0,
  1.0]]

@cython.cdivision(True)
cdef public double binomial_coefficient(int i, int j) noexcept nogil:
    return binomial_coefficients[i][j]


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int find_span(int n, int p, double u, double[:] U, bint is_periodic) noexcept nogil:
    """
    Determine the knot span index for a given parameter value `u`.

    This function finds the knot span index `i` such that the parameter `u` 
    lies within the interval [U[i], U[i+1]] in the knot vector `U`. 
    The knot vector `U` is assumed to be non-decreasing and the parameter 
    `u` is within the range `[U[p], U[n+1]]`.

    Parameters
    ----------
    n : int
        The maximum index of the knot span, typically the number of basis functions minus one.
    p : int
        The degree of the B-spline or NURBS.
    u : float
        The parameter value for which the span index is to be found.
    U : list of float
        The knot vector, a non-decreasing sequence of real numbers.

    Returns
    -------
    int
        The index `i` such that `U[i] <= u < U[i+1]`, where `i` is the knot span.

    Raises
    ------
    ValueError
        If the parameter `u` is outside the bounds of the knot vector `U` or 
        if the function fails to find a valid span within the maximum iterations.

    Notes
    -----
    The function employs a binary search algorithm to efficiently locate 
    the knot span. It handles special cases where `u` is exactly equal to 
    the last value in `U` and when `u` is outside the range of `U`.

    Example
    -------
    >>> U = [0, 0, 0, 0.5, 1, 1, 1]
    >>> find_span(4, 2, 0.3, U)
    2

    >>> find_span(4, 2, 0.5, U)
    3
    """
    cdef double U_min = U[p]
    cdef double U_max = U[n+1]
    cdef double period


    if is_periodic :
        # Wrap u to be within the valid range for periodic and closed curves

        period= U_max - U_min
        while u < U_min:
            u += period
        while u > U_max:
            u -= period

    else:
        # Clamp u to be within the valid range for open curves

        if u >= U[n+1]:

            return n

        elif u < U[0]:

            return p

        # Handle special case for the upper boundary
    if u == U[n + 1]:
        return n


    # Binary search for the correct knot span
    cdef int low = p
    cdef int high = n + 1
    cdef int mid = (low + high) // 2

    while u < U[mid] or u >= U[mid + 1]:
        if u < U[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2

    return mid


@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:, :] all_basis_funs(int span, double u, int p, double[:] U):
    """
    Compute all nonzero basis functions and their derivatives up to the ith-degree basis function.

    Parameters:
    span (int): The knot span index.
    u (double): The parameter value.
    p (int): The degree of the basis functions.
    U (double[:]): The knot vector.

    Returns:
    double[:, :]: The basis functions.
    """

    cdef int pp = p+1
    cdef double[:, :] N = np.zeros((pp, pp))
    cdef double[:] left = np.zeros(pp)
    cdef double[:] right = np.zeros(pp)
    N[0, 0] = 1.0

    cdef int j, r
    cdef double saved, temp
    for j in range(1, pp):
        left[j] = u - U[span + 1 - j]
        right[j] = U[span + j] - u
        saved = 0.0
        for r in range(j):
            N[j, r] = right[r + 1] + left[j - r]
            temp = N[r, j - 1] / N[j, r]
            N[r, j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        N[j, j] = saved

    return N

@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:, :] ders_basis_funs(int i, double u, int p, int n, double[:] U, double[:,:] ders =None):
    """
    Compute the nonzero basis functions and their derivatives for a B-spline.

    This function calculates the nonzero basis functions and their derivatives 
    for a given parameter value `u` in a B-spline curve. The derivatives are 
    computed up to the `n`-th derivative.

    Parameters
    ----------
    i : int
        The knot span index such that `U[i] <= u < U[i+1]`.
    u : double
        The parameter value at which the basis functions and their derivatives 
        are evaluated.
    p : int
        The degree of the B-spline basis functions.
    n : int
        The number of derivatives to compute (0 for just the basis functions).
    U : double[:]
        The knot vector.

    Returns
    -------
    double[:, :]
        A 2D array `ders` of shape `(n+1, p+1)` where `ders[k, j]` contains the 
        `k`-th derivative of the `j`-th nonzero basis function at `u`.

    Notes
    -----
    - The algorithm is based on the approach described in "The NURBS Book" by 
      Piegl and Tiller, which efficiently computes the derivatives using 
      divided differences and recursive evaluation.
    - The function utilizes several optimizations provided by Cython, such as 
      disabling bounds checking and wraparound for faster execution.

    Example
    -------
    Suppose we have a degree 3 B-spline (cubic) with a knot vector `U`, and we 
    want to evaluate the basis functions and their first derivative at `u=0.5` 
    for the knot span `i=3`:

    .. code-block:: python

        import numpy as np
        cdef double[:] U = np.array([0, 0, 0, 0.5, 1, 1, 1], dtype=np.float64)
        ders = ders_basis_funs(3, 0.5, 3, 1, U)
        print(ders)

    This will output the basis functions and their first derivative for the 
    specified parameters.
    """

    cdef int pp=p+1
    cdef int nn = n + 1
    cdef int s1, s2
    
    cdef double[:, :] ndu = np.zeros((pp,pp))
    cdef double[:] left = np.zeros(pp)
    cdef double[:] right = np.zeros(pp)
    cdef double[:, :] a = np.zeros((2, pp))
    ndu[0, 0] = 1.0

    cdef int j, r, k, rk, pk, j1, j2
    cdef double saved, temp, d
    if ders is None:
        ders = np.zeros((nn, pp))
    for j in range(1, pp):
        left[j] = u - U[i + 1 - j]
        right[j] = U[i + j] - u
        saved = 0.0
        for r in range(j):
            ndu[j, r] = right[r + 1] + left[j - r]
            temp = ndu[r, j - 1] / ndu[j, r]
            ndu[r, j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        ndu[j, j] = saved

    for j in range(pp):
        ders[0, j] = ndu[j, p]



    for r in range(pp):
        s1 = 0
        s2 = 1
        a[0, 0] = 1.0
        for k in range(1, n + 1):
            d = 0.0
            rk = r - k
            pk = p - k
            if r >= k:
                a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                d = a[s2, 0] * ndu[rk, pk]
            if rk >= -1:
                j1 = 1
            else:
                j1 = -rk
            if r - 1 <= pk:
                j2 = k - 1
            else:
                j2 = p - r
            for j in range(j1, j2 + 1):
                a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                d += a[s2, j] * ndu[rk + j, pk]
            if r <= pk:
                a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                d += a[s2, k] * ndu[r, pk]
            ders[k, r] = d
            j = s1
            s1 = s2
            s2 = j

    r = p
    for k in range(1, n + 1):
        for j in range(pp):
            ders[k, j] *= r
        r *= (p - k)

    return ders

@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void curve_derivs_alg1(int n, int p, double[:] U, double[:, :] P, double u, int d, double[:, :] CK,bint is_periodic):
    """
    Compute the derivatives of a B-spline curve.

    Parameters:
    n (int): The number of basis functions minus one.
    p (int): The degree of the basis functions.
    U (double[:]): The knot vector.
    P (double[:, :]): The control points of the B-spline curve.
    u (double): The parameter value.
    d (int): The number of derivatives to compute.

    Returns:
    ndarray: The computed curve derivatives.
    """


    cdef int du = min(d, p)
    #cdef double[:, :] CK = np.zeros((du + 1, P.shape[1]))
    cdef int pp=p+1
    cdef int span = find_span_inline(n, p, u, U,is_periodic)
    cdef double[:, :] nders = ders_basis_funs(span, u, p, du, U)

    cdef int k, j, l
    for k in range(du + 1):
        for j in range(pp):
            for l in range(CK.shape[1]):
                CK[k, l] += nders[k, j] * P[span - p + j, l]

    #return np.asarray(CK)

@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void curve_deriv_cpts(int p, double[:] U, double[:, :] P, int d, int r1, int r2, double[:, :, :] PK) :
    """
    Compute control points of curve derivatives.

    Parameters:
    p (int): The degree of the basis functions.
    U (double[:]): The knot vector.
    P (double[:, :]): The control points of the B-spline curve.
    d (int): The number of derivatives to compute.
    r1 (int): The start index for control points.
    r2 (int): The end index for control points.
    PK (double[:, :, :]): The computed control points of curve derivatives.

    Returns:
    void
    """


    cdef int r = r2 - r1
    #cdef double[:, :, :] PK = np.zeros((d + 1, r + 1, P.shape[1]))

    cdef int i, k, j,pp
    pp=p+1
    cdef double tmp
    for i in range(r + 1):

        PK[0, i, :] = P[r1 + i, :]

    for k in range(1, d + 1):
        tmp = p - k + 1
        for i in range(r - k + 1):
            for j in range(P.shape[1]):

                PK[k, i, j] = tmp * (PK[k - 1, i + 1, j] - PK[k - 1, i, j]) / (U[r1 + i + pp] - U[r1 + i + k])



@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void curve_derivs_alg2(int n, int p, double[:] U, double[:, :] P, double u, int d, double[:, :] CK, double[:, :,:] PK,bint is_periodic):
    """
    Compute the derivatives of a B-spline curve.

    Parameters:
    n (int): The number of basis functions minus one.
    p (int): The degree of the basis functions.
    U (double[:]): The knot vector.
    P (double[:, :]): The control points of the B-spline curve.
    u (double): The parameter value.
    d (int): The number of derivatives to compute.

    Returns:
    ndarray: The computed curve derivatives
    """


    cdef int dimension = P.shape[1]
    cdef int degree = p
    cdef double[:] knotvector = U

    cdef int du = min(degree, d)

    #cdef double[:, :] CK = np.zeros((d + 1, dimension))

    cdef int span = find_span_inline(n, degree, u, knotvector,is_periodic)
    #cdef double[:, :, :] PK = np.zeros((d + 1, degree + 1, P.shape[1]))

    cdef double[:, :] bfuns = all_basis_funs(span, u, degree, knotvector)

    curve_deriv_cpts(degree, knotvector, P, d, span - degree, span, PK)

    cdef int k, j, i
    for k in range(du + 1):
        for j in range(degree - k + 1):
            for i in range(P.shape[1]):
                CK[k, i] += bfuns[j, degree - k] * PK[k, j, i]

@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void projective_to_cartesian(double[:] point, double[:] result)  noexcept nogil:
    cdef double w = point[3]
    result[0]=point[0]/w
    result[1]=point[1]/w
    result[2]=point[2]/w

@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void projective_to_cartesian_ptr_ptr(double* point, double* result)  noexcept nogil:
    cdef double w = point[3]
    result[0]=point[0]/w
    result[1]=point[1]/w
    result[2]=point[2]/w
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void projective_to_cartesian_ptr_mem(double* point, double[:] result)  noexcept nogil:
    cdef double w = point[3]
    result[0]=point[0]/w
    result[1]=point[1]/w
    result[2]=point[2]/w


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cpdef int find_multiplicity(double knot, double[:] knot_vector, double tol=1e-07):
    cdef int mult=0
    cdef int l=knot_vector.shape[0]
    cdef int i
    cdef double difference
    for i in range(l):
        difference=knot - knot_vector[i]
        if fabs(difference) <= tol:
            mult += 1
    return mult


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef  knot_insertion(int degree, double[:] knotvector, double[:, :] ctrlpts, double u, int num=1, int s=0, int span=0, bint is_periodic=0,double[:, :] result=None):
   
    cdef int n = ctrlpts.shape[0]
    if span ==0:
        span = find_span_inline( n, degree,  u, knotvector, is_periodic)
    
    if s==0:
        s = find_multiplicity(u, knotvector)
    cdef int nq = n + num
    cdef int dim = ctrlpts.shape[1]


    cdef double* temp = <double*>malloc(sizeof(double) * (degree + 1) * dim)
    
    cdef int i, j, L, idx
    cdef double alpha
    if result is None:
        result = np.zeros((nq, dim), dtype=np.float64)

    
    for i in range(span - degree + 1):
        result[i] = ctrlpts[i]
    for i in range(span - s, n):
        result[i + num] = ctrlpts[i]
    
    for i in range(degree - s + 1):
        memcpy(&temp[i * dim], &ctrlpts[span - degree + i, 0], sizeof(double) * dim)
    
    for j in range(1, num + 1):
        L = span - degree + j
        for i in range(degree - j - s + 1):
            alpha = knot_insertion_alpha(u, knotvector, span, i, L)
            for idx in range(dim):
                temp[i * dim + idx] = alpha * temp[(i + 1) * dim + idx] + (1.0 - alpha) * temp[i * dim + idx]
        memcpy(&result[L, 0], &temp[0], sizeof(double) * dim)
        memcpy(&result[span + num - j - s, 0], &temp[(degree - j - s) * dim], sizeof(double) * dim)
    
    L = span - degree + num
    for i in range(L + 1, span - s):
        memcpy(&result[i, 0], &temp[(i - L) * dim], sizeof(double) * dim)
    
    free(temp)

    return np.asarray(result)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:] knot_insertion_kv(double[:] knotvector, double u, int span, int r):
    cdef int kv_size = knotvector.shape[0]
    cdef double[:] kv_updated = np.zeros(kv_size + r, dtype=np.float64)
    
    cdef int i
    for i in range(span + 1):
        kv_updated[i] = knotvector[i]
    for i in range(1, r + 1):
        kv_updated[span + i] = u
    for i in range(span + 1, kv_size):
        kv_updated[i + r] = knotvector[i]
    
    return kv_updated

cdef inline double point_distance(double* a, double* b ,int dim):
    cdef int i
    cdef double temp=0.
    for i in range(dim):

        temp+= pow(a[i]+b[i], 2)

    return sqrt(temp)



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef knot_removal(int degree, double[:] knotvector, double[:, :] ctrlpts, double u, double tol=1e-4, int num=1,bint is_periodic=0) noexcept:
    cdef int s = find_multiplicity(u, knotvector)
    cdef int n = ctrlpts.shape[0]
    #n, degree,  u, knotvector, is_periodic
    cdef int r = find_span_inline(n, degree,  u, knotvector,is_periodic)
    
    cdef int first = r - degree
    cdef int last = r - s
    

    cdef int dim = ctrlpts.shape[1]
    cdef double[:, :] ctrlpts_new = np.zeros((n, dim), dtype=np.float64)
    memcpy(&ctrlpts_new[0, 0], &ctrlpts[0, 0], sizeof(double) * n * dim)
    
    cdef double* temp = <double*>malloc(sizeof(double) * ((2 * degree) + 1) * dim)
    
    cdef int t, i, j, ii, jj, k
    cdef bint remflag
    cdef double alpha_i, alpha_j
    cdef double[:] ptn = np.zeros(dim, dtype=np.float64)
    
    for t in range(num):
        memcpy(&temp[0], &ctrlpts[first - 1, 0], sizeof(double) * dim)
        memcpy(&temp[(last - first + 2) * dim], &ctrlpts[last + 1, 0], sizeof(double) * dim)
        i = first
        j = last
        ii = 1
        jj = last - first + 1
        remflag = False
        
        while j - i >= t:
            alpha_i = knot_removal_alpha_i(u, degree, knotvector, t, i)
            alpha_j = knot_removal_alpha_j(u, degree, knotvector, t, j)
            for k in range(dim):
                temp[ii * dim + k] = (ctrlpts[i, k] - (1.0 - alpha_i) * temp[(ii - 1) * dim + k]) / alpha_i
                temp[jj * dim + k] = (ctrlpts[j, k] - alpha_j * temp[(jj + 1) * dim + k]) / (1.0 - alpha_j)
            i += 1
            j -= 1
            ii += 1
            jj -= 1
        
        if j - i < t:
            if point_distance(&temp[(ii - 1) * dim], &temp[(jj + 1) * dim], dim) <= tol:
                remflag = True
        else:
            alpha_i = knot_removal_alpha_i(u, degree, knotvector, t, i)
            for k in range(dim):
                ptn[k] = (alpha_i * temp[(ii + t + 1) * dim + k]) + ((1.0 - alpha_i) * temp[(ii - 1) * dim + k])
            if point_distance(&ctrlpts[i, 0], &ptn[0], dim) <= tol:
                remflag = True
        
        if remflag:
            i = first
            j = last
            while j - i > t:
                memcpy(&ctrlpts_new[i, 0], &temp[(i - first + 1) * dim], sizeof(double) * dim)
                memcpy(&ctrlpts_new[j, 0], &temp[(j - first + 1) * dim], sizeof(double) * dim)
                i += 1
                j -= 1
        
        first -= 1
        last += 1
    
    t += 1
    j = (2 * r - s - degree) // 2
    i = j
    for k in range(1, t):
        if k % 2 == 1:
            i += 1
        else:
            j -= 1
    for k in range(i + 1, n):
        memcpy(&ctrlpts_new[j, 0], &ctrlpts[k, 0], sizeof(double) * dim)
        j += 1
    
    free(temp)
    return np.asarray(ctrlpts_new[:-t])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef tuple knot_refinement(int degree, double[:] knotvector, double[:, :] ctrlpts, double[:] knot_list=None,  double[:]  add_knot_list=None, int density=1, bint is_periodic=0) :
    cdef int n = ctrlpts.shape[0] - 1
    cdef int m = n + degree + 1
    cdef int dim = ctrlpts.shape[1]
    cdef double alpha
    cdef int d, i

    if knot_list is None:
        knot_list = np.array(knotvector[degree:-degree], dtype=np.float64)
    
    if add_knot_list is not None:

        knot_list = np.concatenate((knot_list, add_knot_list))
    
    cdef int usz=knot_list.shape[0]
    cdef int new_knot_len

    cdef double* sorted_knots=uniqueSorted(&knot_list[0],usz, &new_knot_len )
    
    knot_list = <double[:new_knot_len]>sorted_knots


    cdef double[:] rknots
    cdef int rknots_size

    for d in range(density - 1):
        rknots_size = usz * 2 - 1
        rknots = np.zeros(rknots_size, dtype=np.float64)
        
        for i in range(usz - 1):
            rknots[2 * i] = knot_list[i]
            rknots[2 * i + 1] = knot_list[i] + (knot_list[i + 1] - knot_list[i]) / 2.0
            
        rknots[-1] = knot_list[-1]
        knot_list = rknots
        usz = rknots_size
    cdef double[:] X = np.zeros(knot_list.shape[0] * degree, dtype=np.float64)
    cdef int x_count = 0
    cdef int s, r
    for mk in knot_list:
        s = find_multiplicity(mk, knotvector)
        r = degree - s
        for _ in range(r):
            X[x_count] = mk
            x_count += 1
    X = X[:x_count]
    
    if x_count == 0:
        raise Exception("Cannot refine knot vector on this parametric dimension")
    
    cdef int r_val = x_count - 1
    cdef int a = find_span_inline(n, degree,  X[0],knotvector, is_periodic)
    #TODO !!!!! Проверить это место если возникнут проблемы
    #n, degree,  u, knotvector, is_periodic
    cdef int b = find_span_inline(n, degree, X[r_val],knotvector,is_periodic) + 1
    
    cdef double[:, :] new_ctrlpts = np.zeros((n + r_val + 2, dim), dtype=np.float64)
    cdef double[:] new_kv = np.zeros(m + r_val + 2, dtype=np.float64)
    
    cdef int j, k, l,idx,idx2
    for j in range(a - degree + 1):
        new_ctrlpts[j] = ctrlpts[j]
    for j in range(b - 1, n + 1):
        new_ctrlpts[j + r_val + 1] = ctrlpts[j]
    
    for j in range(a + 1):
        new_kv[j] = knotvector[j]
    for j in range(b + degree, m + 1):
        new_kv[j + r_val + 1] = knotvector[j]
    
    i = b + degree - 1
    k = b + degree + r_val
    j = r_val
  
    
    while j >= 0:
        while X[j] <= knotvector[i] and i > a:
            new_ctrlpts[k - degree - 1] = ctrlpts[i - degree - 1]
            new_kv[k] = knotvector[i]
            k -= 1
            i -= 1
        memcpy(&new_ctrlpts[k - degree - 1, 0], &new_ctrlpts[k - degree, 0], sizeof(double) * dim)
        for l in range(1, degree + 1):
            idx = k - degree + l
            alpha = new_kv[k + l] - X[j]
            if abs(alpha) < 1e-8:
                memcpy(&new_ctrlpts[idx - 1, 0], &new_ctrlpts[idx, 0], sizeof(double) * dim)
            else:
                alpha = alpha / (new_kv[k + l] - knotvector[i - degree + l])
                for idx2 in range(dim):
                    new_ctrlpts[idx - 1, idx2] = alpha * new_ctrlpts[idx - 1, idx2] + (1.0 - alpha) * new_ctrlpts[idx, idx2]
        new_kv[k] = X[j]
        k -= 1
        j -= 1
    
    return np.asarray(new_ctrlpts), np.asarray(new_kv)







@cython.cdivision(True)
@cython.boundscheck(False)
cpdef void surface_deriv_cpts(int dim, int[:] degree, double[:] kv0, double[:] kv1, double[:, :] cpts, int[:] cpsize, int[:] rs, int[:] ss, int deriv_order, double[:, :, :, :, :] PKL) :
    """
    Compute the control points of the partial derivatives of a B-spline surface.

    This function calculates the control points of the partial derivatives up to a given order 
    for a surface defined by tensor-product B-splines. The derivatives are computed with respect to the 
    parametric directions (U and V) of the surface.

    Parameters
    ----------
    dim : int
        The dimensionality of the control points (e.g., 2 for 2D, 3 for 3D).

    degree : int[:]
        The degrees of the B-spline in the U and V directions.
        `degree[0]` corresponds to the degree in the U direction, and 
        `degree[1]` to the degree in the V direction.

    kv0 : double[:]
        Knot vector for the U direction.

    kv1 : double[:]
        Knot vector for the V direction.

    cpts : double[:, :, :]
        Array of control points for the B-spline surface. It has a shape 
        of (nV, nU, dim), where nU and nV are the numbers of control points 
        in the U and V directions, respectively.

    cpsize : int[:]
        Size of the control points array in each direction (U and V).
        `cpsize[0]` gives the number of control points in the U direction, 
        and `cpsize[1]` gives the number in the V direction.

    rs : int[:]
        Range of indices in the U direction to consider for derivative computation.
        `rs[0]` is the start index, and `rs[1]` is the end index.

    ss : int[:]
        Range of indices in the V direction to consider for derivative computation.
        `ss[0]` is the start index, and `ss[1]` is the end index.

    deriv_order : int
        The order of the highest derivative to compute. For example, if `deriv_order` is 2,
        the function will compute the first and second derivatives.

    PKL : double[:, :, :, :, :]
        Output array to store the computed derivatives of the control points.
        It has a shape of (du+1, dv+1, r+1, s+1, dim), where `du` and `dv` are 
        the minimum of `degree[0]` and `degree[1]` with `deriv_order`, and `r` and `s` 
        are the ranges of indices specified by `rs` and `ss`.

    Notes
    -----
    - The function first computes the U-directional derivatives of the control points
      for each V-curve and stores them in `PKL`.
    - Then, it computes the V-directional derivatives of the already U-differentiated 
      control points to complete the tensor-product differentiation.
    - Cython decorators are used to disable bounds checking and enable division optimizations
      for enhanced performance.
    """
    cdef int du = min(degree[0], deriv_order)
    cdef int dv = min(degree[1], deriv_order)
    cdef int r = rs[1] - rs[0]
    cdef int s = ss[1] - ss[0]
    cdef int i, j, k, l, d,dd
    cdef double[:, :, :] PKu = np.zeros((du + 1, r + 1, dim), dtype=np.double)
    cdef double[:, :, :] PKuv = np.zeros((dv + 1, s + 1, dim), dtype=np.double)
    cdef double[:, :] temp_cpts = np.zeros((cpsize[0], dim), dtype=np.double)

    # Control points of the U derivatives of every U-curve
    for j in range(ss[0], ss[1] + 1):
        for i in range(cpsize[0]):
            temp_cpts[i] = cpts[j + (cpsize[1] * i)]
        
        curve_deriv_cpts(degree[0], kv0, temp_cpts, du, rs[0], rs[1], PKu)

        # Copy into output as the U partial derivatives
        for k in range(du + 1):
            for i in range(r - k + 1):
                for d in range(dim):
                    PKL[k, 0, i, j - ss[0], d] = PKu[k, i, d]

    # Control points of the V derivatives of every U-differentiated V-curve
    for k in range(du):
        for i in range(r - k + 1):
            dd = min(deriv_order - k, dv)

            for j in range(s + 1):
                for d in range(dim):
                    temp_cpts[j, d] = PKL[k, 0, i, j, d]

            curve_deriv_cpts(degree[1], kv1[ss[0]:], temp_cpts, dd, 0, s, PKuv)

            # Copy into output
            for l in range(1, dd + 1):
                for j in range(s - l + 1):
                    for d in range(dim):
                        PKL[k, l, i, j, d] = PKuv[l, j, d]
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def surface_point_py(int n, int p, double[:] U, int m, int q, double[:] V, double[:, :, :] Pw, double u, double v,  bint periodic_u=0 , bint periodic_v=0, double[:] result=None):
    if result is None:
        result=np.zeros((4,))

    surface_point(n,p,U,m,q,V,Pw,u,v,periodic_u,periodic_v,&result[0])
    return np.array(result)





@cython.cdivision(True)
cdef inline int min_int(int a, int b) nogil:
    return a if a < b else b

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void surface_derivatives(int[2] degree, double[:] knots_u,  double[:] knots_v, double[:, :] ctrlpts, int[2] size, double u, double v, int deriv_order, double[:, :, :] SKL) noexcept nogil:
    """
    Compute surface derivatives for a NURBS (Non-Uniform Rational B-Splines) surface.

    This function calculates the derivatives of a NURBS surface at a given parametric point (u, v).
    The derivatives up to a specified order are computed and stored in the provided array `SKL`.

    Parameters
    ----------
    degree : int[2]
        The degrees of the NURBS surface in the u and v directions.
    knotvector : double[:, :]
        The knot vectors for the u and v directions.
    ctrlpts : double[:, :]
        The control points of the NURBS surface, stored as a 2D array.
    size : int[2]
        The number of control points in the u and v directions.
    u : double
        The parametric coordinate in the u direction where derivatives are evaluated.
    v : double
        The parametric coordinate in the v direction where derivatives are evaluated.
    deriv_order : int
        The maximum order of derivatives to be computed.
    SKL : double[:, :, :]
        An array to store the computed derivatives. `SKL[k][l][i]` contains the derivative 
        of order k in the u direction and l in the v direction for the ith coordinate 
        of the surface point.

    Notes
    -----
    - The function uses Cython directives to optimize performance, including disabling 
      bounds checking, wraparound, and enabling C division.
    - The algorithm follows the standard approach for computing NURBS surface derivatives 
      using basis function derivatives.

    Usage
    -----
    Example usage of this function might look like the following:

    .. code-block:: python

        import numpy as np

        # Define parameters
        degree = np.array([3, 3], dtype=np.int32)
        knotvector = np.array([...], dtype=np.double)  # Define appropriate knot vectors
        ctrlpts = np.array([...], dtype=np.double)    # Define appropriate control points
        size = np.array([num_ctrlpts_u, num_ctrlpts_v], dtype=np.int32)
        u = 0.5
        v = 0.5
        deriv_order = 2
        SKL = np.zeros((deriv_order + 1, deriv_order + 1, 4), dtype=np.double)

        # Call the function
        surface_derivatives(degree, knotvector, ctrlpts, size, u, v, deriv_order, SKL)

        # SKL now contains the derivatives of the surface at (u, v)

    Performance Considerations
    --------------------------
    - Memory management is done manually using malloc/free to ensure efficiency.
    - The `nogil` directive allows the function to be run in parallel threads in Python.
    
    References
    ----------
    - Piegl, L., & Tiller, W. (1997). The NURBS Book (2nd ed.). Springer.
    """
    cdef int dimension = 4
    cdef int pdimension = 2
    cdef double[2] uv
    uv[0] = u
    uv[1] = v

    cdef int[2] d
    d[0] = min_int(degree[0], deriv_order)
    d[1] = min_int(degree[1], deriv_order)

    cdef int[2] span
    cdef double* basisdrv_u
    cdef double* basisdrv_v
    cdef double[:,:] mbasisdrv_u
    cdef double[:,:] mbasisdrv_v
    cdef int idx, k, i,s, r, cu, cv, l, dd
    cdef double* temp
    cdef int basisdrv_size_u, basisdrv_size_v

    # Allocate memory and compute basis function derivatives for u and v directions
    span[0] = find_span_inline(size[0], degree[0], u, knots_u, 0)
    basisdrv_size_u = (d[0] + 1) * (degree[0] + 1)
    basisdrv_u = <double*>malloc(basisdrv_size_u * sizeof(double))
    with gil:
        mbasisdrv_u=<double[:(d[0] + 1),:(degree[0] + 1)]>basisdrv_u
   
        ders_basis_funs(span[0], u, degree[0], d[0], knots_u, mbasisdrv_u)

    span[1] = find_span_inline(size[1], degree[1], v, knots_v, 0)
    basisdrv_size_v = (d[1] + 1) * (degree[1] + 1)
    basisdrv_v = <double*>malloc(basisdrv_size_v * sizeof(double))
    with gil:
        mbasisdrv_v=<double[:(d[1] + 1),:(degree[1] + 1)]>basisdrv_v
  
        ders_basis_funs(span[1], v, degree[1], d[1], knots_v, mbasisdrv_v)

    temp = <double*>malloc((degree[1] + 1) * dimension * sizeof(double))

    for k in range(d[0] + 1):
        for s in range(degree[1] + 1):
            for i in range(dimension):
                temp[s * dimension + i] = 0.0
            for r in range(degree[0] + 1):
                cu = span[0] - degree[0] + r
                cv = span[1] - degree[1] + s
                for i in range(dimension):
                    temp[s * dimension + i] += basisdrv_u[k * (degree[0] + 1) + r] * ctrlpts[cv + (size[1] * cu)][i]

        dd = min_int(deriv_order, d[1])
        for l in range(dd + 1):
            for s in range(degree[1] + 1):
                for i in range(dimension):
                    SKL[k][l][i] += basisdrv_v[l * (degree[1] + 1) + s] * temp[s * dimension + i]

    free(temp)
    free(basisdrv_u)
    free(basisdrv_v)



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def surface_derivatives_py(tuple degree, 
            double[:] knots_u, 
            double[:] knots_v, 
            double[:, :] ctrlpts, 
            tuple size, 
            double u, double v, 
            int deriv_order=0, 
            double[:, :, :] SKL=None):
    cdef int[2] deg
    cdef int[2] sz
    deg[0]=degree[0]
    deg[1]=degree[1]
    sz[0]=size[0]
    sz[1]=size[1]
    if SKL is None:
        SKL = np.zeros((deriv_order + 1, deriv_order + 1, 4), dtype=np.double)

        # Call the function

    surface_derivatives(deg,knots_u,knots_v,ctrlpts,sz,u,v,deriv_order,SKL)
    return np.array(SKL)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def rat_surface_derivs_py(double[:, :, :] SKLw, int deriv_order=0, double[:, :, :] SKL=None):
    """
    Computes the derivatives of a rational B-spline surface.

    This function calculates the derivatives of a rational B-spline surface based on the given weighted surface derivative values (`SKLw`). 
    The results are stored in the `SKL` array. The computation takes into account the specified derivative order, 
    and optimizations are applied to speed up the calculation.

    Parameters
    ----------
    SKLw : double[:, :, :]
        A 3D array containing the weighted surface derivatives. 
        The array dimensions are (deriv_order + 1, deriv_order + 1, dimension).
    deriv_order : int
        The highest order of derivative to compute.
    SKL : double[:, :, :]
        A 3D array where the computed derivatives will be stored. 
        The array dimensions are (deriv_order + 1, deriv_order + 1, dimension - 1).

    Notes
    -----
    - The function uses binomial coefficients to adjust the derivatives according to the rational formulation.
    - The function is marked with `nogil` to allow multi-threading in Cython and to avoid Python's Global Interpreter Lock (GIL).
    - Cython compiler directives are used to disable certain checks (e.g., bounds check, wraparound)
     and enable C division for performance.

    No Return
    ---------
        The function operates in-place and does not return any values. The results are directly stored in the provided `SKL` array.

    Examples
    --------
    Consider a scenario where you have a rational B-spline surface and you need to compute its derivatives up to the second order. 
    You would call the function as follows:

    .. code-block:: python

        import numpy as np
        cdef double[:, :, :] SKLw = np.zeros((3, 3, 4))
        cdef double[:, :, :] SKL = np.zeros((3, 3, 3))
        cdef int deriv_order = 2
        
        rat_surface_derivs(SKLw, deriv_order, SKL)
    """
    if SKL is None:

    
        SKL = np.zeros((SKLw.shape[0], SKLw.shape[1], 3))
    rat_surface_derivs(SKLw,deriv_order,SKL)
    return np.array(SKL)


   
