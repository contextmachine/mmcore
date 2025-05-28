cimport cython
cimport numpy as cnp
import numpy as np


from libc.stdlib cimport malloc,free
from libcpp.vector cimport vector
from libcpp.cmath cimport fabs,fmin,fmax,round
from libc.string cimport memcpy
from libcpp.limits cimport numeric_limits
cimport mmcore.geom.knots
cnp.import_array()

cpdef knot_insertion_kv(knotvector, u, span, r):
    """ Computes the knot vector of the rational/non-rational spline after knot insertion.

    Part of Algorithm A5.1 of The NURBS Book by Piegl & Tiller, 2nd Edition.

    :param knotvector: knot vector
    :type knotvector: list, tuple
    :param u: knot
    :type u: float
    :param span: knot span
    :type span: int
    :param r: number of knot insertions
    :type r: int
    :return: updated knot vector
    :rtype: list
    """
    # Initialize variables
    kv_size = len(knotvector)
    kv_updated = [0.0 for _ in range(kv_size + r)]
    
    # Compute new knot vector
    for i in range(0, span + 1):
        kv_updated[i] = knotvector[i]
    for i in range(1, r + 1):
        kv_updated[span + i] = u
    for i in range(span + 1, kv_size):
        kv_updated[i + r] = knotvector[i]
    
    # Return the new knot vector
    return kv_updated

cpdef knot_insertion(int degree, double[:] knotvector, double[:,:] ctrlpts, double u, int num: int, int span, int s):
    """ Computes the control points of the rational/non-rational spline after knot insertion.

    Part of Algorithm A5.1 of The NURBS Book by Piegl & Tiller, 2nd Edition.

    Keyword Arguments:
        * ``num``: number of knot insertions. *Default: 1*
        * ``s``: multiplicity of the knot. *Default: computed via :func:`.find_multiplicity`*
        * ``span``: knot span. *Default: computed via :func:`.find_span_linear`*

    :param degree: degree
    :type degree: int
    :param knotvector: knot vector
    :type knotvector: list, tuple
    :param ctrlpts: control points
    :type ctrlpts: list
    :param u: knot to be inserted
    :type u: float
    :return: updated control points
    :rtype: list
    """
    # Get keyword arguments
    
    #s = find_multiplicity(u, knotvector) if s is None else s  # multiplicity
    
    #k = _find_span_linear(degree, knotvector, len(ctrlpts), u) if span is None else span  # knot span
    cdef int k = span
    # Initialize variables
    cdef int npt = ctrlpts.shape[0]
    cdef int nq = npt + num
    
    # Initialize new control points array (control points may be weighted or not)
    #ctrlpts_new = [[] for _ in range(nq)]
    cdef double[:,:] ctrlpts_new= np.empty((nq,ctrlpts.shape[1]))
    
    cdef int order=degree + 1
    
    # Initialize a local array of length p + 1
    #temp = [[] for _ in range(degree + 1)]
    
    cdef double[:] temp= np.empty((order,ctrlpts.shape[1]))
    cdef int i,j,L

    # Save unaltered control points
    for i in range(0, k - degree + 1):
        ctrlpts_new[i] = ctrlpts[i]
        
    for i in range(k - s, npt):
        ctrlpts_new[i + num] = ctrlpts[i]
    
    # Start filling the temporary local array which will be used to update control points during knot insertion
    for i in range(0, degree - s + 1):
        temp[i] = deepcopy(ctrlpts[k - degree + i])
    
    # Insert knot "num" times
    for j in range(1, num + 1):
        L = k - degree + j
        for i in range(0, degree - j - s + 1):
            
            alpha = knot_insertion_alpha(u, tuple(knotvector), k, i, L)
            if isinstance(temp[i][0], float):
                temp[i][:] = [alpha * elem2 + (1.0 - alpha) * elem1 for elem1, elem2 in zip(temp[i], temp[i + 1])]
            else:
                for idx in range(len(temp[i])):
                    temp[i][idx][:] = [alpha * elem2 + (1.0 - alpha) * elem1 for elem1, elem2 in
                                       zip(temp[i][idx], temp[i + 1][idx])]
        ctrlpts_new[L] = deepcopy(temp[0])
        ctrlpts_new[k + num - j - s] = deepcopy(temp[degree - j - s])
    
    # Load remaining control points
    L = k - degree + num
    for i in range(L + 1, k - s):
        ctrlpts_new[i] = deepcopy(temp[i - L])
    
    # Return control points after knot insertion
    return ctrlpts_new
