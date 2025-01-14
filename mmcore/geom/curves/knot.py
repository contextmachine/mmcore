from __future__ import annotations

import copy
from copy import deepcopy
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import lu_solve,lu_factor
from mmcore.numeric.vectors import norm, scalar_norm

from mmcore.geom.nurbs import NURBSCurve, decompose_curve, knot_removal, knot_insertion_alpha_py, \
    knot_removal_alpha_i_py, knot_removal_alpha_j_py, NURBSSurface

from mmcore.numeric.binom import binomial_coefficient_py

def point_distance(pt1,pt2):
    return scalar_norm(np.array(pt1)-np.array(pt2))
def generate_knot(degree, num_ctrlpts, clamped=True):
    """ Generates an equally spaced knot vector.

    It uses the following equality to generate knot vector: :math:`m = n + p + 1`

    where;

    * :math:`p`, degree
    * :math:`n + 1`, number of control points
    * :math:`m + 1`, number of knots

    Keyword Arguments:

        * ``clamped``: Flag to choose from clamped or unclamped knot vector options. *Default: True*

    :param degree: degree
    :type degree: int
    :param num_ctrlpts: number of control points
    :type num_ctrlpts: int
    :return: knot vector
    :rtype: list
    """
    if degree == 0 or num_ctrlpts == 0:
        raise ValueError("Input values should be different than zero.")

    # Number of repetitions at the start and end of the array
    num_repeat = degree

    # Number of knots in the middle
    num_segments = num_ctrlpts - (degree + 1)

    if not clamped:
        # No repetitions at the start and end
        num_repeat = 0
        # Should conform the rule: m = n + p + 1
        num_segments = degree + num_ctrlpts - 1

    # First knots
    knot_vector = [0.0 for _ in range(0, num_repeat)]

    # Middle knots
    maxval = num_segments + 1.
    knot_vector += np.linspace(0.0, maxval, num_segments + 2).tolist()

    # Last knots
    knot_vector += [maxval for _ in range(0, num_repeat)]

    # Return auto-generated knot vector
    return knot_vector

def basis_function(degree, knot_vector, span, knot):
    """ Computes the non-vanishing basis functions for a single parameter.

    Cox - de Boor recursion formula.

    :param degree: degree, :math:`p`
    :type degree: int
    :param knot_vector: knot vector, :math:`U`
    :type knot_vector: list, tuple
    :param span: knot span, :math:`i`
    :type span: int
    :param knot: knot or parameter, :math:`u`
    :type knot: float
    :return: basis functions
    :rtype: list
    """
    left = [0.0 for _ in range(degree + 1)]
    right = [0.0 for _ in range(degree + 1)]
    N = [1.0 for _ in range(degree + 1)]

    for j in range(1, degree + 1):
        left[j] = knot - knot_vector[span + 1 - j]
        right[j] = knot_vector[span + j] - knot
        saved = 0.0
        for r in range(0, j):
            temp = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        N[j] = saved

    return N


def compute_knot_vector(degree, num_points, params):
    """ Computes a knot vector from the parameter list using averaging method.

    Please refer to the Equation 9.8 on The NURBS Book (2nd Edition), pp.365 for details.

    :param degree: degree
    :type degree: int
    :param num_points: number of data points
    :type num_points: int
    :param params: list of parameters, :math:`\\overline{u}_{k}`
    :type params: list, tuple
    :return: knot vector
    :rtype: list
    """
    # Start knot vector
    kv = [0.0 for _ in range(degree + 1)]

    # Use averaging method (Eqn 9.8) to compute internal knots in the knot vector
    for i in range(num_points - degree - 1):
        temp_kv = (1.0 / degree) * sum([params[j] for j in range(i + 1, i + degree + 1)])
        kv.append(temp_kv)

    # End knot vector
    kv += [1.0 for _ in range(degree + 1)]

    return kv

def build_coefficient_matrix(points, degree, knotvector, params):
    num_points = len(points)
    # Set up coefficient matrix
    matrix_a = [[0.0 for _ in range(num_points)] for _ in range(num_points)]
    for i in range(num_points):
        span = find_span_linear(degree, knotvector, num_points, params[i])

        matrix_a[i][span - degree: span + 1] = basis_function(
            degree, knotvector, span, params[i]
        )
    # Return coefficient matrix
    return matrix_a
def compute_params_curve(points, centripetal=False):
    """
    :param points: data points
    :type points: list, tuple
    :param centripetal: activates centripetal parametrization method
    :type centripetal: bool
    :return: parameter array, :math:`\\overline{u}_{k}`
    :rtype: list
    """



    num_points = len(points)

    # Calculate chord lengths
    cds = [0.0 for _ in range(num_points + 1)]
    cds[-1] = 1.0
    for i in range(1, num_points):
        distance = scalar_norm(points[i]-points[i - 1])
        cds[i] = np.sqrt(distance) if centripetal else distance

    # Find the total chord length
    d = sum(cds[1:-1])

    # Divide individual chord lengths by the total chord length
    uk = [0.0 for _ in range(num_points)]
    for i in range(num_points):
        uk[i] = sum(cds[0:i + 1]) / d

    return uk

def interpolate_curve(points, degree,  use_centripetal=False):
    """ Curve interpolation through the data points.

    Please refer to Algorithm A9.1 on The NURBS Book (2nd Edition), pp.369-370 for details.

    Keyword Arguments:
        * ``centripetal``: activates centripetal parametrization method. *Default: False*

    :param points: data points
    :type points: list, tuple
    :param degree: degree of the output parametric curve
    :type degree: int
    :return: interpolated B-Spline curve as (control_points, knots, degree) tuple.
    :rtype: tuple[list, list, int]
    """
    # Keyword arguments

    points=np.unique(points,axis=0)

    # Number of control points
    num_points = len(points)

    # Get uk
    uk = compute_params_curve(points, use_centripetal)

    # Compute knot vector
    kv = compute_knot_vector(degree, num_points, uk)

    # Do global interpolation
    matrix_a = np.array(build_coefficient_matrix( points,degree, kv, uk))
    #ctrlpts=np.linalg.lstsq(matrix_a, points)[0]
    ctrlpts = lu_solve(lu_factor(matrix_a),points)

    # Generate B-spline curve



    return np.array(ctrlpts,dtype=float),kv,degree






def normalize_knot(knot_vector):
    """ Normalizes the input knot vector to [0, 1] domain.

    :param knot_vector: knot vector to be normalized
    :type knot_vector: list, tuple

    :return: normalized knot vector
    :rtype: list
    """

    first_knot = float(knot_vector[0])
    last_knot = float(knot_vector[-1])
    denominator = last_knot - first_knot

    knot_vector_out = [(float(kv) - first_knot) / denominator
                       for kv in knot_vector]

    return knot_vector_out


def find_span_binsearch(degree, knot_vector, num_ctrlpts, knot, **kwargs):
    """ Finds the span of the knot over the input knot vector using binary search.

    Implementation of Algorithm A2.1 from The NURBS Book by Piegl & Tiller.

    The NURBS Book states that the knot span index always starts from zero, i.e. for a knot vector [0, 0, 1, 1];
    if FindSpan returns 1, then the knot is between the half-open interval [0, 1).

    :param degree: degree, :math:`p`
    :type degree: int
    :param knot_vector: knot vector, :math:`U`
    :type knot_vector: list, tuple
    :param num_ctrlpts: number of control points, :math:`n + 1`
    :type num_ctrlpts: int
    :param knot: knot or parameter, :math:`u`
    :type knot: float
    :return: knot span
    :rtype: int
    """
    # Get tolerance value
    tol = kwargs.get('tol', 10e-6)

    # In The NURBS Book; number of knots = m + 1, number of control points = n + 1, p = degree
    # All knot vectors should follow the rule: m = p + n + 1
    n = num_ctrlpts - 1
    if abs(knot_vector[n + 1] - knot) <= tol:
        return n

    # Set max and min positions of the array to be searched
    low = degree
    high = num_ctrlpts

    # The division could return a float value which makes it impossible to use as an array index
    mid = (low + high) / 2
    # Direct int casting would cause numerical errors due to discarding the significand figures (digits after the dot)
    # The round function could return unexpected results, so we add the floating point with some small number
    # This addition would solve the issues caused by the division operation and how Python stores float numbers.
    # E.g. round(13/2) = 6 (expected to see 7)
    mid = int(round(mid + tol))

    # Search for the span
    while (knot < knot_vector[mid]) or (knot >= knot_vector[mid + 1]):
        if knot < knot_vector[mid]:
            high = mid
        else:
            low = mid
        mid = int((low + high) / 2)

    return mid


def find_span_linear(degree, knot_vector, num_ctrlpts, knot, **kwargs):
    """ Finds the span of a single knot over the knot vector using linear search.

    Alternative implementation for the Algorithm A2.1 from The NURBS Book by Piegl & Tiller.

    :param degree: degree, :math:`p`
    :type degree: int
    :param knot_vector: knot vector, :math:`U`
    :type knot_vector: list, tuple
    :param num_ctrlpts: number of control points, :math:`n + 1`
    :type num_ctrlpts: int
    :param knot: knot or parameter, :math:`u`
    :type knot: float
    :return: knot span
    :rtype: int
    """
    span = degree + 1  # Knot span index starts from zero
    while span < num_ctrlpts and knot_vector[span] <= knot:
        span += 1

    return span - 1


def find_spans(degree, knot_vector, num_ctrlpts, knots, func=find_span_linear):
    """ Finds spans of a list of knots over the knot vector.

    :param degree: degree, :math:`p`
    :type degree: int
    :param knot_vector: knot vector, :math:`U`
    :type knot_vector: list, tuple
    :param num_ctrlpts: number of control points, :math:`n + 1`
    :type num_ctrlpts: int
    :param knots: list of knots or parameters
    :type knots: list, tuple
    :param func: function for span finding, e.g. linear or binary search
    :return: list of spans
    :rtype: list
    """
    spans = []
    for knot in knots:
        spans.append(func(degree, knot_vector, num_ctrlpts, knot))
    return spans


def find_multiplicity(knot, knot_vector, **kwargs):
    """ Finds knot multiplicity over the knot vector.

    Keyword Arguments:
        * ``tol``: tolerance (delta) value for equality checking

    :param knot: knot or parameter, :math:`u`
    :type knot: float
    :param knot_vector: knot vector, :math:`U`
    :type knot_vector: list, tuple
    :return: knot multiplicity, :math:`s`
    :rtype: int
    """
    # Get tolerance value
    tol = kwargs.get('tol', 10e-8)

    mult = 0  # initial multiplicity

    for kv in knot_vector:

        if abs(knot - kv) <= tol:
            mult += 1

    return mult


def knot_insertion_alpknot_removalha(u, knotvector, span, idx, leg):
    """ Computes :math:`\\alpha` coefficient for knot insertion algorithm.

    :param u: knot
    :type u: float
    :param knotvector: knot vector
    :type knotvector: tuple
    :param span: knot span
    :type span: int
    :param idx: index value (degree-dependent)
    :type idx: int
    :param leg: i-th leg of the control points polygon
    :type leg: int
    :return: coefficient value
    :rtype: float
    """
    return (u - knotvector[leg + idx]) / (knotvector[idx + span + 1] - knotvector[leg + idx])


def knot_insertion(degree, knotvector, ctrlpts, u, num=1, **kwargs)->tuple:
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
    :type ctrlpts: np.ndarray[Any, float]
    :param u: knot to be inserted (parameter to be inserted)
    :type u: float
    :return: updated control points
    :rtype: list
    """
    # Get keyword arguments

    s = find_multiplicity(u, knotvector)
    k = find_span_binsearch(degree, knotvector, len(ctrlpts), u)

    # Initialize variables
    cpt_length = len(ctrlpts)
    cpt_length_new = cpt_length + num

    # Initialize new control points array (control points may be weighted or not)
    ctrlpts_new = np.zeros((cpt_length_new,ctrlpts.shape[1]),dtype=float)
    # Initialize a local array of length p + 1
    temp = [[] for _ in range(degree + 1)]

    # Save unaltered control points
    for i in range(0, k - degree + 1):
        ctrlpts_new[i] = ctrlpts[i]
    for i in range(k - s, cpt_length):
        ctrlpts_new[i + num] = ctrlpts[i]

    # Start filling the temporary local array which will be used to update control points during knot insertion
    for i in range(0, degree - s + 1):
        temp[i] = [*ctrlpts[k - degree + i]]

    # Insert knot "num" times
    for j in range(1, num + 1):
        L = k - degree + j
        for i in range(0, degree - j - s + 1):
            alpha = knot_insertion_alpha_py(u, tuple(knotvector), k, i, L)
            if isinstance(temp[i][0], float):
                temp[i][:] = [alpha * elem2 + (1.0 - alpha) * elem1 for elem1, elem2 in zip(temp[i], temp[i + 1])]
            else:
                for idx in range(len(temp[i])):
                    temp[i][idx][:] = [alpha * elem2 + (1.0 - alpha) * elem1 for elem1, elem2 in
                                       zip(temp[i][idx], temp[i + 1][idx])]
        ctrlpts_new[L] = [*(temp[0])]
        ctrlpts_new[k + num - j - s] = [*(temp[degree - j - s])]

    # Load remaining control points
    L = k - degree + num
    for i in range(L + 1, k - s):
        ctrlpts_new[i] = [*(temp[i - L])]

    # Return control points after knot insertion
    return ctrlpts_new, knot_insertion_kv(knotvector,span=k,u=u,r=num)


def knot_insertion_kv(knotvector, u, span, r):
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


def curve_deriv_cpts(dim, degree, kv, cpts, rs, deriv_order=0):
    """ Compute control points of curve derivatives.

    Implementation of Algorithm A3.3 from The NURBS Book by Piegl & Tiller.

    :param dim: spatial dimension of the curve
    :type dim: int
    :param degree: degree of the curve
    :type degree: int
    :param kv: knot vector
    :type kv: list, tuple
    :param cpts: control points
    :type cpts: list, tuple
    :param rs: minimum (r1) and maximum (r2) knot spans that the curve derivative will be computed
    :param deriv_order: derivative order, i.e. the i-th derivative
    :type deriv_order: int
    :return: control points of the derivative curve over the input knot span range
    :rtype: list
    """
    r = rs[1] - rs[0]

    # Initialize return value (control points)
    PK = [[[None for _ in range(dim)] for _ in range(r + 1)] for _ in range(deriv_order + 1)]

    # Algorithm A3.3
    for i in range(0, r + 1):
        PK[0][i][:] = [elem for elem in cpts[rs[0] + i]]

    for k in range(1, deriv_order + 1):
        tmp = degree - k + 1
        for i in range(0, r - k + 1):
            PK[k][i][:] = [tmp * (elem1 - elem2) /
                           (kv[rs[0] + i + degree + 1] - kv[rs[0] + i + k]) for elem1, elem2
                           in zip(PK[k - 1][i + 1], PK[k - 1][i])]

    # Return control points (as a 2-dimensional list of points)
    return PK


def knot_refinement1(degree, knotvector, ctrlpts, **kwargs):
    """ Computes the knot vector and the control points of the rational/non-rational spline after knot refinement.

    Implementation of Algorithm A5.4 of The NURBS Book by Piegl & Tiller, 2nd Edition.

    The algorithm automatically find the knots to be refined, i.e. the middle knots in the knot vector, and their
    multiplicities, i.e. number of same knots in the knot vector. This is the basis of knot refinement algorithm.
    This operation can be overridden by providing a list of knots via ``knot_list`` argument. In addition, users can
    provide a list of additional knots to be inserted in the knot vector via ``add_knot_list`` argument.

    Moreover, a numerical ``density`` argument can be used to automate extra knot insertions. If ``density`` is bigger
    than 1, then the algorithm finds the middle knots in each internal knot span to increase the number of knots to be
    refined.

    **Example**: Let the degree is 2 and the knot vector to be refined is ``[0, 2, 4]`` with the superfluous knots
    from the start and end are removed. Knot vectors with the changing ``density (d)`` value will be:

    * ``d = 1``, knot vector ``[0, 1, 1, 2, 2, 3, 3, 4]``
    * ``d = 2``, knot vector ``[0, 0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2, 2.5, 2.5, 3, 3, 3.5, 3.5, 4]``

    Keyword Arguments:
        * ``knot_list``: knot list to be refined. *Default: list of internal knots*
        * ``add_knot_list``: additional list of knots to be refined. *Default: []*
        * ``density``: Density of the knots. *Default: 1*

    :param degree: degree
    :type degree: int
    :param knotvector: knot vector
    :type knotvector: list, tuple
    :param ctrlpts: control points
    :return: updated control points and knot vector
    :rtype: tuple
    """
    # Get keyword arguments
    tol = kwargs.get('tol', 10e-8)  # tolerance value for zero equality checking
    check_num = kwargs.get('check_num', True)  # enables/disables input validity checking
    knot_list = kwargs.get('knot_list', knotvector[degree:-degree])
    add_knot_list = kwargs.get('add_knot_list', list())
    density = kwargs.get('density', 1)

    # Add additional knots to be refined
    if add_knot_list:
        knot_list += list(add_knot_list)

    # Sort the list and convert to a set to make sure that the values are unique
    knot_list = sorted(set(knot_list))

    # Increase knot density
    for d in range(0, density):
        rknots = []
        for i in range(len(knot_list) - 1):
            knot_tmp = knot_list[i] + ((knot_list[i + 1] - knot_list[i]) / 2.0)
            rknots.append(knot_list[i])
            rknots.append(knot_tmp)
        rknots.append(knot_list[i + 1])
        knot_list = rknots

    # Find how many knot insertions are necessary
    X = []
    for mk in knot_list:
        s = find_multiplicity(mk, knotvector)
        r = degree - s
        X += [mk for _ in range(r)]

    # Initialize common variables
    r = len(X) - 1
    n = len(ctrlpts) - 1
    m = n + degree + 1
    a = find_span_linear(degree, knotvector, n, X[0])
    b = find_span_linear(degree, knotvector, n, X[r]) + 1

    # Initialize new control points array
    if isinstance(ctrlpts[0][0], float):
        new_ctrlpts = [[] for _ in range(n + r + 2)]
    else:
        new_ctrlpts = [[[] for _ in range(len(ctrlpts[0]))] for _ in range(n + r + 2)]

    # Fill unchanged control points
    for j in range(0, a - degree + 1):
        new_ctrlpts[j] = ctrlpts[j]
    for j in range(b - 1, n + 1):
        new_ctrlpts[j + r + 1] = ctrlpts[j]

    # Initialize new knot vector array
    new_kv = [0.0 for _ in range(m + r + 2)]

    # Fill unchanged knots
    for j in range(0, a + 1):
        new_kv[j] = knotvector[j]
    for j in range(b + degree, m + 1):
        new_kv[j + r + 1] = knotvector[j]

    # Initialize variables for knot refinement
    i = b + degree - 1
    k = b + degree + r
    j = r

    # Apply knot refinement
    while j >= 0:
        while X[j] <= knotvector[i] and i > a:
            new_ctrlpts[k - degree - 1] = ctrlpts[i - degree - 1]
            new_kv[k] = knotvector[i]
            k -= 1
            i -= 1
        new_ctrlpts[k - degree - 1] = deepcopy(new_ctrlpts[k - degree])
        for l in range(1, degree + 1):
            idx = k - degree + l
            alpha = new_kv[k + l] - X[j]
            if abs(alpha) < tol:
                new_ctrlpts[idx - 1] = deepcopy(new_ctrlpts[idx])
            else:
                print()
                alpha = alpha / (new_kv[k + l] - knotvector[i - degree + l])
                if isinstance(ctrlpts[0][0], float):
                    new_ctrlpts[idx - 1] = [alpha * p1 + (1.0 - alpha) * p2 for p1, p2 in
                                            zip(new_ctrlpts[idx - 1], new_ctrlpts[idx])]
                else:
                    for idx2 in range(len(ctrlpts[0])):
                        new_ctrlpts[idx - 1][idx2] = [alpha * p1 + (1.0 - alpha) * p2 for p1, p2 in
                                                      zip(new_ctrlpts[idx - 1][idx2], new_ctrlpts[idx][idx2])]
        new_kv[k] = X[j]
        k = k - 1
        j -= 1

    # Return control points and knot vector after refinement
    return new_ctrlpts, new_kv


def nurbs_split(self, t: float) -> tuple:
    """
    :param self:
    :param t:
    :return: a tuple of curve segments

    """

    # Keyword arguments
    span_func = find_span_binsearch  # FindSpan implementation
    insert_knot_func = insert_knot
    knotvector = self.knots
    # Find multiplicity of the knot and define how many times we need to add the knot
    ks = (
            span_func(self.degree, knotvector, len(self.control_points), t)
            - self.degree
            + 1
    )
    s = find_multiplicity(t, knotvector)
    r = self.degree - s

    # Create backups of the original curve
    temp_obj = copy.deepcopy(self)

    # Insert knot
    insert_knot_func(temp_obj, t, num=r)

    # Knot vectors
    knot_span = (
            find_span_linear(temp_obj.degree, temp_obj.knots, len(temp_obj.control_points), t) + 1
    )
    curve1_kv = list(temp_obj.knots.tolist()[0:knot_span])
    curve1_kv.append(t)
    curve2_kv = list(temp_obj.knots.tolist()[knot_span:])
    for _ in range(0, temp_obj.degree + 1):
        curve2_kv.insert(0, t)

    # Control points (use Pw if rational)
    cpts = temp_obj.control_points.tolist()
    curve1_ctrlpts = cpts[0: ks + r]
    curve2_ctrlpts = cpts[ks + r - 1:]

    # Create a new curve for the first half
    curve1 = temp_obj.__class__(
        np.array(curve1_ctrlpts), knots=np.array(curve1_kv), degree=self.degree
    )

    # Create another curve fot the second half
    curve2 = temp_obj.__class__(
        np.array(curve2_ctrlpts), knots=np.array(curve2_kv), degree=self.degree
    )

    # Return the split curves

    return curve1, curve2


def insert_knot(self, t, num=1):
    cpts, knots = knot_insertion(
        self.degree, self.knots.tolist(), self._control_points, t, num=num
    )
    self.set(control_points=np.array(cpts), knots=np.array(knots))
    return True



def compose_curves(curves):
    """
    """
    # Get keyword arguments


    # Validate input

    kv = []  # new knot vector
    cpts = []  # new control points array
    wgts = []  # new weights array
    kv_connected = []  # superfluous knots to be removed

    pdomain_end = 0.

    # Loop though the curves
    for arg in curves:
        arg=arg.copy()
        arg.normalize_knots()

        knots=arg.knots.tolist()
        control_points=arg.control_points.tolist()
        weights = arg.weights.tolist()
        # Process knot vectors
        if not kv:
            kv += list(knots[:-(arg.degree + 1)])  # get rid of the last superfluous knot to maintain split curve notation
            cpts += list(control_points)
            # Process control points
            wgts += list(weights)

        else:
            tmp_kv = [pdomain_end + k for k in knots[1:-(arg.degree + 1)]]
            kv += tmp_kv
            cpts += list(control_points[1:])
            # Process control points

            wgts += list(weights[1:])
        dstart,dend=arg.interval()
        pdomain_end += (dend-dstart)

        kv_connected.append(pdomain_end)

    # Fix curve by appending the last knot to the end
    kv += [pdomain_end for _ in range(arg.degree + 1)]
    # Remove the last knot from knot insertion list
    kv_connected.pop()
    cp=np.zeros((len(cpts),4))
    cp[:,:3]=cpts
    cp[:, 3]=wgts
    return np.array(kv), cp, np.array(kv_connected)
def _compose_curves_no_knots_updates(curves):
    """
    """
    # Get keyword arguments


    # Validate input

    kv = []  # new knot vector
    cpts = []  # new control points array
    wgts = []  # new weights array
    kv_connected = []  # superfluous knots to be removed

    pdomain_end = curves[0].interval()[0]

    # Loop though the curves
    for arg in curves:
        #arg=arg.copy()
        #arg.normalize_knots()

        knots=arg.knots.tolist()
        control_points=arg.control_points.tolist()
        weights = arg.weights.tolist()
        # Process knot vectors
        if not kv:
            kv += list(knots[:-(arg.degree + 1)])  # get rid of the last superfluous knot to maintain split curve notation
            cpts += list(control_points)
            # Process control points
            wgts += list(weights)

        else:
            tmp_kv = [k for k in knots[1:-(arg.degree + 1)]]
            kv += tmp_kv
            cpts += list(control_points[1:])
            # Process control points

            wgts += list(weights[1:])
        dstart,dend=arg.interval()
        pdomain_end=dend

        kv_connected.append(pdomain_end)

    # Fix curve by appending the last knot to the end
    kv += [pdomain_end for _ in range(arg.degree + 1)]
    # Remove the last knot from knot insertion list
    kv_connected.pop()
    cp=np.zeros((len(cpts),4))
    cp[:,:3]=cpts
    cp[:, 3]=wgts
    return np.array(kv), cp, np.array(kv_connected)

def degree_elevate_curve(curve:NURBSCurve, num:int=1):
    """ Applies degree elevation and degree reduction algorithms to spline geometries.

    :param obj: spline geometry
    :type obj: abstract.SplineGeometry
    :param param: operation definition
    :type param: list, tuple
    :return: updated spline geometry
    """

    # Start curve degree manipulation operations


        # Find multiplicity of the internal knots
    int_knots = set(curve.knots[curve.degree + 1:-(curve.degree + 1)])
    mult_arr = []
    for ik in int_knots:
        s = find_multiplicity(ik, curve.knots)
        mult_arr.append(s)


    # Decompose the input by knot insertion
    crv_list = decompose_curve(curve)


    # If parameter is positive, apply degree elevation. Otherwise, apply degree reduction
    crv_list_new=[]
    # Loop through to apply degree elevation
    for crv in crv_list:

            new_cpts,new_deg = degree_elevation_bez( crv.control_pointsw,crv.degree, num=num)
            new_kv=np.zeros((crv.knots.shape[0]+num*2,))

            new_kv[:num]=crv.knots[0]
            new_kv[-num:] = crv.knots[-1]

            new_kv[num:][:-num]=crv.knots

            crv_new=NURBSCurve(new_cpts,new_deg,knots=new_kv)

            crv_list_new.append(crv_new)




    # Compute new degree
    nd = curve.degree + num

    # Number of knot removals
    num_knots_removals = curve.degree + 1


    # Link curves together (reverse of decomposition)
    kv, cpts, knots_connected = _compose_curves_no_knots_updates(crv_list_new)

    # Organize control points (if necessary)
    #ctrlpts = compatibility.combine_ctrlpts_weights(cpts, ws) if obj.rational else cpts






    return NURBSCurve(np.array(cpts), nd, knots=np.array(kv))


from mmcore.geom.nurbs import reverse_curve



def degree_elevation_bez(points:NDArray[float], degree:int, num:int=1)->tuple[NDArray[float], int]:
    """
    Elevates the degree of a Bézier curve by a specified number of times.

    This function computes the degree elevation for Bézier curves, increasing the
    degree while keeping the curve's shape intact. The degree elevation increases
    the number of control points, but ensures the resulting Bézier curve is equivalent
    to the original. This operation is repeated `num` times.

    :param points: Control points of the Bézier curve. The array should have a shape
        of (degree + 1, n), where `degree` is the degree of the Bézier curve and `n`
        is the dimension of each control point.
    :param degree: Degree of the input Bézier curve. Must be consistent with the number
        of control points provided in `points`.
    :param num: Number of times to elevate the degree of the Bézier curve. Default is 1.
    :return: A tuple consisting of the new control points and the elevated degree:
        - new_points (NDArray[float]): Array of the Bézier curve's control points after
          degree elevation.
        - elevated_degree (int): The new degree of the Bézier curve after the operation.
    :raises ValueError: If the number of provided control points does not match
        degree + 1, since piecewise geometry is not supported by this operation.
    """
    if points.shape[0] != (degree + 1):
        raise ValueError(
            "Current operation do not support piecewise geometry, control points count must be equal degree+1.")
    new_points_num = degree + 1 + num
    new_points = np.zeros((new_points_num, points.shape[1]))

    for i in range(0, new_points_num):
        start = max(0, (i - num))
        end = min(degree, i)
        for j in range(start, end + 1):
            coeff = (binomial_coefficient_py(degree, j) * binomial_coefficient_py(num,
                                                                                  (i - j))) / binomial_coefficient_py(
                (degree + num), i)

            new_points[i] += (coeff * points[j])


    return new_points, degree + num



import numpy as np


def find_knot_span(n, p, u, U):
    """
    Find the knot span index for a given parameter value.

    Parameters:
    -----------
    n : int
        Number of control points minus 1
    p : int
        Degree of the curve
    u : float
        Parameter value
    U : array-like
        Knot vector

    Returns:
    --------
    int : Index of the knot span
    """
    # Special case for u at the end of the curve
    if u == U[n + 1]:
        return n

    # Binary search to find knot span
    low = p
    high = n + 1
    mid = (low + high) // 2

    while u < U[mid] or u >= U[mid + 1]:
        if u < U[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2

    return mid


def knot_refinement2(n, p, U, Pw, X):
    """
    Refine a B-spline curve by inserting new knots.

    Parameters:
    -----------
    n : int
        Number of control points minus 1
    p : int
        Degree of the curve
    U : array-like
        Current knot vector
    Pw : array-like
        Current control points
    X : array-like
        New knots to insert (must be sorted)

    Returns:
    --------
    tuple : (new control points, new knot vector)
    """
    U = np.array(U, dtype=float)
    Pw = np.array(Pw, dtype=float)
    X = np.array(X, dtype=float)

    # Get number of new knots
    r = len(X) - 1
    m = n + p + 1

    # Initialize new arrays
    nq = n + r + 1
    Qw = np.zeros((nq + 1, Pw.shape[1]))
    Ubar = np.zeros(len(U) + len(X))

    # Save unaltered control points and knots
    for i in range(0, p + 1):
        Qw[i] = Pw[i]
    for i in range(n - p, n + 1):
        Qw[i + r + 1] = Pw[i]
    for i in range(0, p + 1):
        Ubar[i] = U[0]
    for i in range(len(U) - p - 1, len(U)):
        Ubar[i + r + 1] = U[-1]

    # Initialize variables for refinement
    i = p
    j = r
    k = p + 1

    # Initialize local arrays
    Rw = np.zeros((p + 1, Pw.shape[1]))

    # Main refinement loop
    while j >= 0:  # Insert knots
        while X[j] <= U[i] and i > 0:
            Qw[k] = Pw[i - 1]
            Ubar[k] = U[i]
            k -= 1
            i -= 1

        # Save current control point
        for l in range(p, -1, -1):
            Rw[l] = Pw[i - p + l]

        # Insert knot
        for l in range(1, p + 1):
            if i - p + l <= 0:
                continue

            # Compute alpha
            alpha = (X[j] - U[i - p + l]) / (U[i + l] - U[i - p + l])

            # Update control points
            Rw[l] = alpha * Rw[l] + (1.0 - alpha) * Rw[l - 1]

        # Load refined control points into final array
        Qw[k] = Rw[p]
        Ubar[k] = X[j]

        k -= 1
        j -= 1

    # Set remaining control points and knots
    while i > 0:
        Qw[k] = Pw[i - 1]
        Ubar[k] = U[i]
        k -= 1
        i -= 1

    return Qw, Ubar


def refine_curve(curve, new_knots):
    """
    Refine a NURBS curve by inserting new knots.

    Parameters:
    -----------
    curve : NURBSCurve
        The curve to refine
    new_knots : array-like
        New knots to insert (must be sorted)

    Returns:
    --------
    NURBSCurve : Refined curve
    """
    curve=curve.copy()


    old=np.copy(curve.knots)
    for i,k in enumerate(new_knots):

        if old[0]>k:


            curve.insert_knot(k,1)
        else:
            old=np.delete(old,0)

    return curve

def join_two_curves(curve1:NURBSCurve, curve2:NURBSCurve,tol:float=1e-5):
    start_1=np.array(curve1.start())
    start_2 = np.array(curve2.start())
    end_1 = np.array(curve1.end())
    end_2 =np.array( curve2.end())
    mind=np.inf
    ix=()
#, 1,0 end, start
    for i,v1 in enumerate([start_1,end_1]):
        for j,v2 in  enumerate([start_2, end_2]):
            if np.linalg.norm(v1-v2)<tol:
                ix=(i,j)
    if ix==(1,0):
        ...
    elif ix==(0,0):
        curve1=reverse_curve(curve1)
    elif ix == (0, 1):
        curve1 = reverse_curve(curve1)
        curve2 = reverse_curve(curve2)
    elif ix == (1, 1):

        curve2 = reverse_curve(curve2)

    if curve1.degree != curve2.degree:
        if curve1.degree < curve2.degree:
            # Elevate first curve to match second
            t = curve2.degree - curve1.degree
            curve1 = degree_elevate_curve(curve1, t)

        else:
            # Elevate second curve to match first
            t = curve1.degree - curve2.degree
            curve2 = degree_elevate_curve(curve2, t)

    degree = curve1.degree  # Now both curves have same degree

    # Get control points and weights
    cpts1 = curve1.control_points
    cpts2 = curve2.control_points

    # Create new knot vector by joining and scaling
    knots1 = curve1.knots
    knots2 = curve2.knots

    # Scale second knot vector to start where first one ends


    # Combine knot vectors
    print(knots1,knots1.max()+knots2-knots2.min())
    new_knots = np.concatenate([knots1,knots1.max()+knots2-knots2.min()])

    # Combine control points, merging the connecting points
    new_cpts = np.vstack([
        cpts1[:-1],  # Exclude last point of first curve
        cpts2  # Include all points of second curve
    ])
    return NURBSCurve(new_cpts, degree, new_knots)


import numpy as np


def merge_knot_vectors(knots1, knots2):
    """
    Merge two knot vectors, maintaining multiplicities
    """
    # Combine unique knots from both vectors
    all_knots = np.unique(np.concatenate([knots1, knots2]))

    # For each knot, take maximum multiplicity from either vector
    result = []
    for knot in all_knots:
        mult1 = np.sum(np.isclose(knots1, knot))
        mult2 = np.sum(np.isclose(knots2, knot))
        result.extend([knot] * max(mult1, mult2))

    return np.array(result)


