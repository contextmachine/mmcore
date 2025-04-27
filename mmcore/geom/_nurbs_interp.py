import numpy as np

from scipy.linalg import lu_solve,lu_factor
from numpy.typing import NDArray
from mmcore.geom._nurbs_eval import _find_span_linear,compute_basis_function_derivatives_np,bspline_basis
from mmcore.geom._nurbs_knots import generate_knots
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
        span = _find_span_linear(degree, knotvector, num_points, params[i])

        matrix_a[i][span - degree: span + 1] = compute_basis_function_derivatives_np(
            degree, knotvector, span, params[i],0
        )[0]
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


        distance = np.linalg.norm(points[i]-points[i - 1])
        cds[i] = np.sqrt(distance) if centripetal else distance

    # Find the total chord length
    d = np.sum(cds[1:-1])

    # Divide individual chord lengths by the total chord length
    uk = [0.0 for _ in range(num_points)]
    for i in range(num_points):
        uk[i] = np.sum(cds[0:i + 1]) / d

    return uk,d

from mmcore.numeric.vectors import norm
def _remove_adjacent_duplicates(points, tol=1e-5):
    """
    Remove adjacent duplicates (or almost duplicates) from an array of points.

    Parameters:
        points (np.ndarray): A 2D numpy array of shape (n_points, 3) where each row is [x, y, z].
        tol (float): Tolerance for comparing points. Two points are considered equal if
                     the Euclidean distance between them is <= tol.

    Returns:
        np.ndarray: The array of points with consecutive duplicates removed.
    """
    if points.shape[0] == 0:
        return points

    # Compute the Euclidean distance between consecutive points.
    # np.diff(points, axis=0) gives an array of differences between consecutive rows.
    diffs = np.array(norm(np.diff(points, axis=0)))

    # Create a boolean mask:
    # Always keep the first point.
    # For each subsequent point, keep it only if the distance from the previous point is > tol.
    mask = np.concatenate(([True], diffs > tol))

    return points[mask]

from typing import Literal
def interpolate_curve(points, degree,  use_centripetal=False, method:Literal['lstsq','lu']='lstsq',remove_duplicates:bool=False,tol=1e-6):
    """ Curve interpolation through the data points.

    Please refer to Algorithm A9.1 on The NURBS Book (2nd Edition), pp.369-370 for details.

    """
    # Keyword arguments

    if remove_duplicates:
        points=_remove_adjacent_duplicates(points,tol)
    # Number of control points
    num_points = len(points)

    # Get uk
    uk,total_chord_length = compute_params_curve(points, use_centripetal)

    # Compute knot vector
    kv = compute_knot_vector(degree, num_points, uk)

    # Do global interpolation
    matrix_a = np.array(build_coefficient_matrix( points,degree, kv, uk))
    if method=='lstsq':
        control_points=np.linalg.lstsq(matrix_a, points)[0]
    else:
        control_points = lu_solve(lu_factor(matrix_a),points)
    #print(matrix_a)




    return np.array(control_points,dtype=float),np.array(kv)


import numpy as np


def chord_length_params(points):
    """
    Compute normalized chord-length parameters for a list of points.

    points: array-like, shape (n, dim)
    Returns: 1D numpy array of parameters in [0,1]
    """
    points = np.asarray(points)
    n = len(points)
    u = np.zeros(n)
    for i in range(1, n):
        u[i] = u[i - 1] + np.linalg.norm(points[i] - points[i - 1])
    if u[-1] == 0:
        return u
    return u / u[-1]


def _compute_knot_vector_v2(u, degree, n_ctrlpts):
    """
    Compute a knot vector based on the data parameter values u (from chord-length)
    using the standard knot averaging formula for interpolation, and—if more
    control points than data points are desired—insert extra (uniform) knots.

    Parameters:
        u         : 1D array of parameter values for the data points.
        degree    : degree of the B-spline curve.
        n_ctrlpts : desired number of control points.
                    For example, if you have n_data points and want extra degrees
                    of freedom, one common choice is: n_ctrlpts = n_data + (degree-1)

    Returns:
        knot_vector: 1D numpy array of length n_ctrlpts + degree + 1.
    """
    n_data = len(u)
    # First, compute the standard knot vector for n_data control points.
    U_std = np.zeros(n_data + degree + 1)
    U_std[:degree + 1] = u[0]
    U_std[-(degree + 1):] = u[-1]
    for j in range(1, n_data - degree):
        U_std[j + degree] = np.sum(u[j:j + degree]) / degree

    # If no augmentation is needed, return the standard vector.
    if n_ctrlpts == n_data:
        return U_std
    else:

        return generate_knots(n_ctrlpts,degree,(u[0],u[-1]) )




def construct_basis_matrix(u_vals, degree, knot_vector, n_ctrlpts):
    """
    Construct the B-spline basis matrix.

    Each row corresponds to one parameter value from u_vals and each column to a control point.
    """
    m = len(u_vals)
    B = np.zeros((m, n_ctrlpts))
    for i, u in enumerate(u_vals):
        for j in range(n_ctrlpts):
            B[i, j] = bspline_basis(j, degree, knot_vector, u)
    return B


def fair_interpolate_curve(points, degree, lambda_reg=1e-3)->tuple[NDArray[float],NDArray[float]]:
    """
    :param  points: array-like, shape (n_data, dim). These are the data (e.g. curve isoline positions)
    :param degree: degree of the B-spline curve (for example, 3 for a cubic curve)
    :param lambda_reg : regularization (fairness) weight. Increase to favor smoother (fairer)
                     curves at the expense of less exact interpolation of the interior points.

    :returns:  Tuple with control_points and knot_vector.
        control_points: array of shape (n_ctrlpts, dim) (the computed control net for the fair interpolant)
        knot_vector   : the knot vector (a 1D array of length n_ctrlpts + degree + 1)


    Given a set of points to be interpolated, compute a fair B-spline interpolation
    that (i) uses knot-averaging to parameterize the interpolation, (ii) augments the
    number of control points to allow extra freedom (e.g., n_ctrlpts = n_data + (degree-1)),
    and (iii) selects from the infinite interpolation solutions the one that minimizes
    a fairness penalty based on second finite differences (a proxy for curvature).

    The implementation enforces the end points exactly and solves for the free interior control points
    by minimizing the combined interpolation error (at the given parameter values) and a fairness term.
    The fairness conditions (in the form of second finite differences) are expressed as:

       d[i-1] - 2*d[i] + d[i+1] ≈ 0

    for the (free) control points.
    """
    points = np.asarray(points)
    n_data, dim = points.shape
    # For augmented interpolation (to allow for extra fairing freedom),
    # choose number of control points by adding (degree - 1) extra rows.
    n_ctrlpts = n_data + (degree - 1)

    # 1. Determine parameter values via chord-length.
    u_vals = chord_length_params(points)

    # 2. Compute the knot vector (augmented via uniform insertion) using knot-averaging.
    knot_vector = _compute_knot_vector_v2(u_vals,degree,n_ctrlpts)

    # 3. Build the B-spline basis matrix evaluated at the data parameters.
    B = construct_basis_matrix(u_vals, degree, knot_vector, n_ctrlpts)

    # 4. Enforce interpolation at the endpoints exactly.
    #    Fix control point d0 and d_end.
    d0 = points[0]
    d_end = points[-1]

    # The remaining free unknown control points are indices 1 ... n_ctrlpts-2.
    N_free = n_ctrlpts - 2
    # For each interpolation condition (for each data point), the condition is:
    #    B[i,0]*d0 + sum_{j=1}^{n_ctrlpts-2} B[i,j]*x_{j-1} + B[i,n_ctrlpts-1]*d_end = points[i]
    # Isolate the unknowns:
    A_interp = B[:, 1:-1]  # shape: (n_data, N_free)
    b_interp = points - (B[:, [0]] * d0 + B[:, [-1]] * d_end)  # shape: (n_data, dim)

    # 5. Build a fairness (smoothness) term based on second finite differences.
    #    For a control net d0, d1, ..., d_{n_ctrlpts-1}, the second difference at index i is:
    #         d[i-1] - 2*d[i] + d[i+1]
    #    We enforce these to be (nearly) zero.
    #    In our free unknowns (x = [d1, d2, …, d_{n_ctrlpts-2}]), write equations:
    #      For i = 1 (first free point):  d0 - 2*d1 + d2 = 0   --> -2*x0 + x1 = -d0.
    #      For i = 2,..., n_ctrlpts-3 (interior free points):  d[i-1] - 2*d[i] + d[i+1] = 0.
    #      For i = n_ctrlpts-2 (last free point):  d[n_ctrlpts-3] - 2*d[n_ctrlpts-2] + d_end = 0
    #         --> x[-2] - 2*x[-1] = -d_end.
    N_eq = N_free  # We build one fairness equation per free unknown.
    F = np.zeros((N_eq, N_free))
    f_const = np.zeros((N_eq, dim))
    if N_free > 0:
        # Equation for the first free unknown:
        if N_free >= 2:
            F[0, 0] = -2
            F[0, 1] = 1
            f_const[0] = -d0
        else:
            # If there is only one free unknown, combine the two endpoints.
            F[0, 0] = -2
            f_const[0] = - (d0 + d_end)
        # Equations for interior free unknowns.
        for i in range(1, N_free - 1):
            F[i, i - 1] = 1
            F[i, i] = -2
            F[i, i + 1] = 1
            f_const[i] = 0  # no fixed contribution in the interior
        # Equation for the last free unknown:
        if N_free >= 2:
            F[N_free - 1, N_free - 2] = 1
            F[N_free - 1, N_free - 1] = -2
            f_const[N_free - 1] = -d_end

    # 6. Combine the interpolation conditions and fairness regularization into one augmented system.
    sqrt_lambda = np.sqrt(lambda_reg)
    A_aug = np.vstack([A_interp, sqrt_lambda * F])
    b_aug = np.vstack([b_interp, sqrt_lambda * f_const])

    # Solve the resulting least-squares problem for the free control points.
    # (Since the interpolation is underdetermined, the fairness term selects a unique solution.)
    x, residuals, rank, s = np.linalg.lstsq(A_aug, b_aug, rcond=None)

    # 7. Reassemble the full control net.
    control_points = np.vstack([d0, x, d_end])

    return control_points, knot_vector

