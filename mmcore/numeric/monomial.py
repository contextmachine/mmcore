# mmcore/numeric/monomial.py
from __future__ import annotations
import numpy as np
from scipy.special import comb
from itertools import product
from mmcore.geom.nurbs import NURBSSurface
from mmcore.geom.surfaces import Surface
from mmcore.numeric.binom import binomial_coefficient_py
from functools import lru_cache
@lru_cache(maxsize=None)
def bpmat(n, method="mmcore"):

    """Create Bernstein to monomial conversion matrix.

    """
    method=comb if method=="scipy" else binomial_coefficient_py
    return np.array(
        [
            [
                (-1) ** (j - k) * method(n, j) * method(j, k) if j >= k else 0
                for k in range(n + 1)
            ]
            for j in range(n + 1)
        ]
    )



def bezier_to_monomial(control_points, bmethod="mmcore"):
    """
    Convert Bezier curve control points to monomial coefficients.

    Args:
        control_points (np.ndarray): A 3D array of shape (n, m, dim) containing the
            control points of the Bezier curve.
        bmethod (str): A string indicating the basis transformation method to use.
            Default is "mmcore".

    Returns:
        np.ndarray: A 3D array of shape (n, m, dim) containing the monomial
            coefficients.

    Raises:
        ValueError: If the shape of control_points does not correspond to a 3D array.

    Usage Example:
        >>> control_points = np.array([[[0, 0.,1.], [1, 2, 3.]], [[2, 3, 1.], [4, 5, 7.]]])
        >>> monomial_coeffs = bezier_to_monomial(control_points, bmethod="mmcore")
        >>> print(monomial_coeffs)
        [[[0. 0. 1.]
          [1. 2. 2.]]
         [[2. 3. 0.]
          [1. 0. 4.]]]
    """
    n, m, dim = control_points.shape
    Mu = bpmat(n - 1,method=bmethod)
    Mv = bpmat(m - 1,method=bmethod)

    monomial_coeffs = np.zeros((n, m, dim))
    for d in range(dim):
        monomial_coeffs[:, :, d] = Mu @ control_points[:, :, d] @ Mv.T

    return monomial_coeffs


def monomial_to_bezier(monomial_coeffs, bmethod="mmcore"):
    """
    Converts a tensor of monomial coefficients to Bezier control points.

    This function transforms monomial coefficients into corresponding Bezier control points
    using specified basis transformation methods.

    Args:
        monomial_coeffs (ndarray): A 3D tensor of shape (n, m, dim) containing the monomial coefficients.
        bmethod (str, optional): The basis transformation method to use. Default is "mmcore".

    Returns:
        ndarray: A 3D tensor of shape (n, m, dim) containing the Bezier control points.

    Raises:
        ValueError: If the input monomial_coeffs tensor does not have the required shape.

    Usage Example:
        >>> monomial_coeffs = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        >>> control_points = monomial_to_bezier(monomial_coeffs, bmethod="mmcore")
    """
    n, m, dim = monomial_coeffs.shape
    #print(n - 1)
    #print(m-1)
    Mu_inv = np.linalg.inv(bpmat(n - 1, method=bmethod))
    Mv_inv = np.linalg.inv(bpmat(m - 1, method=bmethod))

    control_points = np.zeros((n, m, dim))
    for d in range(dim):
        control_points[:, :, d] = Mu_inv @ monomial_coeffs[:, :, d] @ Mv_inv.T

    return control_points



def homogeneous_monomial_to_rational_bezier(homogeneous_coeffs):
    """
    Convert homogeneous monomial coefficients to rational Bézier patch control points.

    Parameters:
    homogeneous_coeffs : numpy.ndarray
        4D array of shape (n, m, 4) representing the coefficients of the homogeneous monomial form.

    Returns:
    numpy.ndarray
        4D array of shape (n, m, 4) representing the control points of the rational Bézier patch.
        The fourth component of each control point is its weight.
    """
    n, m, _ = homogeneous_coeffs.shape

    # Convert each component (including the weight) to Bézier form
    bezier_coeffs = np.zeros((n, m, 4))
    for d in range(4):
        bezier_coeffs[:, :, d] = monomial_to_bezier(
            homogeneous_coeffs[:, :, d : d + 1]
        )[:, :, 0]

    # Normalize the control points by the weight
    rational_control_points = np.zeros((n, m, 4))
    rational_control_points[:, :, 3] = bezier_coeffs[:, :, 3]  # Weights
    for d in range(3):
        rational_control_points[:, :, d] = (
            bezier_coeffs[:, :, d] / bezier_coeffs[:, :, 3]
        )

    return rational_control_points




def evaluate_bezier_patch(control_points, u, v):
    """Evaluate a Bézier patch at given parameter values."""
    n, m, dim = control_points.shape
    result = np.zeros(dim)
    for i in range(n):
        for j in range(m):
            b_u = comb(n - 1, i) * (u**i) * ((1 - u) ** (n - 1 - i))
            b_v = comb(m - 1, j) * (v**j) * ((1 - v) ** (m - 1 - j))
            result += control_points[i, j] * b_u * b_v
    return result

def evaluate_rational_bezier_patch(control_points, weights, u, v):
    """
    Evaluate a rational Bézier patch at given parameter values.
    Parameters:
    control_points (numpy.ndarray): Shape (n, m, dim) array of control points
    weights (numpy.ndarray): Shape (n, m) array of weights
    u, v (float): Parameter values, each in the range [0, 1]
    Returns:
    numpy.ndarray: The point on the rational Bézier patch at (u, v)
    """
    n, m, dim = control_points.shape
    numerator = np.zeros(dim)
    denominator = 0.0
    for i in range(n):
        for j in range(m):
            b_u = comb(n - 1, i) * (u**i) * ((1 - u) ** (n - 1 - i))
            b_v = comb(m - 1, j) * (v**j) * ((1 - v) ** (m - 1 - j))
            basis = b_u * b_v * weights[i, j]
            numerator += control_points[i, j] * basis
            denominator += basis
    return numerator / denominator

def evaluate_monomial2d(coeffs, u, v):
    """Evaluate a monomial form surface at (u,v)."""
    n, m, dim = coeffs.shape
    result = np.zeros(dim)
    for i in range(n):
        for j in range(m):
            for d in range(dim):
                result[d] += coeffs[i,j,d] * (u**i) * (v**j)
    return result

def cross_product_monomial(a_coeffs, b_coeffs):
    """
    Compute the cross product of two polynomial vectors in monomial form.

    Parameters:
    a_coeffs, b_coeffs : numpy.ndarray
        3D arrays of shape (n, m, 3) representing the coefficients of the monomial forms.

    Returns:
    numpy.ndarray
        Coefficients of the cross product in monomial form.
    """
    n1, m1, _ = a_coeffs.shape
    n2, m2, _ = b_coeffs.shape
    n, m = n1 + n2 - 1, m1 + m2 - 1

    result = np.zeros((n, m, 3))
    print(list(product(range(n1), range(m1), range(n2), range(m2))))

    for i1, j1, i2, j2 in product(range(n1), range(m1), range(n2), range(m2)):
        i, j = i1 + i2, j1 + j2
        result[i, j, 0] += (
            a_coeffs[i1, j1, 1] * b_coeffs[i2, j2, 2]
            - a_coeffs[i1, j1, 2] * b_coeffs[i2, j2, 1]
        )
        result[i, j, 1] += (
            a_coeffs[i1, j1, 2] * b_coeffs[i2, j2, 0]
            - a_coeffs[i1, j1, 0] * b_coeffs[i2, j2, 2]
        )
        result[i, j, 2] += (
            a_coeffs[i1, j1, 0] * b_coeffs[i2, j2, 1]
            - a_coeffs[i1, j1, 1] * b_coeffs[i2, j2, 0]
        )

    return result



def monomial_partial_derivatives(coeffs):
    """
    Compute partial derivatives of a bivariate polynomial in monomial form.

    Parameters:
    coeffs : numpy.ndarray
        3D array of shape (n, m, 3) representing the coefficients of the monomial form.
        n and m are the degrees of u and v respectively, and 3 is for x, y, z coordinates.

    Returns:
    du_coeffs, dv_coeffs : tuple of numpy.ndarray
        Partial derivatives with respect to u and v, respectively.
    """
    n, m, dim = coeffs.shape

    # Partial derivative with respect to u
    du_coeffs = np.zeros((n - 1, m, dim))
    for i in range(1, n):
        du_coeffs[i - 1, :, :] = i * coeffs[i, :, :]

    # Partial derivative with respect to v
    dv_coeffs = np.zeros((n, m - 1, dim))
    for j in range(1, m):
        dv_coeffs[:, j - 1, :] = j * coeffs[:, j, :]

    return du_coeffs, dv_coeffs


def normal_vector_monomial(coeffs):
    """
    Compute the unnormalized normal vector of a surface in monomial form.

    Parameters:
    coeffs : numpy.ndarray
        3D array of shape (n, m, 3) representing the coefficients of the monomial form.

    Returns:
    numpy.ndarray
        Coefficients of the unnormalized normal vector in monomial form.
    """
    du_coeffs, dv_coeffs = monomial_partial_derivatives(coeffs)
    return cross_product_monomial(du_coeffs, dv_coeffs)


class Monomial2D(Surface):
    """
    Represents a 2D monomial surface with various conversion and operation methods.
    """
    def __init__(self, coeffs):
        super().__init__()

        self.coeffs = np.array(coeffs,dtype=float)
    def evaluate(self,uv):
        return evaluate_monomial2d(self.coeffs, uv[0], uv[1])

    def to_bernstein_basis(self):
        """
        Converts the polynomial coefficients from the monomial basis to the Bernstein basis.

        Returns:
            list: A list of coefficients in the Bernstein basis.

        Example:
            >>> m = Monomial2D(np.array([
            ...     [[0, 0, 0], [0, 0, 0], [1, 3, 1]],
            ...     [[0, -1, -2], [2, 0, 0], [0, 0, 0]],
            ...     [[1, 2, 3], [0, 0, 0], [0, 0, 0]],
            ... ]))
            >>> bernstein_coeffs = polynomial.to_bernstein_basis()
            >>> print(bernstein_coeffs)
        """
        return monomial_to_bezier(self.coeffs)

    @classmethod
    def from_bernstein_basis(cls,coeffs):
        return cls(bezier_to_monomial(coeffs))

    def to_bezier(self)->NURBSSurface:
        m,n=self.coeffs.shape[:2]
        m-=1
        n-=1
        return NURBSSurface(self.coeffs, (m,n))
    def evaluate_v2(self, u,v):
        return evaluate_monomial2d(self.coeffs, u, v)
    @classmethod
    def from_bezier(cls, surf:NURBSSurface):
        return cls(np.array(surf.control_points).copy())

    @classmethod
    def from_nurbs(cls, surf: NURBSSurface):
        m,n=surf.degree
        size_u,size_v,_=surf.control_points.shape

        if m==(size_u-1) and n==(size_v-1):
            return cls.from_bezier(surf)
        else:
            raise NotImplementedError("The transformation for complex NURBS surfaces has not been implemented at this time. Decompose the patch into bezier patches and transform each of them.")

    def to_nurbs(self)->NURBSSurface:
        return self.to_bezier()

    def cross(self, other:Monomial2D)->Monomial2D:
        return Monomial2D(cross_product_monomial(self.coeffs,other.coeffs))

    def monomial_derivatives(self)->tuple[Monomial2D,Monomial2D]:
        du,dv=monomial_partial_derivatives(self.coeffs)
        return Monomial2D(du),Monomial2D(dv)

    def monomial_normal(self)->Monomial2D:
        """
        >>> m = Monomial2D(np.array([
        ...     [[0, 0, 0], [0, 0, 0], [1, 3, 1]],
        ...     [[0, -1, -2], [2, 0, 0], [0, 0, 0]],
        ...     [[1, 2, 3], [0, 0, 0], [0, 0, 0]],
        ... ]))
        >>> normal_coeffs=normal_vector_monomial(m.coeffs)
        >>> normal_vector=evaluate_monomial2d(normal_coeffs,0.5,0.5)
        >>> unit_normal_vector=normal_vector/np.linalg.norm(normal_vector)
        >>> numeric_normal_vector=m.normal(np.array((0.5,0.5)))
        >>> np.allclose(unit_normal_vector,numeric_normal_vector)
        True

        :return:
        """
        n_coeffs = normal_vector_monomial(self.coeffs)
        return Monomial2D(n_coeffs)
