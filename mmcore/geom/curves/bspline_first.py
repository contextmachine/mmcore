import numpy as np
import mmcore.geom.curves._nurbs as nurbs

from mmcore.geom.curves.curve import Curve


def find_span(n, p, u, U):
    """
    Determine the knot span index.
    """
    if u == U[n + 1]:
        return n  # Special case



    low = p
    high = n + 1
    mid = (low + high) // 2
    if u > U[-1]:
        return n

    elif u < U[0]:
        return p

    while u < U[mid] or u >= U[mid + 1]:
        if u < U[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    return mid


def basis_funs(i, u, p, U):
    """
    Compute the nonvanishing basis functions.
    """
    N = [0.0] * (p + 1)
    left = [0.0] * (p + 1)
    right = [0.0] * (p + 1)
    N[0] = 1.0

    for j in range(1, p + 1):
        left[j] = u - U[i + 1 - j]
        right[j] = U[i + j] - u
        saved = 0.0
        #print('bf',i + 1 - j)

        for r in range(j):

            temp = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp

        N[j] = saved

    return N


def curve_point(n, p, U, P, u):
    """
    Compute a point on a B-spline curve.

    Parameters:
    n (int): The number of basis functions minus one.
    p (int): The degree of the basis functions.
    U (list of float): The knot vector.
    P (list of list of float): The control points of the B-spline curve.
    u (float): The parameter value.

    Returns:
    list of float: The computed curve point.
    """
    span = find_span(n, p, u, U)
    N = basis_funs(span, u, p, U)
    C = [0.0] * len(P[0])

    for i in range(p + 1):
        for j in range(len(C)):
            #print('cp', span - p + i, P)
            C[j] += N[i] * P[span - p + i][j]

    return C


def all_basis_funs(span, u, p, U):
    """
    Compute all nonzero basis functions and their derivatives up to the ith-degree basis function.

    Parameters:
    span (int): The knot span index.
    u (float): The parameter value.
    p (int): The degree of the basis functions.
    U (list of float): The knot vector.

    Returns:
    list of list of float: The basis functions.
    """
    N = [[0.0 for _ in range(p + 1)] for _ in range(p + 1)]
    left = [0.0 for _ in range(p + 1)]
    right = [0.0 for _ in range(p + 1)]
    N[0][0] = 1.0

    for j in range(1, p + 1):
        left[j] = u - U[span + 1 - j]
        right[j] = U[span + j] - u
        saved = 0.0
        for r in range(j):
            N[j][r] = right[r + 1] + left[j - r]
            temp = N[r][j - 1] / N[j][r]
            N[r][j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        N[j][j] = saved

    return N


def ders_basis_funs(i, u, p, n, U):
    """
    Compute the nonzero basis functions and their derivatives.
    """
    ders = [[0.0 for _ in range(p + 1)] for _ in range(n + 1)]
    ndu = [[0.0 for _ in range(p + 1)] for _ in range(p + 1)]
    left = [0.0 for _ in range(p + 1)]
    right = [0.0 for _ in range(p + 1)]

    ndu[0][0] = 1.0

    for j in range(1, p + 1):
        left[j] = u - U[i + 1 - j]
        right[j] = U[i + j] - u
        saved = 0.0
        for r in range(j):
            ndu[j][r] = right[r + 1] + left[j - r]
            temp = ndu[r][j - 1] / ndu[j][r]
            ndu[r][j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        ndu[j][j] = saved

    for j in range(p + 1):
        ders[0][j] = ndu[j][p]

    a = [[0.0 for _ in range(p + 1)] for _ in range(2)]
    for r in range(p + 1):
        s1 = 0
        s2 = 1
        a[0][0] = 1.0
        for k in range(1, n + 1):
            d = 0.0
            rk = r - k
            pk = p - k
            if r >= k:
                a[s2][0] = a[s1][0] / ndu[pk + 1][rk]
                d = a[s2][0] * ndu[rk][pk]
            if rk >= -1:
                j1 = 1
            else:
                j1 = -rk
            if r - 1 <= pk:
                j2 = k - 1
            else:
                j2 = p - r
            for j in range(j1, j2 + 1):
                a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk + j]
                d += a[s2][j] * ndu[rk + j][pk]
            if r <= pk:
                a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r]
                d += a[s2][k] * ndu[r][pk]
            ders[k][r] = d
            j = s1
            s1 = s2
            s2 = j

    r = p
    for k in range(1, n + 1):
        for j in range(p + 1):
            ders[k][j] *= r
        r *= (p - k)

    return ders


def curve_derivs_alg1(n, p, U, P, u, d):
    """
    Compute the derivatives of a B-spline curve.

    Parameters:
    n (int): The number of basis functions minus one.
    p (int): The degree of the basis functions.
    U (list of float): The knot vector.
    P (list of list of float): The control points of the B-spline curve.
    u (float): The parameter value.
    d (int): The number of derivatives to compute.

    Returns:
    list of list of float: The computed curve derivatives.
    """
    du = min(d, p)
    CK = [[0.0 for _ in range(len(P[0]))] for _ in range(du + 1)]

    span = find_span(n, p, u, U)
    nders = ders_basis_funs(span, u, p, du, U)

    for k in range(du + 1):
        for j in range(p + 1):
            for l in range(len(CK[0])):
                CK[k][l] += nders[k][j] * P[span - p + j][l]

    return CK

def curve_deriv_cpts( p, U, P, d, r1, r2):
    """
    Compute control points of curve derivatives.

    Parameters:

    p (int): The degree of the basis functions.
    U (list of float): The knot vector.
    P (list of list of float): The control points of the B-spline curve.
    d (int): The number of derivatives to compute.
    r1 (int): The start index for control points.
    r2 (int): The end index for control points.

    Returns:
    list of list of float: The computed control points of curve derivatives.
    """
    r = r2 - r1

    PK = np.zeros((d+1,r+1,P.shape[1]))

    for i in range(0, r + 1):
        PK[0][i][:] =  P[r1 + i]

    for k in range(1, d + 1):
        tmp = p - k + 1
        for i in range(r - k + 1):
            for j in range(len(P[0])):

                PK[k][i][j] = tmp * (PK[k - 1][i + 1][j] - PK[k - 1][i][j]) / (U[r1 + i + p + 1] - U[r1 + i + k])

    return PK
def curve_derivs_alg2(n, p, U, P, u, d):
        """

       Compute the derivatives of a B-spline curve.

    Parameters:
    n (int): The number of basis functions minus one.
    p (int): The degree of the basis functions.
    U (list of float): The knot vector.
    P (list of list of float): The control points of the B-spline curve.
    u (float): The parameter value.
    d (int): The number of derivatives to compute.

    Returns:
        array of derivatives: The computed curve derivatives
    """

        dimension = P.shape[1]
        degree=p
        knotvector=U

        # Algorithm A3.4
        du = min(degree,d)

        CK = [[0.0 for _ in range(dimension)] for _ in range(d + 1)]

        span = find_span(n, degree,  u, knotvector)
        bfuns=all_basis_funs(span,u,degree,knotvector)


        # Algorithm A3.3
        PK = curve_deriv_cpts( degree, knotvector, P, d, (span - degree), span)

        for k in range(0, du + 1):
            for j in range(0, degree - k + 1):

                CK[k][:] = [elem + (bfuns[j][degree - k] * drv_ctl_p) for elem, drv_ctl_p in
                            zip(CK[k], PK[k][j])]

        # Return the derivatives
        return CK
def projective_to_cartesian(point):
    point=point if isinstance(point,np.ndarray) else np.array(point)
    cartesian_point = point[...,:-1] / point[..., -1]
    return cartesian_point

class NURBSCurve(Curve):
    def __init__(self, control_points, degree=3, knots=None):
        super().__init__()
        self._control_points = np.ones((len(control_points), 4))
        self._control_points[:, :-1] = control_points
        self.degree = degree
        self.knots = knots if knots is not None else self.generate_knots()
        self.n = len(self._control_points) - 1
        self._interval=[0., max(self.knots)]

    def find_span(self, t):
        return find_span(self.n, self.degree,  t, self.knots)
    @property
    def interval(self):
        """
        parametric interval
        :return:
        """
        return self._interval

    @interval.setter
    def interval(self,val):
        """
        parametric interval
        """
        s,e=val
        if s<min(self.knots) or e>max(self.knots):
            raise ValueError('interval value out of range')
        self._interval[:]=s,e

    @property
    def control_points(self):
        return self._control_points[..., :-1]

    def generate_knots(self):
        """

        This function generates default knots based on the number of control points
        :return: A list of knots
        """
        n = len(self.control_points)
        knots = np.array([0] * (self.degree + 1) + list(range(1, n - self.degree)) + [n - self.degree] * (self.degree + 1),dtype=float)
        return knots

    def evaluate(self, t: float):
        """
        Compute a point on a NURBS-spline curve.

        :param t: The parameter value.
        :return: np.array with shape (3,).
        """
        return projective_to_cartesian(curve_point(self.n, self.degree, self.knots, self._control_points, t))

    def derivatives(self, t: float, d: int = 1):
        """

        :param t: The parameter value.
         :type t: float
        :param d:  The number of derivatives to compute.
        :type d: int
        :type t:
        :return: np.array with shape (d+1,M) where M is the number of vector components. That is 2 for 2d curves and 3 for spatial curves.
        Returns an array of derivatives. There will always be a point on the curve under index 0. That is, `curve.evaluate(0.5)== curve.derivatives(0.5, 1)[0]`.
Index 1 is the first derivative,
under index 2 is the second derivative, and so on.
        """
        du=min(d, self.degree)
        CK = np.zeros((du + 1, self.control_points.shape[1]), dtype=np.float64)
        curve_derivs_alg2(self.n, self.degree, self.knots, self._control_points, t, d,  CK)

        return np.array(CK)


if __name__ == '__main__':
    control_points = np.array([[20, 110, 10.], [70, 250.3, 20.], [120, 50, 80.], [170, 200, 50.], [220, 60, 20.]],
                              dtype=float)
    curve = NURBSCurve(control_points,degree=3)

    point, first_derivative, second_derivative = curve.derivatives(0.5,2)
    print(point,first_derivative,second_derivative) # [ 79.375    181.115625  34.6875  ] [ 93.75    -54.43125  58.125  ] [ -75.    -458.625   -7.5  ]
    point2 = curve.evaluate(0.5)
    print(np.allclose(point, point2)) # True

