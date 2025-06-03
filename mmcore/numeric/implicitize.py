import numpy as np

def poly2d_mul(polyA, polyB):
    """
    Multiply two 2D polynomials in (u,v).
    polyA and polyB are NumPy arrays of shape (a_u+1, a_v+1) and (b_u+1, b_v+1),
    whose entries are the coefficients for u^i v^j. Returns an array of shape
    (a_u+b_u+1, a_v+b_v+1).
    """
    a_u, a_v = polyA.shape[0] - 1, polyA.shape[1] - 1
    b_u, b_v = polyB.shape[0] - 1, polyB.shape[1] - 1
    result = np.zeros((a_u + b_u + 1, a_v + b_v + 1))
    for i in range(a_u + 1):
        for j in range(a_v + 1):
            coeffA = polyA[i, j]
            if abs(coeffA) < 1e-18:
                continue
            result[i:i + b_u + 1, j:j + b_v + 1] += coeffA * polyB
    return result


def poly2d_pow(poly, exponent):
    """
    Raise a 2D polynomial (in u,v) to a nonnegative integer exponent.
    Uses repeated squaring / repeated multiplication.
    If exponent = 0, returns the constant 1 polynomial [[1.0]].
    """
    if exponent == 0:
        return np.array([[1.0]])
    if exponent == 1:
        return poly.copy()
    half = poly2d_pow(poly, exponent // 2)
    half2 = poly2d_mul(half, half)
    if exponent % 2 == 0:
        return half2
    return poly2d_mul(half2, poly)


def _implicitize_bezier_patch(Ax, Ay, Az, Aw, D=None):
    """
    Given four 2D coefficient arrays Ax, Ay, Az, Aw of shape (m+1, n+1) each,
    representing the power-basis form of X(u,v), Y(u,v), Z(u,v), W(u,v) on a
    single Bézier patch of bidegree (m, n), compute the homogeneous implicit
    polynomial F(X,Y,Z,W) of total degree D (default D = m + n). Returns the 4D
    array C4 of shape (D+1, D+1, D+1, D+1) holding coefficients c[alpha,beta,gamma,delta]
    with alpha + beta + gamma + delta = D.
    """
    m, n = Ax.shape[0] - 1, Ax.shape[1] - 1
    if D is None:
        D = m + n

    # Precompute X^k, Y^k, Z^k, W^k for k = 0..D
    Px = [None] * (D + 1)
    Py = [None] * (D + 1)
    Pz = [None] * (D + 1)
    Pw = [None] * (D + 1)

    Px[0] = np.array([[1.0]])
    Py[0] = np.array([[1.0]])
    Pz[0] = np.array([[1.0]])
    Pw[0] = np.array([[1.0]])

    for k in range(1, D + 1):
        if k <= D:
            Px[k] = poly2d_pow(Ax, k)
            Py[k] = poly2d_pow(Ay, k)
            Pz[k] = poly2d_pow(Az, k)
            Pw[k] = poly2d_pow(Aw, k)

    # List all multi-indices (alpha,beta,gamma,delta) with alpha+beta+gamma+delta = D
    multi_indices = []
    for alpha in range(D + 1):
        for beta in range(D + 1 - alpha):
            for gamma in range(D + 1 - alpha - beta):
                delta = D - (alpha + beta + gamma)
                multi_indices.append((alpha, beta, gamma, delta))
    N_unk = len(multi_indices)

    # Compute the 2D pullback E_{alpha,beta,gamma,delta}[p,q]
    # for each multi-index by convolving Px[alpha], Py[beta], Pz[gamma], Pw[delta]
    E_dict = {}
    for idx, (alpha, beta, gamma, delta) in enumerate(multi_indices):
        poly_uv = Px[alpha]
        poly_uv = poly2d_mul(poly_uv, Py[beta])
        poly_uv = poly2d_mul(poly_uv, Pz[gamma])
        poly_uv = poly2d_mul(poly_uv, Pw[delta])
        E_dict[(alpha, beta, gamma, delta)] = poly_uv

    # Assemble matrix M of size ((D*m+1)*(D*n+1)) x N_unk
    rows = (D * m + 1) * (D * n + 1)
    cols = N_unk
    M = np.zeros((rows, cols))

    for c, (alpha, beta, gamma, delta) in enumerate(multi_indices):
        E = E_dict[(alpha, beta, gamma, delta)]
        assert E.shape == (D * m + 1, D * n + 1)
        for p in range(D * m + 1):
            for q in range(D * n + 1):
                M[p * (D * n + 1) + q, c] = E[p, q]

    # Solve for nullspace of M via SVD (take singular vector for smallest singular value)
    U, Svals, Vt = np.linalg.svd(M)
    c_vec = Vt[-1, :]

    # Reshape c_vec into a 4D array C4[alpha,beta,gamma,delta] of shape (D+1, D+1, D+1, D+1)
    C4 = np.zeros((D + 1, D + 1, D + 1, D + 1))
    for idx, (alpha, beta, gamma, delta) in enumerate(multi_indices):
        C4[alpha, beta, gamma, delta] = c_vec[idx]

    return C4
from mmcore.geom._nurbs_eval import NURBSSurfaceTuple,to_homogeneous_2d
from mmcore.numeric.monomial import bezier_to_monomial,monomial_to_bezier
def implicitize_bezier_patch(patch:NURBSSurfaceTuple, D=None):

    if len(np.unique(patch.knot_u  ))>2 or len(np.unique(patch.knot_v  ))>2:
        raise ValueError('input patch should be Bezier')

    hpoints=to_homogeneous_2d(patch.control_points,patch.weights)
    mono=bezier_to_monomial(hpoints)
    return _implicitize_bezier_patch(mono[...,0],mono[...,1],mono[...,2],mono[...,3],D=D)
def dehomogenize_and_evaluate(C4):
    """
    Given the 4D coefficient array C4 for F(X,Y,Z,W), return the 3D array C3
    for f(x,y,z) = F(x,y,z,1) of degree <= D in (x,y,z). Also returns a function
    `f(x,y,z)` for numerical evaluation.
    """
    D = C4.shape[0] - 1
    C3 = np.zeros((D + 1, D + 1, D + 1))

    for alpha in range(D + 1):
        for beta in range(D + 1 - alpha):
            for gamma in range(D + 1 - alpha - beta):
                delta = D - (alpha + beta + gamma)
                C3[alpha, beta, gamma] = C4[alpha, beta, gamma, delta]

    def f_eval(x, y, z):
        val = 0.0
        for a in range(D + 1):
            for b in range(D + 1 - a):
                for c in range(D + 1 - a - b):
                    coeff = C3[a, b, c]
                    if abs(coeff) < 1e-18:
                        continue
                    val += coeff * (x**a) * (y**b) * (z**c)
        return val

    return C3, f_eval


# Example usage:
if __name__ == "__main__":

    # Example Bézier patch of bidegree (2, 2) in power basis:
    Ax = np.array([[1.0, 0.0, 2.0],
                   [0.0, 1.0, 0.0],
                   [3.0, 0.0, 1.0]])
    Ay = np.array([[0.0, 2.0, 0.0],
                   [1.0, 0.0, 1.0],
                   [0.0, 2.0, 0.0]])
    Az = np.array([[0.0, 0.0, 1.0],
                   [0.0, 1.0, 0.0],
                   [1.0, 0.0, 0.0]])
    Aw = np.ones((3, 3))

    C4 = _implicitize_bezier_patch(Ax, Ay, Az, Aw)
    C3, f = dehomogenize_and_evaluate(C4)

    # Test f on a random point (u, v) -> (x, y, z)
    u_test, v_test = 0.3, 0.6
    # Evaluate X(u,v), Y(u,v), Z(u,v), W(u,v) directly:
    def eval_poly2d(poly, u, v):
        res = 0.0
        deg_u, deg_v = poly.shape[0] - 1, poly.shape[1] - 1
        for i in range(deg_u + 1):
            for j in range(deg_v + 1):
                res += poly[i, j] * (u**i) * (v**j)
        return res

    X_uv = eval_poly2d(Ax, u_test, v_test)
    Y_uv = eval_poly2d(Ay, u_test, v_test)
    Z_uv = eval_poly2d(Az, u_test, v_test)
    W_uv = eval_poly2d(Aw, u_test, v_test)

    x_pt = X_uv / W_uv
    y_pt = Y_uv / W_uv
    z_pt = Z_uv / W_uv

    print("F(x,y,z,1) at (x,y,z) from patch:", f(x_pt, y_pt, z_pt))  # Should be close to 0.0
    from mmcore.construction import cylinder_surface_2pt
    from mmcore.geom._nurbs_knots import decompose_surface
    from mmcore.geom._nurbs_eval import _tuple_to_nurbs,evaluate_nurbs_surface
    from mmcore.numeric.closest_point import closest_point_on_nurbs_surface
    bezier_parts=decompose_surface(cylinder_surface_2pt((0.,0.,-1.), (10.,10.,2.)))
    part = bezier_parts[0]

    pt=[2.,1.,0.]
    pt2=[2.463972, 1, 0.420559]
    pt3 = evaluate_nurbs_surface(part, 0.5, 0.5)['S']

    result=closest_point_on_nurbs_surface(_tuple_to_nurbs(part),pt, tol=1e-3)

    implicit_patch=implicitize_bezier_patch(part)
    C3, f = dehomogenize_and_evaluate(implicit_patch)
    print("F(x,y,z,1) at (x,y,z) from patch:",
          f(*pt))
    print("F(x,y,z,1) at (x,y,z) from patch:", f(*pt2))
    print("F(x,y,z,1) at (x,y,z) from patch:", f(*pt3))
    print(result)
