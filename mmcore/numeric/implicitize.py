import numpy as np
from mmcore.geom._nurbs_eval import (
    NURBSSurfaceTuple,
    NURBSCurveTuple,
    from_homogeneous_2d,
    from_homogeneous_1d,
    to_homogeneous_2d,
    to_homogeneous_1d,
)
from mmcore.geom._nurbs_knots import (
    generate_knots,decompose_surface
)
from mmcore.numeric.monomial import bezier_to_monomial, monomial_to_bezier
from mmcore.numeric._implicitize_utils import poly2d_mul,poly2d_pow
'''
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
'''
def implicitize_power_basis_ruled_patch(Ax, Ay, Az, Aw, D=None):
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
        D = m+ n

    # 1) Precompute X^k, Y^k, Z^k, W^k for k = 0..D
    Px = [None] * (D + 1)
    Py = [None] * (D + 1)
    Pz = [None] * (D + 1)
    Pw = [None] * (D + 1)

    Px[0] = np.array([[1.0]])
    Py[0] = np.array([[1.0]])
    Pz[0] = np.array([[1.0]])
    Pw[0] = np.array([[1.0]])

    for k in range(1, D + 1):
        Px[k] = poly2d_pow(Ax, k)
        Py[k] = poly2d_pow(Ay, k)
        Pz[k] = poly2d_pow(Az, k)
        Pw[k] = poly2d_pow(Aw, k)

    # 2) List all multi-indices (alpha,beta,gamma,delta) with alpha+beta+gamma+delta = D
    multi_indices = []
    for alpha in range(D + 1):
        for beta in range(D + 1 - alpha):
            for gamma in range(D + 1 - alpha - beta):
                delta = D - (alpha + beta + gamma)
                multi_indices.append((alpha, beta, gamma, delta))
    N_unk = len(multi_indices)

    # 3) For each (alpha,beta,gamma,delta), form E_{alpha,beta,gamma,delta}[p,q]
    E_dict = {}
    for alpha, beta, gamma, delta in multi_indices:
        poly_uv = Px[alpha]
        poly_uv = poly2d_mul(poly_uv, Py[beta])
        poly_uv = poly2d_mul(poly_uv, Pz[gamma])
        poly_uv = poly2d_mul(poly_uv, Pw[delta])
        E_dict[(alpha, beta, gamma, delta)] = poly_uv

    # 4) Assemble matrix M of size ((D*m+1)*(D*n+1)) x N_unk
    rows = (D * m + 1) * (D * n + 1)
    cols = N_unk
    M = np.zeros((rows, cols))

    for c, (alpha, beta, gamma, delta) in enumerate(multi_indices):
        E = E_dict[(alpha, beta, gamma, delta)]
        # E.shape should be exactly (D*m+1, D*n+1)
        for p in range(D * m + 1):
            for q in range(D * n + 1):
                M[p * (D * n + 1) + q, c] = E[p, q]

    # 5) Solve for nullspace of M by SVD
    U, Svals, Vt = np.linalg.svd(M)
    c_vec = Vt[-1, :]

    # 6) Reshape c_vec into a 4D array C4[alpha,beta,gamma,delta]
    C4 = np.zeros((D + 1, D + 1, D + 1, D + 1))
    for idx, (alpha, beta, gamma, delta) in enumerate(multi_indices):
        C4[alpha, beta, gamma, delta] = c_vec[idx]

    return C4

def dehomogenize_and_get_f(C4):
    """
    Given the 4D coefficient array C4 for F(X,Y,Z,W), return the 3D array C3
    for f(x,y,z) = F(x,y,z,1) of degree <= D in (x,y,z), and a function f_eval(x,y,z).
    """
    D = C4.shape[0] - 1
    C3 = np.zeros((D + 1, D + 1, D + 1))

    # Collect terms where delta = D - (alpha+beta+gamma)
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

def poly1d_mul(polyA, polyB):
    """
    Multiply two 1D polynomials represented as coefficient arrays,
    polyA of length (p+1), polyB of length (q+1). Returns array of length (p+q+1).
    """
    p = len(polyA) - 1
    q = len(polyB) - 1
    result = np.zeros(p + q + 1)
    for i in range(p + 1):
        coeffA = polyA[i]
        if abs(coeffA) < 1e-18:
            continue
        result[i:i + q + 1] += coeffA * polyB
    return result

def poly1d_pow(poly, exponent):
    """
    Raise a 1D polynomial (represented as coefficient array) to integer exponent.
    Uses repeated squaring.
    """
    if exponent == 0:
        return np.array([1.0])
    if exponent == 1:
        return poly.copy()
    half = poly1d_pow(poly, exponent // 2)
    half2 = poly1d_mul(half, half)
    if exponent % 2 == 0:
        return half2
    return poly1d_mul(half2, poly)

def build_curve_poly(Cx, Cy, Cz, Cw, D_patch):
    """
    Given 1D coefficient arrays Cx, Cy, Cz, Cw of length (p+1) each, representing
    a rational Bézier curve in power basis (x(t) = sum Cx[i]*t^i / sum Cw[i]*t^i, etc.),
    and given patch total degree D_patch, compute all powers Cx^a, Cy^b, Cz^c, Cw^d
    needed up to a+b+c+d = D_patch.
    Returns dictionaries:
      Px_1d[a] = Cx(t)^a  as 1D coefficient array,
      Py_1d[b] = Cy(t)^b,
      Pz_1d[c] = Cz(t)^c,
      Pw_1d[d] = Cw(t)^d.
    """
    # Determine curve degree
    p = len(Cx) - 1

    Px_1d = [None] * (D_patch + 1)
    Py_1d = [None] * (D_patch + 1)
    Pz_1d = [None] * (D_patch + 1)
    Pw_1d = [None] * (D_patch + 1)

    Px_1d[0] = np.array([1.0])
    Py_1d[0] = np.array([1.0])
    Pz_1d[0] = np.array([1.0])
    Pw_1d[0] = np.array([1.0])

    for k in range(1, D_patch + 1):
        Px_1d[k] = poly1d_pow(Cx, k)
        Py_1d[k] = poly1d_pow(Cy, k)
        Pz_1d[k] = poly1d_pow(Cz, k)
        Pw_1d[k] = poly1d_pow(Cw, k)

    return Px_1d, Py_1d, Pz_1d, Pw_1d

def curve_patch_intersection(Ax, Ay, Az, Aw, Cx, Cy, Cz, Cw, tol=1e-6):
    """
    Compute intersection parameters t of the rational curve C(t) and implicit patch f(x,y,z)=0.
    Inputs:
      - Ax, Ay, Az, Aw: 2D power-basis coefficient arrays for one Bézier patch.
      - Cx, Cy, Cz, Cw: 1D power-basis coefficient arrays for one rational Bézier curve.
    Returns:
      roots_t: list of parameter values t where intersection occurs (real in [0,1]).
      points: list of 3D intersection points corresponding to those t.
    """
    # 1) Implicitize patch:
    C4 = implicitize_power_basis_ruled_patch(Ax, Ay, Az, Aw)
    D_patch = C4.shape[0] - 1

    # 2) Dehomogenize to get f(x,y,z):
    C3, f_eval = dehomogenize_and_get_f(C4)

    # 3) Build curve powers up to exponent = D_patch
    Px_1d, Py_1d, Pz_1d, Pw_1d = build_curve_poly(Cx, Cy, Cz, Cw, D_patch)

    # 4) For each monomial in f (alpha, beta, gamma):
    #    f(x,y,z) = sum_{a+b+c ≤ D_patch} C3[a,b,c] * x^a * y^b * z^c.
    #    Since originally F was homogeneous of degree D_patch, let delta = D_patch - (a+b+c).
    #    Hence each term in F(C) yields: C3[a,b,c] * Cx^a * Cy^b * Cz^c * Cw^delta (a 1D polynomial).
    #    Sum them all to get the numerator polynomial N(t).
    numerator = np.zeros((D_patch * (len(Cx)-1) + 1,))  # length enough to hold highest degree
    for a in range(D_patch + 1):
        for b in range(D_patch + 1 - a):
            for c in range(D_patch + 1 - a - b):
                coeff_abc = C3[a, b, c]
                if abs(coeff_abc) < 1e-18:
                    continue
                delta = D_patch - (a + b + c)
                term_poly = Px_1d[a]
                term_poly = poly1d_mul(term_poly, Py_1d[b])
                term_poly = poly1d_mul(term_poly, Pz_1d[c])
                term_poly = poly1d_mul(term_poly, Pw_1d[delta])
                numerator = numerator + coeff_abc * np.pad(term_poly,
                                                          (0, numerator.shape[0] - term_poly.shape[0]),
                                                          'constant', constant_values=(0,0))

    # Trim leading zeros and get final numerator coefficients
    numerator = np.trim_zeros(numerator, trim='b')
    # 5) Find roots of numerator polynomial
    roots = np.roots(numerator[::-1])  # np.roots expects highest-first; numerator[::-1] gives that
    # 6) Filter real roots in [0,1]
    real_roots = []
    for rt in roots:
        if abs(rt.imag) < tol and -tol <= rt.real <= 1+tol:
            t_val = np.clip(rt.real, 0.0, 1.0)
            real_roots.append(t_val)

    # 7) Compute corresponding 3D points
    points = []
    for t in real_roots:
        # Evaluate curve point in homogeneous form:
        #   Cx(t), Cy(t), Cz(t), Cw(t) using Horner's method
        def eval_1d(poly, t):
            res = 0.0
            for coeff in reversed(poly):
                res = res * t + coeff
            return res
    
        xh = eval_1d(Cx, t)
        yh = eval_1d(Cy, t)
        zh = eval_1d(Cz, t)
        wh = eval_1d(Cw, t)
        if abs(wh) < 1e-12:
            continue
        x = xh / wh
        y = yh / wh
        z = zh / wh
        points.append((t, (x, y, z)))

    return points


def nurbs_surf_to_mono(patch: NURBSSurfaceTuple):

    if len(np.unique(patch.knot_u)) > 2 or len(np.unique(patch.knot_v)) > 2:
        raise ValueError("input patch should be Bezier")

    hpoints = to_homogeneous_2d(patch.control_points, patch.weights)
    mono = bezier_to_monomial(hpoints)
    return mono


def nurbs_curve_to_mono(patch: NURBSCurveTuple):

    if len(np.unique(patch.knot)) > 2:
        raise ValueError("input curve should be Bezier")

    hpoints = to_homogeneous_1d(patch.control_points, patch.weights)
    mono = bezier_to_monomial(hpoints)

    return mono


def mono_to_nurbs(mono):

    bern = monomial_to_bezier(mono)
    if len(bern.shape) == 3:
        cpts, ws = from_homogeneous_2d(bern)
        bez = NURBSSurfaceTuple(
            order_u=bern.shape[0],
            order_v=bern.shape[1],
            knot_u=generate_knots(bern.shape[0], bern.shape[0] - 1),
            knot_v=generate_knots(bern.shape[1], bern.shape[1] - 1),
            control_points=cpts,
            weights=ws,
        )

    else:
        cpts, ws = from_homogeneous_1d(bern)
        bez = NURBSCurveTuple(
            order=bern.shape[0],
            knot=generate_knots(bern.shape[0], bern.shape[0] - 1),
            control_points=cpts,
            weights=ws,
        )
    return bez

def implicitize_rational_bezier_ruled_surf(patch:NURBSSurfaceTuple, D=None):

    if len(np.unique(patch.knot_u  ))>2 or len(np.unique(patch.knot_v  ))>2:
        raise ValueError('input patch should be Bezier')


    mono=nurbs_surf_to_mono(patch)
    return implicitize_power_basis_ruled_patch(mono[...,0], mono[...,1], mono[...,2], mono[...,3], D=D)


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
                    if np.abs(coeff) < 1e-18:
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
    nurbs_patch=mono_to_nurbs(np.stack([Ax, Ay, Az, Aw],axis=-1))
    C4 = implicitize_power_basis_ruled_patch(Ax, Ay, Az, Aw)
    C3, f = dehomogenize_and_evaluate(C4)
    from mmcore.geom._nurbs_eval import evaluate_nurbs_surface
    # Test f on a random point (u, v) -> (x, y, z)
    u_test, v_test = 0.3, 0.6
    p=evaluate_nurbs_surface(nurbs_patch, u_test, v_test)
    # Evaluate X(u,v), Y(u,v), Z(u,v), W(u,v) directly:
    def eval_poly2d(poly, u, v):
        res = 0.0
        deg_u, deg_v = poly.shape[0] - 1, poly.shape[1] - 1
        for i in range(deg_u + 1):
            for j in range(deg_v + 1):
                res += poly[i, j] * (u**i) * (v**j)
        return res

    x_pt ,y_pt, z_pt = p['S']

    print("F(x,y,z,1) at (x,y,z) from patch (must be small):", f(x_pt, y_pt, z_pt),f(x_pt, y_pt, z_pt)<1e-4)  # Should be close to 0.0. Don't expect very high accuracy.
    from mmcore.construction import cylinder_surface_2pt
    from mmcore.geom._nurbs_knots import decompose_surface, generate_knots
    from mmcore.geom._nurbs_eval import _tuple_to_nurbs, evaluate_nurbs_surface, NURBSSurfaceTuple

    bezier_parts=decompose_surface(cylinder_surface_2pt((0.,0.,-1.), (10.,10.,2.)))
    part = bezier_parts[0]

    pt=[2.,1.,0.]
    pt2=[2.463972, 1, 0.420559]
    pt3 = evaluate_nurbs_surface(part, 0.5, 0.5)['S']

    implicit_patch=implicitize_rational_bezier_ruled_surf(part)
    C3, f = dehomogenize_and_evaluate(implicit_patch)
    print("F(x,y,z,1) at (x,y,z) from patch (must be negative):",
          f(*pt),f(*pt)<0)
    print("F(x,y,z,1) at (x,y,z) from patch (must be positive):", f(*pt2), f(*pt2)>=0)
    print("F(x,y,z,1) at (x,y,z) from patch (must be small):", f(*pt3), np.isclose(f(*pt3),0))

if __name__ == "__main__":
    # Example Bézier patch of bidegree (2,2) in power basis:
    from mmcore.construction import cylinder_surface_2pt
    from mmcore.geom._nurbs_knots import decompose_surface
    from mmcore.geom._nurbs_eval import _tuple_to_nurbs, evaluate_nurbs_surface
  
    srf=cylinder_surface_2pt((0.0, 0.0, -1.0), (10.0, 10.0, 2.0))
    bezier_parts = decompose_surface(srf)
    part = bezier_parts[0]

    bez_patch = nurbs_surf_to_mono(part)
    from mmcore.numeric.intersection.csx import nurbs_csx

    crv = NURBSCurveTuple(
        4,
        generate_knots(4, 3),
        *from_homogeneous_1d(
            np.array(
                [
                    [1.6187222409821980, 2.0330475406524018, -1.9553544302778452, 1],
                    [-0.97940409945283058, 1.0, 0.0, 1],
                    [-0.33907941456440316, 1.0, 0.42055858109904598, 1],
                    [2.4606925573687284, -2.5611028409132039, 3.3781124057313185, 1],
                ]
            )
        ),
    )
    C = nurbs_curve_to_mono(crv)
    intersections = curve_patch_intersection(
        bez_patch[..., 0], bez_patch[..., 1], bez_patch[..., 2], bez_patch[..., 3], C[..., 0], C[..., 1], C[..., 2], C[..., 3]
    )

    # 1) Find intersection points in 3D and parameter t on curve

    print("Intersection candidates (t, (x,y,z)):")
    print(intersections)
    res_n=nurbs_csx(_tuple_to_nurbs(crv),_tuple_to_nurbs(part))
    print('nurbs_csx result:')
    print(res_n)
