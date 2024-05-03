"""Utility functions for curves module

    Naming conventions:
    ------------------
    :param p, q: degree
    :param u, v: parameter at curve
    :param U, V: knots (knot vector)
    :param span: span (result of find_span)
"""
import numpy as np


def find_span(p, u, U):
    """
    :param p: degree
    :param u: parameter at curve
    :param U: knots (knot vector)
    :return:
    """

    n = len(U) - p - 1
    if u >= U[n]:
        return n - 1
    elif u <= U[p]:
        return p
    else:
        low = p
        high = n
        mid = (low + high) // 2
        while u < U[mid] or u >= U[mid + 1]:
            if u < U[mid]:
                high = mid
            else:
                low = mid
            mid = (low + high) // 2
        return mid


def calc_basis_functions(span, u, p, U):
    N = np.zeros(p + 1)
    left = np.zeros(p + 1)
    right = np.zeros(p + 1)
    N[0] = 1
    for j in range(1, p + 1):
        left[j] = u - U[span + 1 - j]
        right[j] = U[span + j] - u
        saved = 0
        for r in range(j):
            rv = right[r + 1]
            lv = left[p - r]
            temp = N[r] / (rv + lv)
            N[r] = saved + rv * temp
            saved = lv * temp
        N[j] = saved
    return N


def calc_b_spline_point(p, U, P, u):
    """

    :param p: degree
    :param U: knots (knot vector)
    :param P: control points 4d
    :param u: parameter at curve
    :return:
    """
    span = find_span(p, u, U)
    N = calc_basis_functions(span, u, p, U)
    C = [0., 0., 0., 0.]
    for j in range(p + 1):
        point = P[span - p + j]
        Nj = N[j]

        wNj = point[3] * Nj
        C[0] += point[0] * wNj
        C[1] += point[1] * wNj
        C[2] += point[2] * wNj
        C[3] += point[3] * Nj
    return C


def calcKoverI(k, i):
    nom = np.prod(range(2, k + 1))
    denom = np.prod(range(2, i + 1)) * np.prod(range(2, k - i + 1))
    return nom / denom


def calcNURBSDerivatives(p, U, P, u, nd):
    Pders = calc_bspline_derivatives(p, U, P, u, nd)
    return calc_rational_curve_derivatives(Pders)


def calcSurfacePoint(p, q, U, V, P, u, v):
    uspan = find_span(p, u, U)
    vspan = find_span(q, v, V)
    Nu = calc_basis_functions(uspan, u, p, U)
    Nv = calc_basis_functions(vspan, v, q, V)
    temp = [np.array([0, 0, 0, 0]) for _ in range(q + 1)]
    for l in range(q + 1):
        for k in range(p + 1):
            point = P[uspan - p + k][vspan - q + l].copy()
            w = point[3]
            point[:3] *= w
            temp[l] += point * Nu[k]
    Sw = np.array([0, 0, 0, 0])
    for l in range(q + 1):
        Sw += temp[l] * Nv[l]
    Sw /= Sw[3]
    return Sw[:3]


def calc_basis_function_derivatives(span, u, p, n, U):
    zero_arr = [0 for _ in range(p + 1)]
    ders = [list(zero_arr) for _ in range(n + 1)]
    ndu = [list(zero_arr) for _ in range(p + 1)]
    ndu[0][0] = 1
    left = list(zero_arr)
    right = list(zero_arr)
    for j in range(1, p + 1):
        left[j] = u - U[span + 1 - j]
        right[j] = U[span + j] - u
        saved = 0
        for r2 in range(j):
            rv = right[r2 + 1]
            lv = left[j - r2]
            ndu[j][r2] = rv + lv
            temp = ndu[r2][j - 1] / ndu[j][r2]
            ndu[r2][j] = saved + rv * temp
            saved = lv * temp
        ndu[j][j] = saved
    for j in range(p + 1):
        ders[0][j] = ndu[j][p]
    for r2 in range(p + 1):
        s1 = 0
        s2 = 1
        a = [list(zero_arr) for _ in range(p + 1)]
        a[0][0] = 1
        for k in range(1, n + 1):
            d = 0
            rk = r2 - k
            pk = p - k
            if r2 >= k:
                a[s2][0] = a[s1][0] / ndu[pk + 1][rk]
                d = a[s2][0] * ndu[rk][pk]
            j1 = max(1, -rk)
            j2 = min(k - 1, p - r2)
            for j3 in range(j1, j2 + 1):
                a[s2][j3] = (a[s1][j3] - a[s1][j3 - 1]) / ndu[pk + 1][rk + j3]
                d += a[s2][j3] * ndu[rk + j3][pk]
            if r2 <= pk:
                a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r2]
                d += a[s2][k] * ndu[r2][pk]
            ders[k][r2] = d
            s1, s2 = s2, s1
    r = p
    for k in range(1, n + 1):
        for j in range(p + 1):
            ders[k][j] *= r
        r *= p - k
    return ders


def calc_bspline_derivatives(p, U, P, u, nd):
    du = min(nd, p)
    CK = []
    span = find_span(p, u, U)
    nders = calc_basis_function_derivatives(span, u, p, du, U)
    Pw = []
    for i in range(len(P)):
        point = np.array(P[i])
        w = point[3]
        for j in range(3):
            point[j] *= w
        Pw.append(point)

    for k in range(du + 1):
        tmp = np.zeros((4,), dtype=float)
        for j in range(p + 1):
            tmp += Pw[int(span - int(p) + int(j))] * nders[k][j]
        CK.append(tmp)

    for k in range(du + 1, nd + 2):
        CK.append(np.array([0, 0, 0, 0]))

    return CK


## I'm assuming calc_kover_i, calc_rational_curve_derivatives, and calc_nurbs_derivatives functions
## are similar in nature to calc_bspline_derivatives but due to lack of function details I'm unable
## to convert them to Python code.

def calc_surface_point(p, q, U, V, P, u, v):
    uspan = find_span(p, u, U)
    vspan = find_span(q, v, V)
    Nu = calc_basis_functions(uspan, u, p, U)
    Nv = calc_basis_functions(vspan, v, q, V)
    temp = []
    for l in range(q + 1):
        point = [0., 0., 0., 0.]
        for k in range(p + 1):
            point_w = np.array(P[uspan - p + k][vspan - q + l])
            w = point_w[3]
            for j in range(3):
                point_w[j] *= w
            point += point_w * Nu[k]
        temp.append(point)

    Sw = np.array([0, 0, 0, 0])
    for l in range(q + 1):
        Sw += temp[l] * Nv[l]
    Sw = Sw / Sw[3]
    target = Sw[:3]
    return target


def calc_rational_curve_derivatives(Pders):
    nd = len(Pders)
    Aders = []
    wders = []
    for i in range(nd):
        point = Pders[i]
        Aders.append(np.array([point[0], point[1], point[2]]))
        wders.append(point[3])
    CK = []
    for k in range(nd):
        v = Aders[k].copy()
        for i in range(1, k + 1):
            v -= CK[k - i] * calcKoverI(k, i) * wders[i]
        CK.append(v / wders[0])
    return CK
