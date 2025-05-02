# distutils: language = c++
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False

import cython
from libcpp.vector cimport vector as cvector

ctypedef double D

cdef inline int find_span(int p, D* U, int n, D u) nogil:
    cdef int low = p, high = n, mid
    if u >= U[n]:
        return n - 1
    while low <= high:
        mid = (low + high) // 2
        if u < U[mid]:
            high = mid - 1
        elif u >= U[mid + 1]:
            low = mid + 1
        else:
            return mid
    return p

cdef inline void compute_basis_ders(int p, D* U, int span, D u, int d, cvector[D]& ders) nogil:
    cdef int r, j, k, idx_ndu
    cdef D saved, temp
    cdef cvector[D] left = cvector[D](p+1)
    cdef cvector[D] right = cvector[D](p+1)
    cdef cvector[D] ndu = cvector[D]((p+1)*(p+1))
    left[0] = 0.0; right[0] = 0.0; ndu[0] = 1.0
    for j in range(1, p+1):
        left[j] = u - U[span + 1 - j]
        right[j] = U[span + j] - u
        saved = 0.0
        for r in range(j):
            idx_ndu = j*(p+1) + r
            ndu[idx_ndu] = right[r+1] + left[j-r]
            temp = ndu[r*(p+1) + (j-1)] / ndu[idx_ndu]
            ndu[r*(p+1) + j] = saved + right[r+1] * temp
            saved = left[j-r] * temp
        ndu[j*(p+1) + j] = saved
    ders.resize((d+1)*(p+1))
    for r in range(p+1):
        ders[r] = ndu[r*(p+1) + p]
    cdef cvector[cvector[D]] a = cvector[cvector[D]](2)
    a[0] = cvector[D](d+1)
    a[1] = cvector[D](d+1)
    for k in range(1, d+1):
        for r in range(p+1):
            saved = 0.0
            for j in range(k):
                idx_ndu = (p - k + 1 + j)*(p+1) + r
                a[1][j] = a[0][j] / ndu[idx_ndu]
                saved += ndu[idx_ndu + k] * a[1][j]
            ders[k*(p+1) + r] = saved
        a[0], a[1] = a[1], a[0]
    cdef int scale = 1
    for k in range(1, d+1):
        scale *= (p - k + 1)
        for r in range(p+1):
            ders[k*(p+1) + r] *= scale

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void eval_nurbs_curve(  int order,
                             double[::1] U, double[:, ::1] P, double[::1] W,
                             double u, double[:, ::1] R) noexcept nogil:
    cdef size_t num_ctrlpts=P.shape[0]
    cdef size_t dim = P.shape[1]

    cdef int p = order - 1
    cdef int span = find_span(p, &U[0], num_ctrlpts, u)
    cdef int d = p < 2 and p or 2
    cdef cvector[D] ders;
    compute_basis_ders(p, &U[0], span, u, d, ders)
    cdef cvector[D] d_hom = cvector[D]((d+1)*(dim+1))
    for i in range((d+1)*(dim+1)): d_hom[i] = 0.0
    cdef int k, j, m, idxD, idxH
    cdef D w, coeff
    for k in range(d+1):
        for j in range(p+1):
            idxD = k*(p+1) + j
            w = W[span - p + j]
            coeff = ders[idxD]
            for m in range(dim):
                idxH = k*(dim+1) + m
                d_hom[idxH] += coeff * P[span - p + j, m] * w
            d_hom[k*(dim+1) + dim] += coeff * w
    cdef D inv_w0 = 1.0 / d_hom[dim]
    for m in range(dim): R[0, m] = d_hom[m] * inv_w0
    if d >= 1:
        for m in range(dim):
            R[1, m] = (d_hom[(dim+1) + m] - d_hom[(dim+1) + dim] * R[0, m]) * inv_w0
    else:
        for m in range(dim): R[1, m] = 0.0
    if d >= 2:
        cdef D w1 = d_hom[(dim+1) + dim]
        for m in range(dim):
            R[2, m] = ((d_hom[2*(dim+1) + m] - d_hom[2*(dim+1) + dim] * R[0, m]) * inv_w0
                        - 2 * (w1 * inv_w0) * R[1, m])
    else:
        for m in range(dim): R[2, m] = 0.0

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void eval_nurbs_surface( int order_u, int order_v,
                               double[::1] U, double[::1] V,
                               double[:, :, ::1] P, double[:, ::1] W,
                               double u, double v,
                               double[:, ::1] S) noexcept nogil:
    cdef size_t m
    cdef int num_u=P.shape[0]
    cdef int num_v=P.shape[1]
    cdef int dim=P.shape[2]
    cdef int p = order_u - 1
    cdef int q = order_v - 1
    cdef int span_u = find_span(p, &U[0], num_u, u)
    cdef int span_v = find_span(q, &V[0], num_v, v)
    cdef int du = p < 2 and p or 2
    cdef int dv = q < 2 and q or 2
    cdef cvector[D] ders_u, ders_v
    compute_basis_ders(p, &U[0], span_u, u, du, ders_u)
    compute_basis_ders(q, &V[0], span_v, v, dv, ders_v)
    cdef int size = (du+1)*(dv+1)*(dim+1)
    cdef cvector[D] d_hom = cvector[D](size)
    for i in range(size): d_hom[i] = 0.0
    cdef cvector[D] tmp = cvector[D](dim+1)
    cdef int i_u, j_v, i_d, j_d, m, idx_tmp, idx_h
    cdef D w_val, coeff_uv
    for j_v in range(q+1):
        for i_u in range(p+1):
            w_val = W[span_u - p + i_u, span_v - q + j_v]
            for m in range(dim):
                tmp[m] = P[span_u - p + i_u, span_v - q + j_v, m] * w_val
            tmp[dim] = w_val
            for i_d in range(du+1):
                for j_d in range(dv+1):
                    coeff_uv = ders_u[i_d*(p+1) + i_u] * ders_v[j_d*(q+1) + j_v]
                    for m in range(dim+1):
                        idx_h = ((i_d*(dv+1) + j_d)*(dim+1) + m)
                        d_hom[idx_h] += coeff_uv * tmp[m]
    cdef D denom = d_hom[((0*(dv+1)+0)*(dim+1) + dim)]
    cdef D inv_den = 1.0 / denom
    # fill S rows: 0:S,1:Su,2:Sv,3:Suu,4:Suv,5:Svv
    for m in range(dim):
        S[0, m] = d_hom[m] * inv_den
    # Su
    if du >= 1:
        for m in range(dim):
            cdef D w_u = d_hom[((1*(dv+1)+0)*(dim+1) + dim)]
            S[1, m] = (d_hom[((1*(dv+1)+0)*(dim+1) + m)] - w_u * S[0, m]) * inv_den
    else:
        for m in range(dim): S[1, m] = 0.0
    # Sv
    if dv >= 1:
        for m in range(dim):
            cdef D w_v = d_hom[((0*(dv+1)+1)*(dim+1) + dim)]
            S[2, m] = (d_hom[((0*(dv+1)+1)*(dim+1) + m)] - w_v * S[0, m]) * inv_den
    else:
        for m in range(dim): S[2, m] = 0.0
    # Suu
    if du >= 2:
        for m in range(dim):
            cdef D w_uu = d_hom[((2*(dv+1)+0)*(dim+1) + dim)]
            S[3, m] = ((d_hom[((2*(dv+1)+0)*(dim+1) + m)] - w_uu * S[0, m]) * inv_den
                        - 2 * (w_u * inv_den) * S[1, m])
    else:
        for m in range(dim): S[3, m] = 0.0
    # Suv
    if du >= 1 and dv >= 1:
        for m in range(dim):
            cdef D w_uv = d_hom[((1*(dv+1)+1)*(dim+1) + dim)]
            S[4, m] = ((d_hom[((1*(dv+1)+1)*(dim+1) + m)] - w_uv * S[0, m]) * inv_den
                        - (d_hom[((1*(dv+1)+0)*(dim+1) + dim)] * inv_den) * S[2, m]
                        - (d_hom[((0*(dv+1)+1)*(dim+1) + dim)] * inv_den) * S[1, m])
    else:
        for m in range(dim): S[4, m] = 0.0
    # Svv
    if dv >= 2:
        for m in range(dim):
            cdef D w_vv = d_hom[((0*(dv+1)+2)*(dim+1) + dim)]
            S[5, m] = ((d_hom[((0*(dv+1)+2)*(dim+1) + m)] - w_vv * S[0, m]) * inv_den
                        - 2 * (w_v * inv_den) * S[2, m])
    else:
        for m in range(dim): S[5, m] = 0.0

