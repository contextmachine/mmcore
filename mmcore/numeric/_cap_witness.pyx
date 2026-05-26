# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt

cnp.import_array()



cdef inline double dot3(double ax, double ay, double az,
                        double bx, double by, double bz) noexcept nogil:
    return ax*bx + ay*by + az*bz

cdef inline void cross3(double ax, double ay, double az,
                        double bx, double by, double bz,
                        double* rx, double* ry, double* rz) noexcept nogil:
    rx[0] = ay*bz - az*by
    ry[0] = az*bx - ax*bz
    rz[0] = ax*by - ay*bx

cdef inline double norm3(double x, double y, double z) nogil:
    return sqrt(x*x + y*y + z*z)

# Fast RNG for shuffling indices
cdef unsigned int _rng_state = 2463534242

cpdef void set_rng_seed(unsigned int seed):
    global _rng_state
    if seed == 0:
        _rng_state = 2463534242
    else:
        _rng_state = seed

cdef inline unsigned int xorshift32() nogil:
    global _rng_state
    cdef unsigned int x = _rng_state
    # Unsigned math wraps naturally; keep this in C to stay nogil-safe.
    x ^= (x << 13)
    x ^= (x >> 17)
    x ^= (x << 5)
    _rng_state = x
    return x

cdef inline bint cap_contains(double cx, double cy, double cz,
                              double m,
                              double px, double py, double pz,
                              double tol) nogil:
    return dot3(cx, cy, cz, px, py, pz) >= (m - tol)

cdef inline double cap_from_2(double ax, double ay, double az,
                              double bx, double by, double bz,
                              double* cx, double* cy, double* cz,
                              double tol) noexcept nogil:
    cdef double sx = ax + bx
    cdef double sy = ay + by
    cdef double sz = az + bz
    cdef double ns = norm3(sx, sy, sz)
    if ns <= tol:
        cx[0] = 0.0
        cy[0] = 0.0
        cz[0] = 0.0
        return 0.0
    cx[0] = sx / ns
    cy[0] = sy / ns
    cz[0] = sz / ns
    return dot3(cx[0], cy[0], cz[0], ax, ay, az)

cdef inline void cap_from_3(double ax, double ay, double az,
                            double bx, double by, double bz,
                            double dx, double dy, double dz,
                            double* cx, double* cy, double* cz,
                            double* mout,
                            double tol) noexcept nogil:
    """
    Best cap supported by up to 3 points (a,b,d).
    Declarations at top for Cython compatibility.
    """
    cdef double best_m, tcx, tcy, tcz, m2
    cdef double ux, uy, uz, vx, vy, vz
    cdef double nx, ny, nz, nn

    cdef double cax, cay, caz, mab
    cdef double cadx, cady, cadz, mad
    cdef double cbdx, cbdy, cbdz, mbd

    cdef double m0, mb, md

    best_m = -1e300

    # pair (a,b) contains d?
    m2 = cap_from_2(ax, ay, az, bx, by, bz, &tcx, &tcy, &tcz, tol)
    if m2 > best_m and cap_contains(tcx, tcy, tcz, m2, dx, dy, dz, tol):
        best_m = m2
        cx[0] = tcx
        cy[0] = tcy
        cz[0] = tcz

    # pair (a,d) contains b?
    m2 = cap_from_2(ax, ay, az, dx, dy, dz, &tcx, &tcy, &tcz, tol)
    if m2 > best_m and cap_contains(tcx, tcy, tcz, m2, bx, by, bz, tol):
        best_m = m2
        cx[0] = tcx
        cy[0] = tcy
        cz[0] = tcz

    # pair (b,d) contains a?
    m2 = cap_from_2(bx, by, bz, dx, dy, dz, &tcx, &tcy, &tcz, tol)
    if m2 > best_m and cap_contains(tcx, tcy, tcz, m2, ax, ay, az, tol):
        best_m = m2
        cx[0] = tcx
        cy[0] = tcy
        cz[0] = tcz

    if best_m > -1e200:
        mout[0] = best_m
        return

    # acute case: normal to (b-a, d-a)
    ux = bx - ax
    uy = by - ay
    uz = bz - az
    vx = dx - ax
    vy = dy - ay
    vz = dz - az

    cross3(ux, uy, uz, vx, vy, vz, &nx, &ny, &nz)
    nn = norm3(nx, ny, nz)

    if nn <= tol:
        # nearly collinear: fall back to best pair cap, then tighten to contain all 3

        mab = cap_from_2(ax, ay, az, bx, by, bz, &cax, &cay, &caz, tol)
        mad = cap_from_2(ax, ay, az, dx, dy, dz, &cadx, &cady, &cadz, tol)
        mbd = cap_from_2(bx, by, bz, dx, dy, dz, &cbdx, &cbdy, &cbdz, tol)

        if mad > mab and mad >= mbd:
            cx[0] = cadx
            cy[0] = cady
            cz[0] = cadz
        elif mbd > mab and mbd >= mad:
            cx[0] = cbdx
            cy[0] = cbdy
            cz[0] = cbdz
        else:
            cx[0] = cax
            cy[0] = cay
            cz[0] = caz

        m0 = dot3(cx[0], cy[0], cz[0], ax, ay, az)
        mb = dot3(cx[0], cy[0], cz[0], bx, by, bz)
        md = dot3(cx[0], cy[0], cz[0], dx, dy, dz)
        if mb < m0: m0 = mb
        if md < m0: m0 = md
        mout[0] = m0
        return

    cx[0] = nx / nn
    cy[0] = ny / nn
    cz[0] = nz / nn
    if dot3(cx[0], cy[0], cz[0], ax, ay, az) < 0.0:
        cx[0] = -cx[0]
        cy[0] = -cy[0]
        cz[0] = -cz[0]

    m0 = dot3(cx[0], cy[0], cz[0], ax, ay, az)
    mb = dot3(cx[0], cy[0], cz[0], bx, by, bz)
    md = dot3(cx[0], cy[0], cz[0], dx, dy, dz)
    if mb < m0: m0 = mb
    if md < m0: m0 = md
    mout[0] = m0


def hemisphere_witness_incremental(normals,
                                   double eps=1e-8,
                                   double tol=1e-12,
                                   bint shuffle=True):
    """
    Fast Cython witness.
    Returns (center(3,), margin) or None.
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=2] arr = np.ascontiguousarray(normals, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("normals must be (n,3) float64 array")

    cdef Py_ssize_t n_in = arr.shape[0]
    if n_in == 0:
        return None

    cdef double[:, ::1] Nin = arr

    cdef cnp.ndarray[cnp.float64_t, ndim=2] P = np.empty((n_in, 3), dtype=np.float64)
    cdef double[:, ::1] Pv = P

    cdef Py_ssize_t i, j, t
    cdef Py_ssize_t m = 0

    cdef double x, y, z, ln
    cdef double sx, sy, sz
    cdef double cx0, cy0, cz0, md, dtmp

    for i in range(n_in):
        x = Nin[i, 0]
        y = Nin[i, 1]
        z = Nin[i, 2]
        ln = norm3(x, y, z)
        if ln > tol:
            Pv[m, 0] = x / ln
            Pv[m, 1] = y / ln
            Pv[m, 2] = z / ln
            m += 1

    if m == 0:
        return None
    if m == 1:
        return np.array([Pv[0, 0], Pv[0, 1], Pv[0, 2]], dtype=np.float64), 1.0

    # mean shortcut
    sx = 0.0; sy = 0.0; sz = 0.0
    for i in range(m):
        sx += Pv[i, 0]
        sy += Pv[i, 1]
        sz += Pv[i, 2]
    ln = norm3(sx, sy, sz)
    if ln > tol:
        cx0 = sx / ln
        cy0 = sy / ln
        cz0 = sz / ln
        md = 1e300
        for i in range(m):
            dtmp = dot3(cx0, cy0, cz0, Pv[i, 0], Pv[i, 1], Pv[i, 2])
            if dtmp < md:
                md = dtmp
        if md > eps:
            return np.array([cx0, cy0, cz0], dtype=np.float64), float(md)

    # index array
    cdef cnp.ndarray[cnp.int32_t, ndim=1] idx = np.empty((m,), dtype=np.int32)
    cdef int[:] iv = idx
    for i in range(m):
        iv[i] = <int>i

    # shuffle
    cdef int tmpi, r
    cdef Py_ssize_t k
    if shuffle:
        for k in range(m - 1, 0, -1):
            r = <int>(xorshift32() % <unsigned int>(k + 1))
            tmpi = iv[k]
            iv[k] = iv[r]
            iv[r] = tmpi

    # incremental cap
    cdef double cx, cy, cz, mcur
    cdef int pi, pj, pk
    cdef double ax, ay, az, bx, by, bz, dx, dy, dz
    cdef double m3

    pi = iv[0]
    cx = Pv[pi, 0]; cy = Pv[pi, 1]; cz = Pv[pi, 2]
    mcur = 1.0

    for i in range(1, m):
        pi = iv[i]
        ax = Pv[pi, 0]; ay = Pv[pi, 1]; az = Pv[pi, 2]
        if dot3(cx, cy, cz, ax, ay, az) >= (mcur - tol):
            continue

        cx = ax; cy = ay; cz = az
        mcur = 1.0

        for j in range(i):
            pj = iv[j]
            bx = Pv[pj, 0]; by = Pv[pj, 1]; bz = Pv[pj, 2]
            if dot3(cx, cy, cz, bx, by, bz) >= (mcur - tol):
                continue

            mcur = cap_from_2(ax, ay, az, bx, by, bz, &cx, &cy, &cz, tol)
            if mcur <= eps:
                return None

            for t in range(j):
                pk = iv[t]
                dx = Pv[pk, 0]; dy = Pv[pk, 1]; dz = Pv[pk, 2]
                if dot3(cx, cy, cz, dx, dy, dz) >= (mcur - tol):
                    continue

                cap_from_3(ax, ay, az, bx, by, bz, dx, dy, dz, &cx, &cy, &cz, &m3, tol)
                mcur = m3
                if mcur <= eps:
                    return None

    # verify (no false positives)
    cdef double mfinal = 1e300
    cdef double dd
    for i in range(m):
        dd = dot3(cx, cy, cz, Pv[i, 0], Pv[i, 1], Pv[i, 2])
        if dd < mfinal:
            mfinal = dd

    if mfinal <= eps:
        return None

    return np.array([cx, cy, cz], dtype=np.float64), float(mfinal)
