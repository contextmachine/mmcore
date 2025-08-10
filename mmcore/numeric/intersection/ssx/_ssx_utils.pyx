
cimport cython
from libc.math cimport fabs,fmin,fmax
cimport numpy as cnp
import numpy as np
cnp.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bint improve_uv(double[:] du, double[:] dv, double[:] xyz_old, double[:] xyz_better,double[:] result):
    cdef long double[3][2][2] matrix=[[[du[0], dv[0]], [du[1], dv[1]]],[[du[0], dv[0]], [du[2], dv[2]]],[[du[1], dv[1]], [du[2], dv[2]]]]
    cdef long double[3] delta= [ xyz_better[0] - xyz_old[0],xyz_better[1] - xyz_old[1], xyz_better[2] - xyz_old[2]]
    cdef long double[3][2] y=    [[delta[0], delta[1]], [delta[0], delta[2]],[delta[1], delta[2]]]

    cdef long double dett=0.
    cdef long double temp
    cdef int i
    cdef int j=0



    for i in range(3):
        temp=matrix[i][0][0] * matrix[i][1][1] - matrix[i][0][1] * matrix[i][1][0]
        if temp>dett:
            j=i
            dett=temp
    if dett<1e-9:
        return 1


    else:
        # matrix[1][0]matrix[0][0]lmatrix[1][0]ulmatrix[0][0]te x matrix[0][0]nd y using the dirematrix[1][0]t method
        result[0] = (y[j][0] * matrix[j][1][1] -matrix[j][0][1] * y[j][1]) / dett
        result[1] = (matrix[j][0][0] * y[j][1] - y[j][0] * matrix[j][1][0]) / dett

        return 0



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bint points_equal(tuple p, tuple q, double spt, double param_tol, double tan_tol, double s_min, double t_min, double u_min, double v_min, double s_max, double t_max, double u_max, double v_max):
        """
        p, q: each is a tuple (xyz, stuv, tan)
           xyz: (x,y,z)
           stuv: (s,t,u,v)
           tan:   (tx,ty,tz)
        tolerances: spt, param_tol, tan_tol
        
        Note
        -------
        Same readable python code:
        
def _param_dist_edge_wrap(stuv1, stuv2,
                          param_min: np.ndarray,
                          param_max: np.ndarray,
                          spt: float) -> float:
 
 
    mins  = param_min
    maxs  = param_max
    delta = np.abs(stuv1 - stuv2)

    # Is a wrap-around match?
    at_min_1 = np.abs(stuv1 - mins) < spt
    at_max_1 = np.abs(stuv1 - maxs) < spt
    at_min_2 = np.abs(stuv2 - mins) < spt
    at_max_2 = np.abs(stuv2 - maxs) < spt

    wrap_match = (at_min_1 & at_max_2) | (at_max_1 & at_min_2)

    # If direct close, keep delta; else if wrap_match then zero; else keep delta
    # So we can just zero out the ones that wrap.
    effective = delta * (~wrap_match)
    

    return float(np.max(effective))


def points_equal(p, q,
                 spt:  float,
                 param_tol: float,
                 tan_tol:   float,
                 param_min: np.ndarray,
                 param_max: np.ndarray) -> bool:

    xyz1, stuv1, tan1 = p
    xyz2, stuv2, tan2 = q

    # 1) Cartesian
    cart_d = np.linalg.norm(xyz1 - xyz2)

    # 2) Parametric w/ edge-wrap
    param_d = _param_dist_edge_wrap(stuv1, stuv2,
                                    param_min, param_max,
                                    spt=param_tol)

    # 3) Tangent misalignment
    dot   = float(np.dot(tan1, tan2))
    tan_d = 1.0 - abs(dot)

    #_logger.debug(f"cart_d={cart_d}, param_d={param_d}, tan_d={tan_d} (dot={dot})")

    return (cart_d < spt and
            param_d < param_tol and
            tan_d < tan_tol)
        """
        # unpack coords
        cdef double x1,y1,z1 ,x2,y2,z2,dx ,dy,dz,s1,t1,u1,v1,  s2,t2,u2,v2,ds,dt,du,dv,dot,t1x,t1y,t1z,t2x,t2y,t2z
        x1,y1,z1   = p[0]
        x2,y2,z2 = q[0]
        # 1) Cartesian: squared distance < spt^2?
        dx = x1 - x2;  dy = y1 - y2;  dz = z1 - z2
        if dx*dx + dy*dy + dz*dz >= spt*spt:
            return False

        # unpack params
        s1,t1,u1,v1 = p[1]
        s2,t2,u2,v2 = q[1]

        # inline edge-wrap check for each param:
        #   if |d1-d2| < spt → OK
        #   elif one is within spt of min and the other within spt of max → OK
        #   else → FAIL

        # s
        ds = s1 - s2
        if ds < 0: ds = -ds
        if ds >= param_tol:
            if not ((fabs(s1 - s_min) < param_tol and fabs(s2 - s_max) < param_tol)
                    or (fabs(s2 - s_min) < param_tol and fabs(s1 - s_max) < param_tol)):
                return False

        # t
        dt = t1 - t2
        if dt < 0: dt = -dt
        if dt >= param_tol:
            if not ((fabs(t1 - t_min) < param_tol and fabs(t2 - t_max) < param_tol)
                    or (fabs(t2 - t_min) < param_tol and fabs(t1 - t_max) < param_tol)):
                return False

        # u
        du = u1 - u2
        if du < 0: du = -du
        if du >= param_tol:
            if not ((fabs(u1 - u_min) < param_tol and fabs(u2 - u_max) < param_tol)
                    or (fabs(u2 - u_min) < param_tol and fabs(u1 - u_max) < param_tol)):
                return False

        # v
        dv = v1 - v2
        if dv < 0: dv = -dv
        if dv >= param_tol:
            if not ((fabs(v1 - v_min) < param_tol and fabs(v2 - v_max) < param_tol)
                    or (fabs(v2 - v_min) < param_tol and fabs(v1 - v_max) < param_tol)):
                return False

        # 3) Tangent: 1 - |dot| < tan_tol?
        t1x,t1y,t1z = p[2]
        t2x,t2y,t2z = q[2]
        dot = t1x*t2x + t1y*t2y + t1z*t2z
        if 1.0 - fabs(dot) >= tan_tol:
            return False

        # all tests passed
        return True

