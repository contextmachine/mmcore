cimport cython
cimport numpy as cnp
import numpy as np

cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef uvs(Py_ssize_t u_count, Py_ssize_t v_count, tuple bounds_u=(0.,1.), tuple bounds_v=(0.,1.) ):
    cdef double bounds_u_min=bounds_u[0]
    cdef double bounds_u_max=bounds_u[1]
    cdef double bounds_v_min = bounds_v[0]
    cdef double bounds_v_max = bounds_v[1]
    cdef Py_ssize_t i,j
    cdef Py_ssize_t l = u_count*v_count
    cdef double[:] u=np.linspace(bounds_u_min,bounds_u_max,u_count)
    cdef double[:] v=np.linspace(bounds_v_min,bounds_v_max,v_count)
    cdef double[:,:] uv = np.empty((l,2))
    cdef Py_ssize_t k=0
    for i in range(u_count):
        for j in range(v_count):
            uv[k][0]=u[i]
            uv[k][1] =v[j]
            k+=1
    return np.asarray(uv)

