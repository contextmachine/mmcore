# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: embedsignature=True
# cython: infer_types=False
# cython: initializedcheck=False
# distutils: language = c++

cimport cython

cimport mmcore.numeric.algorithms.cygjk
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(False)
def gjk(double[:,:] v1, double[:,:] v2, double tol):
    cdef vector[Vec3[double]] vf1=vector[Vec3[double]](v1.shape[0])
    cdef vector[Vec3[double]] vf2=vector[Vec3[double]](v2.shape[0])
    cdef int i;
    cdef size_t max_iter=v1.shape[0]*v2.shape[0]
    for i in range(v1.shape[0]):
        vf1[i][0]=v1[i][0]
        vf1[i][1]=v1[i][1]
        vf1[i][2]=v1[i][2]
    for i in range(v2.shape[0]):
        vf2[i][0]=v2[i][0]
        vf2[i][1]=v2[i][1]
        vf2[i][2]=v2[i][2]
    cdef bool result= gjk_collision_detection(vf1, vf2, tol, max_iter)
    return result
