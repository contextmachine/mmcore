#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
#cython: initializedcheck=False


cimport cython

from libc.math cimport sin,cos,fabs,sqrt
cimport numpy as cnp
import numpy as np
cnp.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _cython_callback2(double u, double v,double[:] res):
    cdef double cosu=cos(u)
    cdef double sinu=sin(u)
    cdef double cosv=cos(v)
    cdef double sinv = sin(v)

    res[0]=cosu*sinv
    res[1] = sinu * sinv
    res[2] = cosv


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:] cython_callback2(double u, double v):
      cdef double[:] res =np.empty((3,))
      _cython_callback2(u,v,res)
      return res
cpdef void callback_map(double[:] u, double[:] v, double[:,:] result):
    cdef int i
    for  i in range(result.shape[0]):
        _cython_callback2(u[i], v[i], result[i])

#def second_derivative(x, h=1e-5):
#    return (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)

cdef double DEFAULT_H=1e-3

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void c_derivative_u(callback, double u, double v, double[:] result):
     cdef double[:] a
     cdef double[:] b
     cdef double h2= 2 *DEFAULT_H
     cdef double u1,u2
     u1=u-DEFAULT_H
     u2=u+DEFAULT_H

     #result[0] =callback(u, v)
     a=callback(u1,v)
     b=callback(u2,v)
     result[0] = (b[0] - a[0]) / h2
     result[1] = (b[1] - a[1]) / h2
     result[ 2] = (b[2] - a[2]) / h2

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void c_derivative_v(callback, double u, double v, double[:] result):
     cdef double[:] a
     cdef double[:] b
     cdef double h2= 2 *DEFAULT_H
     cdef double u1,u2
     v1=v-DEFAULT_H
     v2=v+DEFAULT_H

     #result[0] =callback(u, v)
     a=callback(u,v1)
     b=callback(u,v2)
     result[ 0] = (b[0] - a[0]) / h2
     result[ 1] = (b[1] - a[1]) / h2
     result[ 2] = (b[2] - a[2]) / h2

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void c_derivative_vv(callback, double u, double v, double[:] result):


     cdef double[:] pt=callback(u,v)
     cdef double[:] dv0,dv1
     cdef double h2= 2 *DEFAULT_H
     cdef double h_sq= DEFAULT_H**2

     cdef double v1,v2


     v1=v-DEFAULT_H
     v2=v+DEFAULT_H
     #result[0] =callback(u, v)


     dv0 = callback(u, v1)
     dv1 = callback(u, v2)

     result[ 0] = (dv1[0] - 2 * pt[0] + dv0[0]) / h_sq
     result[ 1] = (dv1[1] - 2 * pt[1] + dv0[1]) / h_sq
     result[ 2] = (dv1[2] - 2 * pt[2] + dv0[2]) / h_sq

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void c_derivative_uu(callback, double u, double v, double[:] result):


     cdef double[:] pt=callback(u,v)
     cdef double[:] du0,du1
     cdef double h2= 2 *DEFAULT_H
     cdef double h_sq= DEFAULT_H**2
     cdef double u1,u2


     u1=u-DEFAULT_H
     u2=u+DEFAULT_H

     #result[0] =callback(u, v)
     du0= callback(u1,v)
     du1= callback(u2,v)

     result[ 0] = (du1[0] - 2 * pt[0] + du0[0]) / h_sq
     result[ 1] = (du1[1] - 2 * pt[1] + du0[1]) / h_sq
     result[ 2] = (du1[2] - 2 * pt[2] + du0[2]) / h_sq


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void c_derivative_uv(callback, double u, double v, double[:] result):



     cdef double[:] du0,du1,dv0,dv1
     cdef double h2= 2 *DEFAULT_H
     cdef double h_sq= DEFAULT_H**2
     cdef double h2_sq= 4*h_sq
     cdef double u1,v1,u2,v2


     u1=u-DEFAULT_H
     u2=u+DEFAULT_H
     v1=v-DEFAULT_H
     v2=v+DEFAULT_H





     du1v1= callback(u2, v2)
     du1v0=  callback(u2, v1)
     du0v1 = callback(u1, v2)
     du0v0 = callback(u1, v1)

     result[ 0] = (du1v1[0] -  du1v0[0] -  du0v1[0] +  du0v0[0]) / h2_sq
     result[ 1] = (du1v1[1] -  du1v0[1] -  du0v1[1] +  du0v0[1]) /  h2_sq
     result[ 2] = (du1v1[2] -  du1v0[2] -  du0v1[2] +  du0v0[2]) /  h2_sq

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void c_derivatives(callback, double u, double v, double[:,:] result):
    """
    Function that accepts a callback and calls it with a fixed argument.
    """

    cdef double[:] a
    cdef double[:] b
    cdef double h2= 2 *DEFAULT_H
    cdef double u1,v1,u2,v2
    u1=u-DEFAULT_H
    u2=u+DEFAULT_H
    v1=v-DEFAULT_H
    v2=v+DEFAULT_H
    #result[0] =callback(u, v)
    a=callback(u1,v)
    b=callback(u2,v)


    result[0, 0]= (b[0] - a[0]) /h2
    result[0, 1]= (b[1] - a[1]) / h2
    result[0, 2] = (b[2] - a[2]) / h2
    a = callback(u, v1)
    b = callback(u, v2)

    result[1, 0] = (b[0] - a[0]) / h2
    result[1, 1] = (b[1] - a[1]) / h2
    result[1, 2] = (b[2] - a[2]) /h2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void c_normal( double[:,:] result):
    cdef double n

    result[2, 0] = (result[0, 1] *result[1, 2]) - (result[0, 2] * result[1,1])
    result[2, 1] = (result[0, 2] *result[1, 0]) - (result[0, 0] * result[1,2])
    result[2, 2] = (result[0, 0] *result[1, 1]) - (result[0, 1] * result[1,0])

    n=sqrt(result[ 2,  0] ** 2 + result[ 2,1] ** 2 + result[ 2,2] ** 2)
    result[ 2, 0] /= n
    result[2, 1] /= n
    result[ 2, 2] /= n
    

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void c_second_derivatives(callback, double u, double v, double[:,:] result):
    """
    Function that accepts a callback and calls it with a fixed argument.
    """
    cdef double[:] pt=callback(u,v)
    cdef double[:] du0,du1,dv0,dv1
    cdef double h2= 2 *DEFAULT_H
    cdef double h_sq= DEFAULT_H**2
    cdef double h2_sq= 4*h_sq
    cdef double u1,v1,u2,v2


    u1=u-DEFAULT_H
    u2=u+DEFAULT_H
    v1=v-DEFAULT_H
    v2=v+DEFAULT_H
    #result[0] =callback(u, v)
    du0= callback(u1,v)
    du1= callback(u2,v)
    dv0 = callback(u, v1)
    dv1 = callback(u, v2)
    du1v1= callback(u2, v2)
    du1v0=  callback(u2, v1)
    du0v1 = callback(u1, v2)
    du0v0 = callback(u1, v1)
    result[0, 0] = pt[0]
    result[0, 1] = pt[1]
    result[0, 2] = pt[2]
    
    
    
    result[1, 0] = (du1[0] - du0[0]) / h2
    result[1, 1] = (du1[1] - du0[1]) / h2
    result[1, 2] = (du1[2] - du0[2]) / h2

    result[2, 0] = (dv1[0] - dv0[0]) / h2
    result[2, 1] = (dv1[1] - dv0[1]) / h2
    result[2, 2] = (dv1[2] - dv0[2]) / h2



    result[3, 0]=  (du1[0] - 2 * pt[0] + du0[0]) /h_sq
    result[3, 1]=  (du1[1] - 2 * pt[1] + du0[1]) / h_sq
    result[3, 2] =  (du1[2] - 2 * pt[2] + du0[2]) / h_sq

    
    result[4, 0] = (dv1[0] - 2 * pt[0] + dv0[0]) / h_sq
    result[4, 1] = (dv1[1] - 2 * pt[1] + dv0[1]) / h_sq
    result[4, 2] = (dv1[2] - 2 * pt[2] + dv0[2]) / h_sq
  

    result[5, 0] = (du1v1[0] -  du1v0[0] -  du0v1[0] +  du0v0[0]) / h2_sq
    result[5, 1] = (du1v1[1] -  du1v0[1] -  du0v1[1] +  du0v0[1]) /  h2_sq
    result[5, 2] = (du1v1[2] -  du1v0[2] -  du0v1[2] +  du0v0[2]) /  h2_sq


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void c_origin_derivatives_normal(callback, double u, double v, double[:,:] result):


    cdef double[:] a
    cdef double[:] b

    cdef double[:] pt=callback(u,v)
    cdef double n
    cdef double h2= 2 *DEFAULT_H
    cdef double u1,v1,u2,v2
    u1=u-DEFAULT_H
    u2=u+DEFAULT_H
    v1=v-DEFAULT_H
    v2=v+DEFAULT_H
    #result[0] =callback(u, v)
    a=callback(u1,v)
    b=callback(u2,v)
    result[0, 0] = pt[0]
    result[0, 1] = pt[1]

    result[0, 2] = pt[2]

    result[1, 0]= (b[0] - a[0]) /h2
    result[1, 1]= (b[1] - a[1]) / h2
    result[1, 2] = (b[2] - a[2]) / h2
    a = callback(u, v1)
    b = callback(u, v2)

    result[2, 0] = (b[0] - a[0]) / h2
    result[2, 1] = (b[1] - a[1]) / h2
    result[2, 2] = (b[2] - a[2]) /h2


    result[3, 0] = (result[1, 1] * result[2, 2]) - (result[1, 2] * result[2, 1])
    result[3, 1] = (result[1, 2] * result[2, 0]) - (result[1, 0] * result[2, 2])
    result[3, 2] = (result[1, 0] * result[2, 1]) - (result[1, 1] * result[2, 0])

    n = sqrt(result[3, 0] ** 2 + result[3, 1] ** 2 + result[3, 2] ** 2)
    result[3, 0] /= n
    result[3, 1] /= n
    result[3, 2] /= n

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef cnp.ndarray second_derivatives(callback, double u, double v):
    """
    Function that accepts a callback and calls it with a fixed argument.
    """
    cdef double[:,:] result=np.empty((6, 3))

    c_second_derivatives(callback,u,v, result)
  

    return np.array(result)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef cnp.ndarray derivatives_normal(callback, double u, double v):
    """
    Function that accepts a callback and calls it with a fixed argument.
    """

    cdef double[:,:] result=np.empty((3, 3))

    c_derivatives(callback,u,v, result)
    
    c_normal(result)

    return np.array(result)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef cnp.ndarray derivatives_normal_array(callback, double[:] u, double[:] v):
    """
    Function that accepts a callback and calls it with a fixed argument.
    """
    cdef double[:,:,:] result=np.empty((u.shape[0],3,3))

    cdef int i
    for i in range(u.shape[0]):
        c_derivatives(callback, u[i], v[i], result[i])
        c_normal( result[i])




    return np.array(result)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef cnp.ndarray origin_derivatives_normal(callback, double u, double v):
    """
    Function that accepts a callback and calls it with a fixed argument.
    """
    
    cdef double[:,:] result=np.empty((4, 3))
    c_origin_derivatives_normal(callback, u, v, result)

    


    return np.array(result)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef cnp.ndarray origin_derivatives_normal_array(callback,  double[:] u, double[:] v):
    """
    Function that accepts a callback and calls it with a fixed argument.
    """

    cdef double[:,:,:] result=np.empty((u.shape[0],4,3))
    cdef int i
    for i in range(u.shape[0]):
        c_origin_derivatives_normal(callback, u[i], v[i], result[i])

    return np.array(result)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef cnp.ndarray second_derivatives_array(callback, double[:] u, double[:] v):
    """
    Function that accepts a callback and calls it with a fixed argument.
    """
    cdef double[:,:,:] result=np.empty((u.shape[0], 6, 3))
    cdef int i
    for i in range(u.shape[0]):
        
        c_second_derivatives(callback,u[i],v[i], result[i])


    return np.array(result)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef cnp.ndarray derivatives(callback, double u, double v):
    """
    Function that accepts a callback and calls it with a fixed argument.
    """
    cdef double[:,:] result=np.empty((2,3))

    c_derivatives(callback,u,v,result)

    return np.array(result)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef cnp.ndarray derivatives_array( callback, double[:] u, double[:] v):
    """
    Function that accepts a callback and calls it with a fixed argument.
    """
    cdef double[:,:,:] result=np.empty((u.shape[0],2,3))
    cdef int i
    for i in range(u.shape[0]):
        c_derivatives(callback,u[i],v[i],result[i])

    return np.array(result)


