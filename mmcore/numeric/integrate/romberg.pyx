#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
#cython: initializedcheck=False
#cython: language_level=3

cimport cython
import numpy as np
cimport numpy as cnp
cnp.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double romberg1d(fun, double a, double b, int max_steps=16, double acc=1e-7) :
  """
  Calculates the integral of a function using Romberg integration.

  Args:
      f: The function to integrate.
      a: Lower limit of integration.
      b: Upper limit of integration.
      max_steps: Maximum number of steps.
      acc: Desired accuracy.

  Returns:
      The approximate value of the integral.
  """
  cdef double[:] Rp
  cdef double[:] Rc
  cdef double h, c,n_k;
  cdef int i,j,ep;
  cdef double[:] R1
  cdef double[:] R2

  R1= np.zeros((max_steps,))
  R2= np.zeros((max_steps,))
  # Pointers to previous and current rows

  Rp = R1
  Rc= R2

  h = b - a  # Step size
  Rp[0] = 0.5 * h * (fun(a) + fun(b))  # First trapezoidal step

  with nogil:
    for i in range(1, max_steps):
      h /= 2.
      c = 0
      ep = 1 << (i - 1)  # 2^(i-1)
      for j in range(1, ep + 1):
        with gil:
          c += fun(a + (2 * j - 1) * h)
      Rc[0] = h * c + 0.5 * Rp[0]  # R(i,0)

      for j in range(1, i + 1):
        n_k = 4**j
        Rc[j] = (n_k * Rc[j - 1] - Rp[j - 1]) / (n_k - 1)  # Compute R(i,j)

      # Print ith row of R, R[i,i] is the best estimate so far


      if i > 1 and abs(Rp[i - 1] - Rc[i]) < acc:

        return Rc[i]

      # Swap Rn and Rc for next iteration

      Rp, Rc = Rc, Rp


  return Rp[max_steps - 1]