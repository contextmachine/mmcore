#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
#cython: initializedcheck=False

import functools

cimport cython
import numpy as np
cimport numpy as cnp


from libc.stdlib cimport malloc,free,realloc
from mmcore.geom.parametric cimport ParametricCurve
from mmcore.numeric cimport vectors,calgorithms
from mmcore.geom.nurbs.algorithms cimport curve_point,curve_derivs_alg2, curve_derivs_alg1,curve_deriv_cpts,find_span,find_multiplicity,find_span_inline,knot_insertion_kv,knot_insertion
from libc.math cimport fabs, sqrt,fmin,fmax,pow
from libc.string cimport memcpy,memcmp
from libc.stdint cimport uint32_t, int32_t
cnp.import_array()
cdef public char* MAGIC_BYTES=b"NRBC"
cdef public uint32_t VERSION=1
cdef int MAGIC_BYTES_SIZE=4
cdef class NURBSCurve(ParametricCurve):


    def __cinit__(self):
        self._evaluate_cached = functools.lru_cache(maxsize=None)(self._evaluate)
        self._knots=np.zeros((0,))
        self._periodic=0
        self._control_points=np.zeros((0,4))
        self._interval=np.zeros((2,))
    def __reduce__(self):
        return (self.__class__, (np.asarray(self._control_points),self._degree,  np.asarray(self._knots),  self._periodic))
    def __init__(self, double[:,:] control_points, int degree=3, double[:] knots=None, bint periodic=0):
        super().__init__()
        self._degree = degree
        self._periodic = periodic
        self._control_points = np.ones((control_points.shape[0], 4))

        if control_points.shape[1]==4:

            self._control_points[:,:] = control_points
        else:
            self._control_points[:,:-1]=control_points


        if knots is None:
                self.generate_knots()
        else:
            self._knots=knots
            self.knots_update_hook()

        if  periodic:
            self.make_periodic()





    def __deepcopy__(self, memodict={}):
        obj=self.__class__(control_points=self.control_points.copy(), degree=self._degree, knots=np.asarray(self._knots).copy())

        obj.weights=self.weights


        return obj
    cpdef void set_degree(self, int val):
        self._degree=val


    cpdef int get_degree(self):
        return self._degree
    @property
    def degree(self):
        return self._degree
    @degree.setter
    def degree(self,v):
        cdef int val=int(v)
        self.set_degree(val)

    cpdef bint is_periodic(self):
        """
        Check if the NURBS curve is periodic
        """
        cdef bint res = True
        cdef int i,j
        cdef double[:,:] part1=self._control_points[:self._degree]
        cdef double[:,:] part2= self._control_points[-self._degree:]
        for j in range(part1.shape[0]):

            for i in range(4):

                res = part1[j][i] == part2[j][i]
                if not res:
                    break

        return res

    def __getstate__(self):
        state=dict()
        state['_control_points']=np.asarray(self._control_points)
        state['_knots'] = np.asarray(self._knots)
        state['_degree'] = self._degree
        state['_periodic']=self._periodic
        state['_interval']=np.asarray(self._interval)
        state['_weights'] = np.asarray(self.weights)
        return state

    def __setstate__(self,state):
        cdef int i
        self._control_points=np.ones((len(state['_control_points']),4))
        for i in range(len(state['_control_points'])):
            self._control_points[i,0]=state['_control_points'][i][  0]
            self._control_points[i, 1] = state['_control_points'][i][1]
            self._control_points[i, 2] = state['_control_points'][i][2]
            self._control_points[i, 3] = state['_weights'][i]

        self._knots = state['_knots']
        self._degree = state['_degree']
        self._periodic = state['_periodic']
        self.weights = state['_control_points'][:,3]
        self._interval = state['_interval']


    @property
    def periodic(self):
        return self._periodic



    @property
    def control_points(self):
        return np.asarray(self._control_points[:,:-1])


    @control_points.setter
    def control_points(self, control_points):
        self._control_points = np.ones((control_points.shape[0],4))
        if control_points.shape[1]==4:
            self._control_points[:]=control_points
        else:

            self._control_points[:, :control_points.shape[1]] = control_points
        self._evaluate_cached.cache_clear()
    


    @property
    def knots(self):
        return np.asarray(self._knots)
    @knots.setter
    def knots(self, double[:] v):
        self._knots=v
        self.knots_update_hook()
        self._evaluate_cached.cache_clear()

    @property
    def weights(self):
        return np.asarray(self._control_points[:, 3])
    @weights.setter
    def weights(self, double[:] v):
        self._control_points[:, 3]=v
        self._evaluate_cached.cache_clear()

    @property
    def greville_abscissae(self):
        return np.asarray(self._greville_abscissae)


    cdef void generate_knots(self):
        """
        This function generates default knots based on the number of control points
        :return: A numpy array of knots
        
        Notes
        ------
        **Difference with OpenNURBS**
        
        OpenNURBS uses a knots vector shorter by one knot on each side. 
        The original explanation can be found in `opennurbs/opennurbs_evaluate_nurbs.h`.
        [source](https://github.com/mcneel/opennurbs/blob/19df20038249fc40771dbd80201253a76100842c/opennurbs_evaluate_nurbs.h#L116-L148)
        mmcore uses the standard knotvector length according to DeBoor and The NURBS Book.

        **Difference with geomdl**
        
        Unlike geomdl, the knots vector is not automatically normalised from 0 to 1.
        However, there are no restrictions on the use of the knots normalised vector. 

        """
        cdef int n = len(self._control_points)
        self._knots = np.concatenate((
            np.zeros(self._degree + 1),
            np.arange(1, n - self._degree),
            np.full(self._degree + 1, n - self._degree)
        ))

        self.knots_update_hook()
    cpdef knots_update_hook(self):
        self._update_interval()
        self._greville_abscissae = greville_abscissae(self.knots,self.degree
                                                      )


    cdef void generate_knots_periodic(self):
        """
        This function generates knots for a periodic NURBS curve
        """
        cdef int i
        cdef int n = len(self._control_points)
        cdef int m = n + self.degree + 1
        self._knots = np.zeros(m)
        for i in range(m):
            self._knots[i] = i - self.degree
        self.knots_update_hook()

    cdef _update_interval(self):
        self._interval[0] = self._knots[self._degree]
        self._interval[1] = self._knots[self._knots.shape[0] - self._degree-1 ]
    cpdef double[:,:] generate_control_points_periodic(self, double[:,:] cpts):
        cdef int n = len(cpts)
        cdef int i
        cdef int new_n = n + self.degree
        cdef double[:,:] new_control_points = np.zeros((new_n, 4))
        new_control_points[:n, :] = cpts
        for i in range(self.degree):
            new_control_points[n + i, :] = cpts[i, :]
        return new_control_points
    cpdef void make_periodic(self):
        """
        Modify the NURBS curve to make it periodic
        """

        if self.is_periodic():
            return
        cdef int n = len(self.control_points)
        cdef int new_n = n + self.degree
        cdef double[:,:] new_control_points = np.zeros((new_n, 4))
        cdef int i
        # Copy the original control points

        new_control_points[:n, :] = self._control_points

        # Add the first degree control points to the end to make it periodic
        for i in range(self.degree):
            new_control_points[n + i, :] = self._control_points[i, :]

        self._control_points = new_control_points
        self.generate_knots_periodic()

        self._periodic=True
        self._evaluate_cached.cache_clear()





    cpdef void make_open(self):
        """
        Modify the NURBS curve to make it open
        """
        if not self.is_open():
            return
        cdef int n = len(self._control_points) - self._degree  # Calculate the original number of control points
        self._control_points = self._control_points[:n, :]  # Trim the extra control points
        self.generate_knots()  # Generate an open knot vector

        self._periodic = False
        self._evaluate_cached.cache_clear()

    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void ctangent(self, double t,double[:] result):
        cdef double[:,:] ders=np.zeros((3,3))
        self.cderivatives2(t,2, ders)
        calgorithms.evaluate_tangent(ders[1],ders[2],result)


    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void ccurvature(self, double t,double[:] result):
           cdef double[:,:] ders=np.zeros((3,3))
           cdef double nrm=0
           self.cderivatives2(t,2, ders)
           calgorithms.evaluate_curvature(ders[1],ders[2],ders[0],result)

    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void cevaluate(self, double t, double[:] result) noexcept nogil:
        """
        Compute a point on a NURBS-spline curve.

        :param t: The parameter value.
        :return: np.array with shape (3,).
        """
        cdef double w
        cdef double* _result_buffer=<double*>malloc(sizeof(double)*4)
        cdef int n= len(self._control_points)-1
        #cdef double * res = <double *> malloc(sizeof(double) * 4)

        _result_buffer[0] = 0.
        _result_buffer[1] = 0.
        _result_buffer[2] = 0.
        _result_buffer[3] = 0.

        curve_point(n, self._degree, self._knots, self._control_points, t, _result_buffer, self._periodic)

        result[0] = _result_buffer[0]
        result[1] = _result_buffer[1]
        result[2] = _result_buffer[2]

        free(_result_buffer)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef evaluate4d(self, double t) :

        cdef double * _result_buffer = <double *> malloc(sizeof(double) * 4)
        cdef int n = len(self._control_points) - 1
        #cdef double * res = <double *> malloc(sizeof(double) * 4)
        cdef double[:] result=np.zeros((4,))
        _result_buffer[0] = 0.
        _result_buffer[1] = 0.
        _result_buffer[2] = 0.
        _result_buffer[3] = 0.

        curve_point(n, self._degree, self._knots, self._control_points, t, _result_buffer, self._periodic)

        result[0] = _result_buffer[0]
        result[1] = _result_buffer[1]
        result[2] = _result_buffer[2]
        result[3] = _result_buffer[3]

        free(_result_buffer)
        return np.asarray(result)


    cpdef set(self, double[:,:] control_points, double[:] knots ):
        self._control_points=control_points
        self._knots=knots
        self._evaluate_cached.cache_clear()

    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void cevaluate_ptr(self, double t, double *result ) noexcept nogil:
        """
        Compute a point on a NURBS-spline curve.

        :param t: The parameter value.
        :return: np.array with shape (3,).
        """
        cdef double w
        cdef int n = len(self._control_points) - 1

        result[0]=0.
        result[1] = 0.
        result[2]=0.
        result[3] = 0.


        curve_point(n, self._degree, self._knots, self._control_points, t, result, self._periodic)


        result[3]=1.

    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _evaluate(self, double t):
        cdef cnp.ndarray[double,ndim=1] result =np.zeros((3,))
        self.cevaluate(t, result)

        return result

    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def evaluate(self, double t):
        return self._evaluate_cached(t)

    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def evaluate_multi(self, double[:] t):
        cdef cnp.ndarray[double, ndim=2] result=np.empty((t.shape[0],4))
        cdef int i;
        for i in range(t.shape[0]):


            self.cevaluate(t[i],result[i])
        return result[:,:3]

    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def derivative(self, t):
        cdef cnp.ndarray[double, ndim=1] result =np.zeros((3,))
        self.cderivative(t,result)
        return result


    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void cderivatives1(self, double t, int d, double[:,:] CK ) :
        """
        :param t: The parameter value.
        :type t: float
        :param d: The number of derivatives to compute.
        :type d: int
        :return: np.array with shape (d+1,M) where M is the number of vector components.
        """
        cdef int n = len(self._control_points) - 1
        cdef int i
        #cdef double[:, :]  CK = np.zeros((du + 1, 4))
        #cdef double[:, :, :] PK = np.zeros((d + 1, self._degree + 1,  self._control_points.shape[1]-1))
        curve_derivs_alg1(n, self._degree, self._knots, self._control_points[:,:-1], t, d, CK,self._periodic)

    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void cderivatives2(self, double t, int d, double[:,:] CK ) :
           """
           :param t: The parameter value.
           :type t: float
           :param d: The number of derivatives to compute.
           :type d: int
           :return: np.array with shape (d+1,M) where M is the number of vector components.
           """
           cdef int n = len(self._control_points) - 1
           cdef int i
           #cdef double[:, :]  CK = np.zeros((du + 1, 4))
           #cdef double[:, :, :] PK = np.zeros((d + 1, self._degree + 1,  self._control_points.shape[1]-1))
           cdef double[:,:,:] PK= np.zeros((d + 1, self._degree + 1, self._control_points.shape[1]-1 ))

           curve_derivs_alg2(n, self._degree, self._knots, self._control_points[:,:-1], t, d, CK, PK,self._periodic)

    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void cplane(self, double t, double[:,:] result):

        self.cderivatives2(t,2, result[:3,:])


        cdef double nrm = vectors.scalar_norm(result[1])
        result[1,0]/= nrm
        result[1,1] /= nrm
        result[1,2] /= nrm

        vectors.scalar_gram_schmidt_emplace(result[1],result[2])
        nrm=vectors.scalar_norm(result[2])

        result[2, 0] /= nrm
        result[2, 1] /= nrm
        result[2, 2] /= nrm

        result[3,0] = (result[1][1] * result[2][2]) - (result[1][2] * result[2][1])
        result[3,1] = (result[1][2] * result[2][0]) - (result[1][0] * result[2][2])
        result[3,2] = (result[1][0] * result[2][1]) - (result[1][1] * result[2][0])
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void cnormal(self, double t, double[:] result):
        cdef double[:,:] vecs=np.zeros((3,3))
        self.cplane(t,vecs)
        result[:]=vecs[2,:]

    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def derivatives2(self, double t, int d=1 ) :
        """
        :param t: The parameter value.
        :type t: float
        :param d: The number of derivatives to compute.
        :type d: int
        :return: np.array with shape (d+1,M) where M is the number of vector components.
        """

        cdef int du = min(d, self._degree)
        cdef cnp.ndarray[double, ndim=2]  CK = np.zeros((du + 1, 3))


        self.cderivatives2(t,d,CK)
        return CK
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def derivatives1(self, double t, int d=1 ) :
            """
            :param t: The parameter value.
            :type t: float
            :param d: The number of derivatives to compute.
            :type d: int
            :return: np.array with shape (d+1,M) where M is the number of vector components.
            """
            cdef int du = min(d, self._degree)
            cdef cnp.ndarray[double, ndim=2] CK = np.zeros((du + 1, 3))

            self.cderivatives1(t,d, CK)
            return CK
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def derivative_control_points(self, double t, int d=1 ) :
            """
               :param t: The parameter value.
               :type t: float
               :param d: The number of derivatives to compute.
               :type d: int
               :return: np.array with shape (d+1,M) where M is the number of vector components.
            """
            cdef int n = len(self._control_points) - 1
            cdef int span = find_span(n, self._degree, t,self._knots,self._periodic)
            cdef cnp.ndarray[double, ndim=3] PK = np.zeros((d + 1, self._degree + 1, self._control_points.shape[1]-1))
            curve_deriv_cpts(self._degree, self._knots,self._control_points, d, span - self._degree, span, PK)

            return PK
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void cderivative(self, double t, double[:] result):
        cdef double[:,:] res=np.zeros((2,3))
        self.cderivatives2(t, 1, res)
        result[0]=res[1][0]
        result[1] = res[1][1]
        result[2] = res[1][2]
    
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void csecond_derivative(self, double t, double[:] result):
        cdef double[:,:] res=np.zeros((3,3))
        self.cderivatives2(t, 2, res)
        result[0]= res[2][0]
        result[1]= res[2][1]
        result[2]= res[2][2]
    
    cpdef void insert_knot(self, double t, int count):
        """ Inserts knots n-times to a spline geometry.

    
        """

        # Start curve knot insertion
        cdef int n = self._control_points.shape[0] - 1

        # Find knot multiplicity
        cdef int s = find_multiplicity(t, self._knots)

        # Find knot span
        cdef int span = find_span_inline(n, self._degree, t, self._knots,self._periodic)

        # Compute new knot vector
        self._knots = knot_insertion_kv(self._knots, t,   span, count)

        # Compute new control points

        self._control_points = knot_insertion(self._degree, self._knots,  self._control_points, t,
                                                count, s=s, span=span,is_periodic=self._periodic)

        # Update curve


        self.knots_update_hook()
        self._evaluate_cached.cache_clear()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef NURBSCurve ccopy(self):
        cdef int n_ctrlpts = self._control_points.shape[0]
        cdef int dim = self._control_points.shape[1]
        cdef int n_knots = self._knots.shape[0]
        
        # Allocate memory for new arrays
        cdef double[:, :] new_control_points = np.empty((n_ctrlpts, dim), dtype=np.float64)
        cdef double[:] new_knots = np.empty(n_knots, dtype=np.float64)
        
        # Copy data using memcpy for efficiency
        cdef double* src_ptr
        cdef double* dst_ptr
        
        # Copy control points
        src_ptr = &self._control_points[0, 0]
        dst_ptr = &new_control_points[0, 0]
        memcpy(dst_ptr, src_ptr, n_ctrlpts * dim * sizeof(double))
        
        # Copy knots
        src_ptr = &self._knots[0]
        dst_ptr = &new_knots[0]
        memcpy(dst_ptr, src_ptr, n_knots * sizeof(double))
        
        # Create new NURBSCurve object
        cdef NURBSCurve new_curve = NURBSCurve.__new__(NURBSCurve)
        new_curve._control_points = new_control_points
        new_curve.knots = new_knots
        new_curve._degree = self._degree
        new_curve._periodic = self._periodic
        new_curve._interval[0] = self._interval[0]
        new_curve._interval[1] = self._interval[1]
        
        # Copy Greville abscissae if they exist
        if self._greville_abscissae is not None:
            new_curve._greville_abscissae = np.array(self._greville_abscissae, dtype=np.float64)
        
        return new_curve

    # Method to call ccopy from Python
    def copy(self):
        cdef NURBSCurve crv = self.ccopy()

        crv._evaluate_cached = functools.lru_cache(maxsize=None)(self._evaluate)
        return crv
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef bytes cserialize(self):
        cdef int n_ctrlpts = self._control_points.shape[0]
        cdef int dim = self._control_points.shape[1]
        cdef int n_knots = self._knots.shape[0]
        
        # Calculate total size of the byte array
        cdef size_t total_size = (
            sizeof(MAGIC_BYTES) +
            sizeof(uint32_t) + # Magic bytes and version
            sizeof(uint32_t) +  # degree
            sizeof(bint) +  # periodic flag
            sizeof(uint32_t)+  # number of control points
            sizeof(uint32_t) +  # number of knots
            n_ctrlpts * dim * sizeof(double) +  # control points
            n_knots * sizeof(double)  # knots
        )
        cdef char* buffer = <char*>malloc(total_size)
        # Allocate memory for the byte array
       
        cdef char* current = buffer
        cdef char* magic
        magic=<char*>&(MAGIC_BYTES[0])
        cdef uint32_t vers=<uint32_t> VERSION
        # Write magic bytes and version
        memcpy(buffer, magic, MAGIC_BYTES_SIZE)
        current += MAGIC_BYTES_SIZE
        (<uint32_t*>current)[0] = vers  # version
        current += sizeof(uint32_t)


        
        # Write degree and periodic flag
        (<uint32_t*>current)[0] = self._degree
        current += sizeof(uint32_t)
        (<uint32_t*>current)[0] = self._periodic
        current += sizeof(uint32_t)
        
        # Write number of control points and knots
        (<uint32_t*>current)[0] = n_ctrlpts
        current += sizeof(uint32_t)
        (<uint32_t*>current)[0] = n_knots
        current += sizeof(uint32_t)
        cdef double* src_ptr
        cdef double* dst_ptr
        
        # Copy control points
        src_ptr = &self._control_points[0, 0]
        dst_ptr = &self._knots[0]
        # Write control points
        memcpy(current,src_ptr, n_ctrlpts * dim * sizeof(double))
        current += (n_ctrlpts * dim * sizeof(double))
        
        # Write knots
        memcpy(current, dst_ptr, n_knots * sizeof(double))
        
        # Create Python bytes object and free the buffer
        cdef bytes result = buffer[:total_size]
        free(buffer)
        return result
        
    

    def serialize(self):
        
        cdef bytes res = self.cserialize()
       
   
        return res

    @staticmethod
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef NURBSCurve cdeserialize(const unsigned char[:] data):
        cdef const char* buffer = <const char*>&data[0]
        cdef const char* current = buffer
        cdef int vers
        # Check magic bytes and version
        if memcmp(current, MAGIC_BYTES, MAGIC_BYTES_SIZE) != 0:
            raise ValueError(f"Invalid magic bytes: {current}")
        current += MAGIC_BYTES_SIZE
        vers=(<uint32_t*>current)[0]
        if (<uint32_t*>current)[0] != VERSION:
            raise ValueError(f"Unsupported version: {vers}")
        current += sizeof(uint32_t)
        
        # Read degree and periodic flag
        cdef int degree = (<uint32_t*>current)[0]
        current += sizeof(uint32_t)
        cdef bint periodic = (<uint32_t*>current)[0]
        current += sizeof(uint32_t)
        
        # Read number of control points and knots
        cdef int n_ctrlpts = (<uint32_t*>current)[0]
        current += sizeof(uint32_t)
        cdef int n_knots = (<uint32_t*>current)[0]
        current += sizeof(uint32_t)
        
        # Allocate memory for control points and knots
        cdef double* control_points_data = <double*>malloc(n_ctrlpts * 4 * sizeof(double))
        cdef double* knots_data = <double*>malloc(n_knots * sizeof(double))
        
        if control_points_data == NULL or knots_data == NULL:
            free(control_points_data)
            free(knots_data)
            raise MemoryError("Failed to allocate memory for deserialization")
        
        # Copy control points data
        memcpy(control_points_data, current, n_ctrlpts * 4 * sizeof(double))
        current += n_ctrlpts * 4 * sizeof(double)
        
        # Copy knots data
        memcpy(knots_data, current, n_knots * sizeof(double))
        
        # Create memory views for the allocated arrays
        cdef double[:, :] control_points = <double[:n_ctrlpts, :4]>control_points_data
        cdef double[:] knots = <double[:n_knots]>knots_data
        
        # Create new NURBSCurve object
        cdef NURBSCurve new_curve = NURBSCurve.__new__(NURBSCurve)
        new_curve._control_points = control_points
        new_curve._knots = knots
        new_curve._degree = degree
        new_curve._periodic = periodic
        
        # Update interval and other necessary properties
        new_curve.knots_update_hook()
        
        return new_curve

    @staticmethod
    def deserialize(data):
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("Input must be bytes or bytearray")
        return NURBSCurve.cdeserialize(data)
    def __dealloc__(self):
        if self._control_points is not None:
            free(&self._control_points[0, 0])
        if self._knots is not None:
            free(&self._knots[0])
        
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:] greville_abscissae(double[:] knots, int degree):
    cdef int n = knots.shape[0] - degree - 1
    cdef double[:] greville=np.empty((n,))
    cdef double temp
    cdef int i,j,k
    for i in range(n):
        temp=0.
        for k in range(degree):
            j = i + 1 + k
            temp+=knots[j]
        temp/= degree
        greville[i]=temp
    return greville


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef tuple split_curve(NURBSCurve obj, double param):
    cdef int degree = obj._degree
    cdef double[:] knotvector = obj._knots
    cdef double[:, :] ctrlpts = obj._control_points
    cdef int n_ctrlpts = ctrlpts.shape[0]
    cdef int dim = ctrlpts.shape[1]
    cdef int ks, s, r, knot_span
    cdef int i, j
    cdef bint is_periodic = obj.is_periodic()
    
    if param <= obj._interval[0] or param >= obj._interval[1]:
        
        raise ValueError("Cannot split from the domain edge")
    
    ks = find_span_inline(n_ctrlpts - 1, degree, param, knotvector, is_periodic) - degree + 1
    s = find_multiplicity(param, knotvector)
    r = degree - s
    
    # Insert knot
    cdef NURBSCurve temp_obj = obj.ccopy()
    temp_obj.insert_knot(param, r)
    
    # Knot vectors
    knot_span = find_span_inline(temp_obj._control_points.shape[0] - 1, degree, param, temp_obj._knots, is_periodic) + 1
    cdef double[:] curve1_kv = np.empty((knot_span + 1,), dtype=np.float64)
    cdef double[:] curve2_kv = np.empty((temp_obj._knots.shape[0] - knot_span+  degree+ 1,), dtype=np.float64)
    
    for i in range(knot_span):
        curve1_kv[i] = temp_obj._knots[i]
    curve1_kv[knot_span] = param
    
    for i in range(degree + 1):
        curve2_kv[i] = param
    for i in range(temp_obj._knots.shape[0] - knot_span):
        curve2_kv[i + degree + 1] = temp_obj._knots[i + knot_span]
    
    # Control points
    cdef double[:, :] curve1_ctrlpts = np.empty((ks + r, dim), dtype=np.float64)
    cdef double[:, :] curve2_ctrlpts = np.empty((n_ctrlpts - ks - r + 1, dim), dtype=np.float64)
    
    for i in range(ks + r):
        for j in range(dim):
            curve1_ctrlpts[i, j] = temp_obj._control_points[i, j]
    
    for i in range(n_ctrlpts - ks - r + 1):
        for j in range(dim):
            curve2_ctrlpts[i, j] = temp_obj._control_points[i + ks + r - 1, j]
    
    # Create new curves
    cdef NURBSCurve curve1 = NURBSCurve(curve1_ctrlpts, degree, curve1_kv)
    cdef NURBSCurve curve2 = NURBSCurve(curve2_ctrlpts, degree, curve2_kv)
    
    return curve1, curve2


