#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
#cython: initializedcheck=False

cimport cython
from libc.stdlib cimport malloc,free,realloc
cimport mmcore.geom.nurbs.surface
from mmcore.geom.nurbs.algorithms cimport surface_point, find_multiplicity,find_span_inline,knot_insertion,knot_insertion_kv
cimport numpy as cnp
import numpy as np
from libc.string cimport memcpy
cnp.import_array()

cdef class NURBSSurface(ParametricSurface):
    def __init__(self, double[:,:,:] control_points, tuple degree, double[:] knots_u=None, double[:] knots_v=None ):
        super().__init__()
        self._interval=np.zeros((2,2))
        self._size =[control_points.shape[0],control_points.shape[1]]
        self._degree=[degree[0],degree[1]]

        cdef int cpt_count=  self._size[0]*  self._size[1]
        self._control_points_arr=<double*>malloc(self._size[0]*self._size[1]*4*sizeof(double))
        self.control_points_view=<double[:control_points.shape[0],:control_points.shape[1],:4 ]>self._control_points_arr
        self.control_points_flat_view=<double[:cpt_count,:4 ]>self._control_points_arr
        cdef int i, j, k
        if control_points.shape[2]<4:
            for i in range(cpt_count):
                self.control_points_flat_view[i][3]=1.

        for i in range(control_points.shape[0]):
            for j in range(control_points.shape[1]):
                for k in range(control_points.shape[2]):
                    self.control_points_view[i][j][k]=control_points[i][j][k]




        if knots_u is None:
            self.generate_knots_u()
        else:
            self._knots_u=knots_u
        if knots_v is None:
            self.generate_knots_v()
        else:
            self._knots_v=knots_v
        self._update_interval()
    @property
    def knots_u(self):
        return np.array(self._knots_u)
    @knots_u.setter
    def knots_u(self,val):
        self._knots_u=val
        self._update_interval()
   
    @property
    def knots_v(self):
        return np.array(self._knots_v)
    @property
    def control_points(self):
        return np.array(self.control_points_view)
   
    
    @knots_v.setter
    def knots_v(self,val):
        self._knots_v=val
        self._update_interval()
    @property
    def degree(self):
        cdef int[:] dg=self._degree
        return dg
    @degree.setter
    def degree(self, val):
        self._degree[0]=val[0]
        self._degree[1]=val[1]
        self.generate_knots_u()
        self.generate_knots_v()
        
        
    cpdef void _update_interval(self):
        self._interval[0][0] = self._knots_u[self._degree[0]]
        self._interval[0][1] = self._knots_u[self._knots_u.shape[0] - self._degree[0]-1 ]
        self._interval[1][0] = self._knots_v[self._degree[1]]
        self._interval[1][1] = self._knots_v[self._knots_v.shape[0] - self._degree[1]-1 ]

    cdef void generate_knots_u(self):
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
        cdef int nu = self._size[0]
       
        
        self._knots_u = np.concatenate((
            np.zeros(self._degree[0] + 1),
            np.arange(1, nu - self._degree[0]),
            np.full(self._degree[0] + 1, nu - self._degree[0])
        ))
   

    cdef void generate_knots_v(self):
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
        cdef int nv = self._size[1]
       
        
        self._knots_v = np.concatenate((
            np.zeros(self._degree[1]+ 1),
            np.arange(1, nv - self._degree[1]),
            np.full(self._degree[1] + 1, nv - self._degree[1])
        ))
   


 
    cdef void cevaluate(self, double u, double v,double[:] result):
        cdef double* res=<double*>malloc(4*sizeof(double))
        res[0]=0.
        res[1]=0.
        res[2]=0.
        res[3] = 0.
        surface_point(self._size[0],self._degree[0],self._knots_u,
        self._size[1],self._degree[1],self._knots_v,self.control_points_view,u,v, 0, 0,res)
        result[0]=res[0]
        result[1] = res[1]
        result[2] = res[2]
        free(res)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef NURBSSurface ccopy(self):
        cdef int n_ctrlpts_u = self._size[0]
        cdef int n_ctrlpts_v = self._size[1]
        cdef int n_knots_u = self._knots_u.shape[0]
        cdef int n_knots_v = self._knots_v.shape[0]

        # Allocate memory for new arrays
        cdef double* new_control_points_arr = <double*>malloc(n_ctrlpts_u * n_ctrlpts_v * 4 * sizeof(double))
        cdef double* new_knots_u = <double*>malloc(n_knots_u * sizeof(double))
        cdef double* new_knots_v = <double*>malloc(n_knots_v * sizeof(double))

        if new_control_points_arr == NULL or new_knots_u == NULL or new_knots_v == NULL:
            free(new_control_points_arr)
            free(new_knots_u)
            free(new_knots_v)
            raise MemoryError("Failed to allocate memory for surface copy")

        # Copy data using memcpy for efficiency
        memcpy(new_control_points_arr, self._control_points_arr, n_ctrlpts_u * n_ctrlpts_v * 4 * sizeof(double))
        memcpy(new_knots_u, &self._knots_u[0], n_knots_u * sizeof(double))
        memcpy(new_knots_v, &self._knots_v[0], n_knots_v * sizeof(double))

        # Create new NURBSSurface object
        cdef NURBSSurface new_surface = NURBSSurface.__new__(NURBSSurface)
        new_surface._control_points_arr = new_control_points_arr
        new_surface.control_points_view = <double[:n_ctrlpts_u, :n_ctrlpts_v, :4]>new_control_points_arr
        new_surface.control_points_flat_view = <double[:n_ctrlpts_u*n_ctrlpts_v, :4]>new_control_points_arr
        new_surface._knots_u = <double[:n_knots_u]>new_knots_u
        new_surface._knots_v = <double[:n_knots_v]>new_knots_v
        new_surface._degree[0] = self._degree[0]
        new_surface._degree[1] = self._degree[1]
        new_surface._size[0] = self._size[0]
        new_surface._size[1] = self._size[1]

        # Copy interval
        new_surface._interval = self._interval.copy()


        return new_surface
    cdef void realloc_control_points(self, size_t new_size_u, size_t new_size_v ) noexcept nogil:

        self._control_points_arr= <double*> realloc(self._control_points_arr, new_size_u*new_size_v*4*sizeof(double))
        with gil:
            self.control_points_view=<double[:new_size_u,:new_size_v,:4]>self._control_points_arr
            self.control_points_flat_view = <double[:(new_size_u*new_size_v), :4]>self._control_points_arr
        self._size[0]=new_size_u
        self._size[1] = new_size_v


    cdef void cnormalize_knots_u(self) noexcept nogil:
        cdef double mx=self._knots_u[self._knots_u.shape[0]-1]
        cdef double mn=self._knots_u[0]
        cdef double d=mx-mn
        cdef int i

        for i in range(self._knots_u.shape[0]):
            self._knots_u[i]=((self._knots_u[i]-mn)/d)

    cdef void cnormalize_knots_v(self) noexcept nogil:
        cdef double mx = self._knots_v[self._knots_v.shape[0] - 1]
        cdef double mn = self._knots_v[0]
        cdef double d = mx - mn
        cdef int i
        for i in range(self._knots_v.shape[0]):
            self._knots_v[i] = ((self._knots_v[i] - mn) / d)

    cdef void cnormalize_knots(self) noexcept nogil:
        self.cnormalize_knots_u()
        self.cnormalize_knots_v()


    def normalize_knots(self):
        self.cnormalize_knots()
        self._update_interval()



    def copy(self):
        cdef NURBSSurface new_surface = self.ccopy()
        # If there are any Python-level attributes that need to be copied, do it here
        # For example, if there was a cached evaluation function:
        # new_surface._evaluate_cached = functools.lru_cache(maxsize=None)(new_surface._evaluate)
        return new_surface
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef void insert_knot_u(self, double t, int count):



        cdef int new_count_u = self._size[0] + count
        cdef int new_count_v = self._size[1]
        cdef double[:,:,:] cpts=self.control_points_view.copy()
        
        cdef int span = find_span_inline(self._size[0]-1, 
            self._degree[0], t, self._knots_u, 0
        )

        # Compute new knot vector
        cdef double[:] k_v = knot_insertion_kv(self._knots_u, t, span, count)
        cdef int s_u = find_multiplicity(t, self._knots_u)

        self._control_points_arr= <double*> realloc(self._control_points_arr, new_count_u*new_count_v*4*sizeof(double))
        self.control_points_view=<double[:new_count_u,:new_count_v,:4]>self._control_points_arr
        self.control_points_flat_view = <double[:(new_count_u*new_count_v), :4]>self._control_points_arr


        for v in range(self._size[1]):
     
            knot_insertion(
                self._degree[0],
            self._knots_u,
            cpts[:,v,:],
            t,
            count,
            s_u,
            span,0,  self.control_points_view[:,v,:])




        # Update surface properties
        free(&self._knots_u[0])
        self._knots_u = k_v
        self._size[0] = new_count_u
        self._size[1] = new_count_v
        self._update_interval()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef void insert_knot_v(self, double t, int count):



        cdef int new_count_u = self._size[0] 
        cdef int new_count_v = self._size[1] + count
        cdef double[:,:,:] cpts=self.control_points_view.copy()
        
        cdef int span = find_span_inline(self._size[1]-1, 
            self._degree[1], t, self._knots_v, 0
        )

    # Compute new knot vector
        cdef double[:] k_v = knot_insertion_kv(self._knots_v, t, span, count)
        cdef int s_v = find_multiplicity(t, self._knots_v)

        self._control_points_arr= <double*> realloc(self._control_points_arr, new_count_u*new_count_v*4*sizeof(double))
        self.control_points_view=<double[:new_count_u,:new_count_v,:4]>self._control_points_arr
        self.control_points_flat_view = <double[:(new_count_u*new_count_v), :4]>self._control_points_arr


        for u in range(self._size[0]):
     
            knot_insertion(self._degree[1],
            self._knots_v,
            cpts[u,:,:],
            t,
            count,
            s_v,
            span,0,self.control_points_view[u,:,:])



        # Update surface properties



        free(&self._knots_v[0])
        self._knots_v = k_v
        self._size[0] =new_count_u
        self._size[1] =new_count_v
        self._update_interval()

   
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef tuple split_surface_u(NURBSSurface obj, double param) :
    cdef int size_u = obj.control_points_view.shape[0]
    cdef int size_v = obj.control_points_view.shape[1]
    cdef int ks, s, r, knot_span
    cdef double[:, :, :] cpts = obj.control_points_view
    cdef double[:] knots_u = obj._knots_u
    cdef double[:] knots_v = obj._knots_v
    cdef int degree_u = obj._degree[0]
    cdef int degree_v = obj._degree[1]

    ks = find_span_inline(size_u, degree_u, param, knots_u, 0) - degree_u + 1
    s = find_multiplicity(param, knots_u)
    r = degree_u - s

    # Create a copy of the original surface and insert knot
    cdef NURBSSurface temp_obj = obj.copy()
    temp_obj.insert_knot_u(param, r)

    cdef double[:, :, :] tcpts = temp_obj.control_points_view
    cdef double[:] temp_knots_u = temp_obj._knots_u

    knot_span = find_span_inline(size_u, degree_u, param, temp_knots_u, 0) + 1

    # Create knot vectors for the two new surfaces
    cdef double[:] surf1_kv = <double[:knot_span]>malloc(knot_span * sizeof(double))
    cdef double[:] surf2_kv = <double[:len(temp_knots_u) - knot_span]>malloc((len(temp_knots_u) - knot_span) * sizeof(double))

    cdef int i
    for i in range(knot_span):
        surf1_kv[i] = temp_knots_u[i]
    for i in range(len(temp_knots_u) - knot_span):
        surf2_kv[i] = temp_knots_u[i + knot_span]

    # Add param to the end of surf1_kv and beginning of surf2_kv
    cdef int j
    for j in range(degree_u):
        surf1_kv = np.append(surf1_kv, param)
        surf2_kv = np.insert(surf2_kv, 0, param)

    # Create control points for the two new surfaces
    cdef double[:, :, :] surf1_ctrlpts = tcpts[:ks + r, :, :]
    cdef double[:, :, :] surf2_ctrlpts = tcpts[ks + r - 1:, :, :]

    # Create new surfaces
    cdef NURBSSurface surf1 = NURBSSurface(np.asarray(surf1_ctrlpts), (degree_u, degree_v), np.asarray(surf1_kv), knots_v)
    cdef NURBSSurface surf2 = NURBSSurface(np.asarray(surf2_ctrlpts), (degree_u, degree_v), np.asarray(surf2_kv), knots_v)

    return surf1, surf2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef tuple split_surface_v(NURBSSurface obj, double param) :
    cdef int size_u = obj.control_points_view.shape[0]
    cdef int size_v = obj.control_points_view.shape[1]
    cdef int ks, s, r, knot_span
    cdef double[:, :, :] cpts = obj.control_points_view
    cdef double[:] knots_u = obj._knots_u
    cdef double[:] knots_v = obj._knots_v
    cdef int degree_u = obj._degree[0]
    cdef int degree_v = obj._degree[1]

    ks = find_span_inline(size_v, degree_v, param, knots_v, 0) - degree_v + 1
    s = find_multiplicity(param, knots_v)
    r = degree_v - s

    # Create a copy of the original surface and insert knot
    cdef NURBSSurface temp_obj = obj.ccopy()
    temp_obj.insert_knot_v(param, r)

    cdef double[:, :, :] tcpts = temp_obj.control_points_view
    cdef double[:] temp_knots_v = temp_obj._knots_v

    knot_span = find_span_inline(size_v, degree_v, param, temp_knots_v, 0) + 1

    # Create knot vectors for the two new surfaces
    cdef double[:] surf1_kv = <double[:knot_span]>malloc(knot_span * sizeof(double))
    cdef double[:] surf2_kv = <double[:len(temp_knots_v) - knot_span]>malloc((len(temp_knots_v) - knot_span) * sizeof(double))

    cdef int i
    for i in range(knot_span):
        surf1_kv[i] = temp_knots_v[i]
    for i in range(len(temp_knots_v) - knot_span):
        surf2_kv[i] = temp_knots_v[i + knot_span]

    # Add param to the end of surf1_kv and beginning of surf2_kv
    cdef int j
    for j in range(degree_u):
        surf1_kv = np.append(surf1_kv, param)
        surf2_kv = np.insert(surf2_kv, 0, param)

    # Create control points for the two new surfaces
    cdef double[:, :, :] surf1_ctrlpts = tcpts[:,:ks + r, :]
    cdef double[:, :, :] surf2_ctrlpts = tcpts[:,ks + r - 1:, :]

    # Create new surfaces
    cdef NURBSSurface surf1 = NURBSSurface(surf1_ctrlpts, (degree_u, degree_v), knots_u, np.asarray(surf1_kv))
    cdef NURBSSurface surf2 = NURBSSurface(surf2_ctrlpts, (degree_u, degree_v), knots_u,np.asarray(surf2_kv))

    return surf1, surf2