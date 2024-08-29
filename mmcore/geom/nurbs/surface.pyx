# distutils: language = c++
from libcpp.vector cimport vector
cimport cython
from libc.stdlib cimport malloc,free,realloc
from libc.math cimport fmin,fmax
cimport mmcore.geom.nurbs.surface
from mmcore.geom.nurbs.algorithms cimport surface_point, find_multiplicity,find_span_inline,knot_insertion,knot_insertion_kv
cimport numpy as cnp
import numpy as np
from libc.string cimport memcpy
cnp.import_array()
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void aabb(double[:,:] points, double[:,:] min_max_vals) noexcept nogil:
    """
    AABB (Axis-Aligned Bounding Box) of a point collection.
    :param points: Points
    :rtype: np.ndarray[(2, K), np.dtype[float]] where:
        - N is a points count.
        - K is the number of dims. For example in 3d case (x,y,z) K=3.
    :return: AABB of a point collection.
    :rtype: np.ndarray[(2, K), np.dtype[float]] at [a1_min, a2_min, ... an_min],[a1_max, a2_max, ... an_max],
    """

    cdef int K = 3
    cdef int N = points.shape[0]
    #cdef double[:,:] min_max_vals = np.empty((2,K), dtype=np.float64)
    cdef double p
    cdef int i, j

    # Initialize min_vals and max_vals with the first point's coordinates
    for i in range(K):
        min_max_vals[0][i] = (points[0, i]/points[0, 3])
        min_max_vals[1][i] = (points[0, i]/points[0, 3])

    # Find the min and max for each dimension
    for j in range(1, N):
        for i in range(K):
            p=(points[j, i]/points[j, 3])
            if  p < min_max_vals[0][i]:
                min_max_vals[0][i] =  p
            if  p > min_max_vals[1][i]:
                min_max_vals[1][i] =  p



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
        cdef double w;
        cdef int i,j
        cdef double[:,:,:] pts=self.control_points_view[...,:3].copy()
        for i in range(self.control_points_view.shape[0]):
            for j in range(self.control_points_view.shape[1]):
                w=self.control_points_view[i,j,3]
                if w!=1.:

                    pts[i,j,0]/=w
                    pts[i,j,1]/=w
                    pts[i,j,2]/=w

        return pts


    @property
    def control_points_flat(self):
        cdef double w;
        cdef int i
        cdef double[:,:] pts=self.control_points_flat_view[...,:3].copy()
        for i in range(self.control_points_flat_view.shape[0]):
            w=self.control_points_flat_view[i,3]
            if w!=1.:
                pts[i,0]/=w
                pts[i,1]/=w
                pts[i,2]/=w
        return pts
    @property
    def control_points_flat_w(self):
        
        return self.control_points_flat_view
    @property
    def shape(self):
        return tuple(self._size)
    @property
    def control_points_w(self):
        return self.control_points_view
    
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


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _update_interval(self) noexcept nogil:
        self._interval[0][0] = self._knots_u[self._degree[0]]
        self._interval[0][1] = self._knots_u[self._knots_u.shape[0] - self._degree[0]-1 ]
        self._interval[1][0] = self._knots_v[self._degree[1]]
        self._interval[1][1] = self._knots_v[self._knots_v.shape[0] - self._degree[1]-1 ]
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
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
   
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
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
   
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void cevaluate(self, double u, double v,double[:] result) noexcept nogil:
        surface_point(self._size[0]-1,self._degree[0],self._knots_u,self._size[1]-1,self._degree[1],self._knots_v, self.control_points_view, u,v, 0, 0, result)



    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef NURBSSurface ccopy(self) :
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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void cnormalize_knots_u(self) noexcept nogil:

        cdef double mx=self._knots_u[self._knots_u.shape[0]-1]
        cdef double mn=self._knots_u[0]
        cdef double d=mx-mn
        cdef int i

        for i in range(self._knots_u.shape[0]):
            self._knots_u[i]=((self._knots_u[i]-mn)/d)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void cnormalize_knots_v(self) noexcept nogil:
        cdef double mx = self._knots_v[self._knots_v.shape[0] - 1]
        cdef double mn = self._knots_v[0]
        cdef double d = mx - mn
        cdef int i
        for i in range(self._knots_v.shape[0]):
            self._knots_v[i] = ((self._knots_v[i] - mn) / d)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void cnormalize_knots(self) noexcept nogil:
        self.cnormalize_knots_u()
        self.cnormalize_knots_v()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def normalize_knots(self) :
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
        cdef int v

        cdef int new_count_u = self._size[0] + count
        cdef int new_count_v = self._size[1]

        cdef double[:,:,:] cpts=self.control_points_view.copy()
        cdef int span = find_span_inline(self._size[0]-1, 
            self._degree[0], t, self._knots_u, 0
        )

        # Compute new knot vector
        cdef double[:] k_v = knot_insertion_kv(self._knots_u, t, span, count)
        cdef int s_u = find_multiplicity(t, self._knots_u,1e-8)
      
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
    cdef void cbbox(self, double[:,:] result) noexcept nogil:
        result[:]=0.
        aabb(self.control_points_flat_view, result)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef double[:,:] bbox(self):

        cdef double[:,:] bb=np.empty((2,3))


        aabb(self.control_points_flat_view, bb)
        return bb

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef void insert_knot_v(self, double t, int count) :
        cdef int u


        cdef int new_count_u = self._size[0] 
        cdef int new_count_v = self._size[1] + count
        cdef double[:,:,:] cpts=self.control_points_view.copy()
        
        cdef int span = find_span_inline(self._size[1]-1, 
            self._degree[1], t, self._knots_v, 0
        )

        # Compute new knot vector
        cdef double[:] k_v = knot_insertion_kv(self._knots_v, t, span, count)
        cdef int s_v = find_multiplicity(t, self._knots_v,1e-8)

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
    s = find_multiplicity(param, knots_u, 1e-8)
    r = degree_u - s

    # Create a copy of the original surface and insert knot
    cdef NURBSSurface temp_obj = obj.copy()
    temp_obj.insert_knot_u(param, r)

    cdef double[:, :, :] tcpts = temp_obj.control_points_view
    cdef double[:] temp_knots_u = temp_obj._knots_u

    knot_span = find_span_inline(temp_obj._size[0], degree_u, param, temp_knots_u, 0) + 1

    # Create knot vectors for the two new surfaces
    cdef vector[double] surf1_kv = vector[double](knot_span )
    cdef vector[double] surf2_kv = vector[double](temp_knots_u.shape[0] - knot_span)

    cdef int i
    for i in range(knot_span):
        surf1_kv[i] = temp_knots_u[i]
    for i in range(temp_knots_u.shape[0] - knot_span):

        surf2_kv[i] = temp_knots_u[i + knot_span]

    # Add param to the end of surf1_kv and beginning of surf2_kv
    cdef int j
    surf1_kv.push_back(param)
    for j in range(degree_u+1):

        surf2_kv.insert(surf2_kv.begin(), param);





    # Create control points for the two new surfaces
    cdef double[:, :, :] surf1_ctrlpts = tcpts[:ks + r, :, :]
    cdef double[:, :, :] surf2_ctrlpts = tcpts[ks + r - 1:, :, :]

    cdef double[:] surf1_kvm=np.empty(surf1_kv.size())
    cdef double[:] surf2_kvm=np.empty(surf2_kv.size())
    for i in range(surf1_kv.size()):
        surf1_kvm[i]=surf1_kv[i]
    for i in range(surf2_kv.size()):
        surf2_kvm[i]=surf2_kv[i]

    # Create new surfaces
    cdef NURBSSurface surf1 = NURBSSurface(np.asarray(surf1_ctrlpts.copy()), (degree_u, degree_v),surf1_kvm,knots_v.copy() )
    cdef NURBSSurface surf2 = NURBSSurface(np.asarray(surf2_ctrlpts.copy()), (degree_u, degree_v), surf2_kvm, knots_v.copy())


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

    knot_span = find_span_inline(temp_obj._size[1], degree_v, param, temp_knots_v, 0) + 1

    # Create knot vectors for the two new surfaces
    cdef vector[double] surf1_kv = vector[double](knot_span )
    cdef vector[double] surf2_kv = vector[double](temp_knots_v.shape[0] - knot_span)

    cdef int i
    for i in range(knot_span):
        surf1_kv[i] = temp_knots_v[i]
    for i in range(temp_knots_v.shape[0] - knot_span):

        surf2_kv[i] = temp_knots_v[i + knot_span]

    # Add param to the end of surf1_kv and beginning of surf2_kv
    cdef int j
    surf1_kv.push_back(param)
    for j in range(degree_v+1):

        surf2_kv.insert(surf2_kv.begin(), param);





    # Create control points for the two new surfaces
    cdef double[:, :, :] surf1_ctrlpts = tcpts[:,:ks + r, :]
    cdef double[:, :, :] surf2_ctrlpts = tcpts[:,ks + r - 1:, :]

    cdef double[:] surf1_kvm=np.empty(surf1_kv.size())
    cdef double[:] surf2_kvm=np.empty(surf2_kv.size())

    for i in range(surf1_kv.size()):
        surf1_kvm[i]=surf1_kv[i]
    for i in range(surf2_kv.size()):
            surf2_kvm[i]=surf2_kv[i]

    # Create new surfaces
    cdef NURBSSurface surf1 = NURBSSurface(surf1_ctrlpts.copy(), (degree_u, degree_v), knots_u.copy(), surf1_kvm)
    cdef NURBSSurface surf2 = NURBSSurface(surf2_ctrlpts.copy(), (degree_u, degree_v), knots_u.copy(),surf2_kvm)

    return surf1, surf2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef tuple subdivide_surface(NURBSSurface surface, double u=0.5,double v=0.5,bint normalize_knots=True):
    cdef tuple surfs1 = split_surface_u(surface, u)
    cdef NURBSSurface surf1, surf2, surf11, surf12, surf21, surf22
    surf1= surfs1[0]
    surf2 = surfs1[1]
    if normalize_knots:
        surf1.normalize_knots()
        surf2.normalize_knots()


    surf11, surf12 = split_surface_v(surf1,v)
    surf21, surf22 = split_surface_v(surf2,v)
    if normalize_knots:
        surf11.normalize_knots()
        surf12.normalize_knots()
        surf21.normalize_knots()
        surf22.normalize_knots()
    return surf11,surf12,surf21,surf22
