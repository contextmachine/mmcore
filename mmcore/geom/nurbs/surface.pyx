cimport cython
from libc.stdlib cimport malloc,free
cimport mmcore.geom.nurbs.surface
from mmcore.geom.nurbs.algorithms cimport surface_point
cimport numpy as cnp
import numpy as np
from mmcore.geom.nurbs.curve import sizeof
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