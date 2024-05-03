import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport free,malloc,realloc
cdef extern from "cmmcore_array.h" nogil:
    ctypedef struct TdoubleArray:
        double *array;
        size_t used;
        size_t size;
    cdef void initTdoubleArray( TdoubleArray *a, size_t size)
    cdef void insertTdoubleArray( TdoubleArray *a, double element)
    cdef void freeTdoubleArray( TdoubleArray *a)
ctypedef np.float64_t DTYPE_t
ctypedef long INDICES_t
ctypedef struct CMesh:
    size_t size
    double *position
    double *normal
    double *color

cdef struct MeshPool
cdef  void initCMesh(CMesh *msh, size_t size):
    
    msh.position=<double*>malloc(sizeof(double)*size)
    msh.normal=<double*>malloc(sizeof(double)*size)
    msh.color=<double*>malloc(sizeof(double)*size)
    for i in range(size):
        msh.color[i][0]=0.5
        msh.color[i][1] = 0.5
        msh.color[i][2] = 0.5
cdef  void initCMesh(CMesh *msh, size_t size):
    msh.position = <double *> malloc(sizeof(double) * size)
    msh.normal = <double *> malloc(sizeof(double) * size)
    msh.color = <double *> malloc(sizeof(double) * size)
    for i in range(size):
        msh.color[i][0] = 0.5
        msh.color[i][1] = 0.5
        msh.color[i][2] = 0.5
cdef  void fuseCMesh(CMesh *msh, CMesh *other):
    cdef size_t j=0.0
    msh.size+=other.size
    
    msh.position = <double *>realloc(msh.position,sizeof(double) * msh.size)
    msh.normal = <double *> realloc(msh.position,sizeof(double) * msh.size)
    msh.color = <double *> realloc(msh.position,sizeof(double) * msh.size)
    for i in range(other.size):
        j=msh.size +i
        
        msh.color[j][0] = other.color[j][0]
        msh.color[j][1] = other.color[j][1]
        msh.color[j][2] = other.color[j][2]
        
        msh.position[j][0] = other.position[j][0]   
        msh.position[j][1] = other.position[j][1]   
        msh.position[j][2] = other.position[j][2]   
        
        msh.normal[j][0] = other.normal[j][0]    
        msh.normal[j][1] = other.normal[j][1]    
        msh.normal[j][2] = other.normal[j][2]







cdef class Mesh:
    cdef  CMesh _msh
    cdef  long[:] indices


    def __cinit__(self):
        self._msh=CMesh()

        cdef long[:] indices = np.empty(1, int)
        self.indices=indices


    cdef initSize(self,size_t size):
        initCMesh(&self._msh, size)

    def asdict(self):
        return dict(attributes=dict(position=self.position, normal=self.normal, color=self.color), indices=self.indices)
    def colorise(self, np.ndarray[double, ndim=1] color):
        self.color = np.empty((self.position.shape[0],))
        for i in range(self.position.shape[0] // 3):
            self.color[i * 3] = color[0]
            self.color[i * 3 + 1] = color[1]
            self.color[i * 3 + 2] = color[2]

    def get_position(self):
        return self.position
    def set_position(self, np.ndarray position):
        self.position = position.flatten()
    def get_normal(self):
        return self.normal
    def set_normal(self, np.ndarray normal):
        self.normal = normal.flatten()

    def get_indices(self):
        return self.indices
    def set_indices(self, np.ndarray indices):
        self.indices = indices.flatten()
    def get_color(self):
        return self.color
    def set_color(self, np.ndarray color):
        self.color = color.flatten()

    cdef merge_indices(self, Mesh other):
        cdef long[:] indices = np.empty((self.indices.shape[0] + other.indices.shape[0],), int)
        cdef long maxindex = 0

        for i in range(self.indices.shape[0]):
            indices[i] = self.indices[i]
            if self.indices[i] > maxindex:
                maxindex = self.indices[i]

        for j in range(other.indices.shape[0]):
            indices[j + i] = other.indices[i] + maxindex + 1

        return indices

    def fuse(self, Mesh other):
        return self.cfuse(other)
    cdef Mesh cfuse(self, Mesh other):

        cdef double[:] position = np.empty((self.position.shape[0] + other.position.shape[0],), float)
        cdef double[:] normal = np.empty((self.normal.shape[0] + other.normal.shape[0],), float)
        cdef double[:] color = np.empty((self.color.shape[0] + other.color.shape[0],), float)

        for i in range(self.position.shape[0]):
            position[i] = self.position[i]
            normal[i] = self.normal[i]
            color[i] = self.color[i]
        for j in range(other.position.shape[0]):
            position[i + j] = other.position[j]
            normal[i + j] = other.normal[j]
            color[i + j] = other.color[j]

        cdef Mesh msh = Mesh(position, normal, self.merge_indices(other))
        msh.color = color

        return msh

cdef Mesh cfuse(list[Mesh] meshes):
    cdef Mesh msh = meshes[0]
    cdef msh1 = Mesh(msh.position, msh.normal, msh.indices)
    msh1.color = msh.color

    for i in range(len(meshes) - 1):
        msh1 = msh1.fuse(meshes[i + 1])

    return msh1

def fuse(list[Mesh] meshes):
    return cfuse(meshes)
