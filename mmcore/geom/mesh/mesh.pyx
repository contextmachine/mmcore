import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float64_t DTYPE_t
ctypedef long INDICES_t

cdef class Mesh:
    cdef  np.ndarray position
    cdef  np.ndarray normal
    cdef  np.ndarray color
    cdef  np.ndarray indices

    def __cinit__(self, position, normal, indices):
        self.color = np.ones((position.shape[0],))

        self.position = position
        self.normal = normal
        self.indices = indices
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
        cdef np.ndarray[INDICES_t, ndim=1] indices = np.empty((self.indices.shape[0] + other.indices.shape[0],), int)
        cdef INDICES_t maxindex = 0

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

        cdef np.ndarray[DTYPE_t, ndim=1] position = np.empty((self.position.shape[0] + other.position.shape[0],), float)
        cdef np.ndarray[DTYPE_t, ndim=1] normal = np.empty((self.normal.shape[0] + other.normal.shape[0],), float)
        cdef np.ndarray[DTYPE_t, ndim=1] color = np.empty((self.color.shape[0] + other.color.shape[0],), float)

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
