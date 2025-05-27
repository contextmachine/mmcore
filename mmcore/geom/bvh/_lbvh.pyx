# distutils: language = c++
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
cimport cython

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy


# Inline utility functions
cdef inline int left(int i) nogil: return 2*i + 1
cdef inline int right(int i) nogil: return 2*i + 2
cdef inline int parent(int i) nogil: return (i - 1) // 2

# AABB as C struct
cdef struct AABBStruct:
    double min0, min1, min2
    double max0, max1, max2

# Initialize to infinite bounds
cdef inline void aabb_init(AABBStruct* a) noexcept nogil:
    a.min0 = a.min1 = a.min2 = 1e308
    a.max0 = a.max1 = a.max2 = -1e308

# Build AABB from M points (row-major)
cdef inline void aabb_from_points(double* pts, int M, AABBStruct* a) noexcept nogil:
    cdef double x0, x1, y0, y1, z0, z1, px, py, pz
    cdef int i
    # first point
    x0 = x1 = pts[0]
    y0 = y1 = pts[1]
    z0 = z1 = pts[2]
    for i in range(1, M):


        px = pts[3*i]; py = pts[3*i + 1]; pz = pts[3*i + 2]
        if px < x0: x0 = px
        elif px > x1: x1 = px
        if py < y0: y0 = py
        elif py > y1: y1 = py
        if pz < z0: z0 = pz
        elif pz > z1: z1 = pz
    a.min0 = x0; a.min1 = y0; a.min2 = z0
    a.max0 = x1; a.max1 = y1; a.max2 = z1

# Merge two AABBs
cdef inline void aabb_merge(const AABBStruct* a, const AABBStruct* b, AABBStruct* out) noexcept nogil:
    out.min0 = a.min0 < b.min0 and a.min0 or b.min0
    out.min1 = a.min1 < b.min1 and a.min1 or b.min1
    out.min2 = a.min2 < b.min2 and a.min2 or b.min2
    out.max0 = a.max0 > b.max0 and a.max0 or b.max0
    out.max1 = a.max1 > b.max1 and a.max1 or b.max1
    out.max2 = a.max2 > b.max2 and a.max2 or b.max2

# Intersection tests
cdef inline bint aabb_intersects(const AABBStruct* a, const AABBStruct* b) noexcept nogil:
    if a.max0 < b.min0 or a.max1 < b.min1 or a.max2 < b.min2 or \
       a.min0 > b.max0 or a.min1 > b.max1 or a.min2 > b.max2:
        return False
    return True

cdef inline bint aabb_intersects_exact(const AABBStruct* a, const AABBStruct* b) noexcept nogil:
    if a.max0 <= b.min0 or a.max1 <= b.min1 or a.max2 <= b.min2 or \
       a.min0 >= b.max0 or a.min1 >= b.max1 or a.min2 >= b.max2:
        return False
    return True

# Partition objects by centroid along longest axis
cdef void split_objects(int* objs, int n, AABBStruct* bboxes,
                         int* left_objs, int* right_objs,
                         int* left_count, int* right_count) noexcept nogil:
    cdef int i,j, axis = 0
    cdef double cmin[3]
    cdef double cen[3]
    cdef double mid
    cdef double cmax[3]
    # init from first
    cen[0] = (bboxes[objs[0]].min0 + bboxes[objs[0]].max0) * 0.5
    cen[1] = (bboxes[objs[0]].min1 + bboxes[objs[0]].max1) * 0.5
    cen[2] = (bboxes[objs[0]].min2 + bboxes[objs[0]].max2) * 0.5
    for j in range(3): cmin[j] = cmax[j] = cen[j]
    # compute centroid bounds
    for i in range(1, n):
        cen[0] = (bboxes[objs[i]].min0 + bboxes[objs[i]].max0) * 0.5
        cen[1] = (bboxes[objs[i]].min1 + bboxes[objs[i]].max1) * 0.5
        cen[2] = (bboxes[objs[i]].min2 + bboxes[objs[i]].max2) * 0.5
        for j in range(3):
            if cen[j] < cmin[j]: cmin[j] = cen[j]
            elif cen[j] > cmax[j]: cmax[j] = cen[j]
    # choose axis
    axis = 0
    if cmax[1] - cmin[1] > cmax[axis] - cmin[axis]: axis = 1
    if cmax[2] - cmin[2] > cmax[axis] - cmin[axis]: axis = 2
    mid = 0.5 * (cmin[axis] + cmax[axis])
    # partition
    cdef int lc = 0, rc = 0
    for i in range(n):
        cen[axis] = (axis == 0 and (bboxes[objs[i]].min0 + bboxes[objs[i]].max0) * 0.5) or \
                    (axis == 1 and (bboxes[objs[i]].min1 + bboxes[objs[i]].max1) * 0.5) or \
                    (bboxes[objs[i]].min2 + bboxes[objs[i]].max2) * 0.5
        if cen[axis] < mid:
            left_objs[lc] = objs[i]; lc += 1
        else:
            right_objs[rc] = objs[i]; rc += 1
    # fallback
    if lc == 0 or rc == 0:
        lc = n // 2; rc = n - lc
        for i in range(n):
            if i < lc: left_objs[i] = objs[i]
            else: right_objs[i - lc] = objs[i]
    left_count[0] = lc; right_count[0] = rc

# BVH node as C struct
cdef struct BVHNodeStruct:
    AABBStruct bbox
    int left, right, object_index



cdef class BVH:
    cdef AABBStruct* bboxes
    cdef BVHNodeStruct* nodes
    cdef int n_objects
    cdef int max_leaf_objects
    cdef Py_ssize_t total_nodes
    cdef list leaf_indices
    cdef int root_index
    cdef double[:, :, ::1] _points
    
    def __repr__(self):
        # Safe repr hides cdef pointers
        return f"<BVH root={self.root_index} leaves={len(self.leaf_indices)}>"

    def __getstate__(self):
        # Only pickle safe Python-level state
        return {"root_index": self.root_index,
                "leaf_indices": self.leaf_indices}

    def __setstate__(self, state):
        self.root_index = state.get("root_index", -1)
        self.leaf_indices = state.get("leaf_indices", [])
        
    def __reduce__(self):
        return (self.__class__, (), self.__getstate__())
    
    
    def __init__(self):
        self.bboxes = NULL
        self.nodes = NULL
        self.n_objects = 0
        self.max_leaf_objects = 1
        self.leaf_indices = []
        self.root_index = -1

    cdef int _build_internal(self, int* objs, int n, int current_index) noexcept nogil:
        cdef BVHNodeStruct* node = &self.nodes[current_index]
        if n <= self.max_leaf_objects:
            memcpy(&node.bbox, &self.bboxes[objs[0]], sizeof(AABBStruct))
            node.left = node.right = -1
            node.object_index = objs[0]
            return current_index
        # split
        cdef int* left_objs = <int*>malloc(n * sizeof(int))
        cdef int* right_objs = <int*>malloc(n * sizeof(int))
        cdef int lc_arr[1]
        cdef int rc_arr[1]
        split_objects(objs, n, self.bboxes, left_objs, right_objs, &lc_arr[0], &rc_arr[0])
        cdef int li = lc_arr[0], ri = rc_arr[0]
        cdef int lidx = self._build_internal(left_objs, li, left(current_index))
        cdef int ridx = self._build_internal(right_objs, ri, right(current_index))
        aabb_merge(&self.nodes[lidx].bbox, &self.nodes[ridx].bbox, &node.bbox)
        node.left = lidx; node.right = ridx; node.object_index = -1
        free(left_objs); free(right_objs)
        return current_index

    cpdef build(self, double[:, :, ::1] points, int max_objects_in_leaf=1):
        self._points = points
        cdef Py_ssize_t N = points.shape[0]
        cdef Py_ssize_t M = points.shape[1]
        self.n_objects = N
        self.max_leaf_objects = max_objects_in_leaf
        self.total_nodes = 2 * N - 1
        # allocate aabbs
        self.bboxes = <AABBStruct*>malloc(N * sizeof(AABBStruct))
        for i in range(N):
            aabb_from_points(&points[i, 0, 0], M, &self.bboxes[i])
        # allocate nodes
        self.nodes = <BVHNodeStruct*>malloc(self.total_nodes * sizeof(BVHNodeStruct))
        for i in range(self.total_nodes):
            aabb_init(&self.nodes[i].bbox)
            self.nodes[i].left = self.nodes[i].right = -1
            self.nodes[i].object_index = -1
        # object indices
        cdef int* objs = <int*>malloc(N * sizeof(int))
        for i in range(N): objs[i] = i
        self.root_index = self._build_internal(objs, N, 0)
        free(objs)
        # collect leaves
        self.leaf_indices = []
        for i in range(self.total_nodes):
            if self.nodes[i].object_index >= 0:
                self.leaf_indices.append(i)
        

    cpdef dict get_leaf_overlaps(self, bint exact=True):
        cdef dict result = {}
        cdef int i, j
        for i in self.leaf_indices:
            result[i] = []
        for idx_i in range(len(self.leaf_indices)):
            i = self.leaf_indices[idx_i]
            for idx_j in range(idx_i + 1, len(self.leaf_indices)):
                j = self.leaf_indices[idx_j]
                if exact:
                    if aabb_intersects_exact(&self.nodes[i].bbox, &self.nodes[j].bbox):
                        result[i].append(j)
                        result.setdefault(j, []).append(i)
                else:
                    if aabb_intersects(&self.nodes[i].bbox, &self.nodes[j].bbox):
                        result[i].append(j)
                        result.setdefault(j, []).append(i)
        return result

    def indices_in_bbox(self, double min0, double min1, double min2, double max0, double max1, double max2):
        """
        Return a list of (object_index, point_local_index) for all points whose coordinates lie within the given axis-aligned box.
        """
        cdef list result = []
        cdef int idx, pi,i,obj
        cdef double x,y,z
        # Traverse nodes
        cdef BVHNodeStruct *node
        cdef AABBStruct query
        for i in range(len(self.leaf_indices)):
            idx=self.leaf_indices[i]
            node = &self.nodes[idx]
            # Check leaf bbox intersects query box
           
            query.min0 = min0;
            query.min1 = min1;
            query.min2 = min2
            query.max0 = max0;
            query.max1 = max1;
            query.max2 = max2
            if not aabb_intersects_exact(&node.bbox, &query):
                continue
            # Node.object_index holds the object id
            obj = node.object_index
            # access points via stored memoryview
            for pi in range(self._M):
                x = self._points[obj, pi, 0]
                y = self._points[obj, pi, 1]
                z = self._points[obj, pi, 2]
                if x >= min0 and x <= max0 and y >= min1 and y <= max1 and z >= min2 and z <= max2:
                    result.append((obj, pi))
        return result
# Python API

    
