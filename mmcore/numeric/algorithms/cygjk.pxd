
# distutils: language = c++
cimport cython
from libcpp.vector cimport vector
from libcpp cimport pair,bool

# Declare the necessary imports for C++ standard headers


cdef extern from "_gjk.cpp" nogil:  # Use "*" to include the custom C++ header/code directly
    # Declare the Vec3 template
 
    cdef cppclass Vec3[T]:
        ctypedef T value_type
        ctypedef size_t size_type
        T& operator[](size_type)
        void assign(size_type, const T&)
        void assign[InputIt](InputIt, InputIt) except +
        
        
    bool gjk_collision_detection(const vector[Vec3[double]]& vertices1, const vector[Vec3[double]]& vertices2, double tol, size_t max_iter);

    

    