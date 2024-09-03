#include <array>
#include <cstdint>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__AVX__)
#include <immintrin.h>
#endif


template<typename T>
using Vec3 = std::array<T, 3>;

typedef Vec3<float> Vec3_f;
typedef Vec3<double> Vec3_d;


template<typename T>
inline T dot(const Vec3<T>& a, const Vec3<T>& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}


template<typename T>
inline Vec3<T> cross(const Vec3<T>& a, const Vec3<T>& b) {
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}

template<typename T>
inline Vec3<T> operator-(const Vec3<T>& a, const Vec3<T>& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

template<typename T>
inline int support_vector(const std::vector<Vec3<T>>& vertices, const Vec3<T>& d, Vec3<T>& support) {
    T highest = -std::numeric_limits<T>::max();
    support[0]=0;
    support[1]=0;
    support[2]=0;
    int support_i = -1;

    for (size_t i = 0; i < vertices.size(); ++i) {
        T dot_value = dot(vertices[i], d);
        if (dot_value > highest) {
            highest = dot_value;
            support[0] = vertices[i][0];
            support[1] = vertices[i][1];
            support[2] = vertices[i][2];
            support_i = i;
        }
    }

    return support_i;
}

template<typename T>
bool handle_simplex(std::vector<Vec3<T>>& simplex, Vec3<T>& d, T tol) {
    switch (simplex.size()) {
        case 2: {
            const auto& a = simplex[1];
            const auto& b = simplex[0];
            Vec3<T> ab = b - a;
            Vec3<T> ao = {-a[0], -a[1], -a[2]};

            if (dot(ab, ao) > tol) {
                d = cross(cross(ab, ao), ab);
            } else {
                simplex.erase(simplex.begin());
                d = ao;
            }
            break;
        }
        case 3: {
            const auto& a = simplex[2];
            const auto& b = simplex[1];
            const auto& c = simplex[0];
            Vec3<T> ab = b - a;
            Vec3<T> ac = c - a;
            Vec3<T> ao = {-a[0], -a[1], -a[2]};
            Vec3<T> abc = cross(ab, ac);

            if (dot(cross(abc, ac), ao) > tol) {
                if (dot(ac, ao) > 0) {
                    simplex.erase(simplex.begin() + 1);
                    d = cross(cross(ac, ao), ac);
                } else {
                    simplex.erase(simplex.begin());
                    return handle_simplex<T>(simplex, d, tol);
                }
            } else {
                if (dot(cross(ab, abc), ao) > tol) {
                    simplex.erase(simplex.begin());
                    return handle_simplex<T>(simplex, d, tol);
                } else {
                    if (dot(abc, ao) > tol) {
                        d = abc;
                    } else {
                        std::swap(simplex[0], simplex[1]);
                        d = {-abc[0], -abc[1], -abc[2]};
                    }
                }
            }
            break;
        }
        case 4: {
            const auto& a = simplex[3];
            const auto& b = simplex[2];
            const auto& c = simplex[1];
            const auto& d_point = simplex[0];

            Vec3<T> ab = b - a;
            Vec3<T> ac = c - a;
            Vec3<T> ad = d_point - a;
            Vec3<T> ao = {-a[0], -a[1], -a[2]};

            Vec3<T> abc = cross<T>(ab, ac);
            Vec3<T> acd = cross<T>(ac, ad);
            Vec3<T> adb = cross<T>(ad, ab);

            if (dot(abc, ao) > tol) {
                simplex.erase(simplex.begin());
                return handle_simplex<T>(simplex, d, tol);
            } else if (dot(acd, ao) > tol) {
                simplex.erase(simplex.begin() + 1);
                return handle_simplex<T>(simplex, d, tol);
            } else if (dot(adb, ao) > tol) {
                simplex.erase(simplex.begin() + 2);
                return handle_simplex<T>(simplex, d, tol);
            } else {
                return true;
            }
        }
    }
    return false;
}

inline bool isVisited(bool* arr, const size_t cols, const size_t i, const size_t j) {
        return arr[i * cols + j];
    };
inline Vec3_d& getVisited(Vec3_d* tt, const size_t cols, const size_t i, const size_t j) {
        return tt[i * cols + j];
    };
inline void setVisited(bool* arr, Vec3<double>* tt, const size_t cols, const size_t i, const size_t j, const Vec3_d& val) {
        arr[i * cols + j]=true;
        tt[i*cols+j]=val;
};
bool gjk_collision_detection(const std::vector<Vec3<double>>& vertices1, const std::vector<Vec3<double>>& vertices2, double tol , size_t max_iter) {
    if (max_iter == 0) {
        max_iter = vertices1.size() * vertices2.size();
    }
    const size_t rows=vertices1.size();
    const size_t cols=vertices2.size();
    bool* arr= new bool[rows * cols];
    Vec3<double>* tt= new Vec3<double>[rows * cols];

    std::vector<Vec3<double>> simplex;
    simplex.reserve(4);  // Pre-allocate space for efficiency

    auto support = [&](const Vec3<double>& d, size_t &i_index, size_t &j_index) -> Vec3<double> {
        Vec3<double> p1;
        Vec3<double> p2;
        i_index = support_vector<double>(vertices1, d, p1);
        j_index = support_vector<double>(vertices2, {-d[0], -d[1], -d[2]}, p2);
      
        return p1 - p2;
    };
    size_t i,j;
    Vec3<double> d = {0, 0, 1};
    auto new_point=support(d, i,j);
    setVisited(arr,tt,cols,i,j, new_point);
    simplex.push_back(new_point);
    d = {-simplex[0][0], -simplex[0][1], -simplex[0][2]};
   
    for (size_t iter = 0; iter < max_iter; ++iter) {
        new_point = support(d,i,j );
        if (isVisited(arr,cols, i,j)){
            auto vec=new_point-getVisited(tt, cols, i,j);

            
            if (dot(vec, vec)==0){
                delete tt;
                delete arr;
                return true;
            }
        } else{
            setVisited(arr,tt, cols,i,j, new_point);
        }


        if (dot(new_point, d) < 0) {
            delete tt;
            delete arr;
            return false;
        }

        simplex.push_back(new_point);

        if (handle_simplex<double>(simplex, d, tol)) {
            delete tt;
            delete arr;
            return true;
        }
    }

    delete tt;
    delete arr;

    return false;
}


