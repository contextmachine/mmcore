#include <array>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <stdexcept>

template<typename T>
using Vec3 = std::array<T, 3>;

using Vec3f = Vec3<float>;
using Vec3d = Vec3<double>;

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
inline int supportVector(const std::vector<Vec3<T>>& vertices, const Vec3<T>& d) {
    T highest = -std::numeric_limits<T>::max();
    int supportIndex = -1;

    for (size_t i = 0; i < vertices.size(); ++i) {
        T dotValue = dot(vertices[i], d);
        if (dotValue > highest) {
            highest = dotValue;
            supportIndex = static_cast<int>(i);
        }
    }

    return supportIndex;
}

template<typename T>
bool handleSimplex(std::vector<Vec3<T>>& simplex, Vec3<T>& d, T tol) {
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
                    return handleSimplex(simplex, d, tol);
                }
            } else {
                if (dot(cross(ab, abc), ao) > tol) {
                    simplex.erase(simplex.begin());
                    return handleSimplex(simplex, d, tol);
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
            const auto& dPoint = simplex[0];

            Vec3<T> ab = b - a;
            Vec3<T> ac = c - a;
            Vec3<T> ad = dPoint - a;
            Vec3<T> ao = {-a[0], -a[1], -a[2]};

            Vec3<T> abc = cross(ab, ac);
            Vec3<T> acd = cross(ac, ad);
            Vec3<T> adb = cross(ad, ab);

            if (dot(abc, ao) > 0) {
                simplex.erase(simplex.begin());
                return handleSimplex(simplex, d, tol);
            } else if (dot(acd, ao) > 0) {
                simplex.erase(simplex.begin() + 1);
                return handleSimplex(simplex, d, tol);
            } else if (dot(adb, ao) > 0) {
                simplex.erase(simplex.begin() + 2);
                return handleSimplex(simplex, d, tol);
            } else {
                return true;
            }
        }
    }
    return false;
}

template<typename T>
bool gjk_collision_detection(const std::vector<Vec3<T>>& vertices1, const std::vector<Vec3<T>>& vertices2, T tol, size_t maxIter = 0) {
    if (vertices1.empty() || vertices2.empty()) {
        throw std::invalid_argument("Input vertex sets cannot be empty");
    }

    if (maxIter == 0) {
        maxIter = std::min(vertices1.size() * vertices2.size(), static_cast<size_t>(std::numeric_limits<size_t>::max()));
    }

        const size_t rows = vertices1.size();
        const size_t cols = vertices2.size();
        std::vector<bool> visited(rows * cols, false);
        std::vector<Vec3<T>> cache(rows * cols);

        std::vector<Vec3<T>> simplex;
        simplex.reserve(4);  // Pre-allocate space for efficiency

        auto support = [&](const Vec3<T>& d) -> Vec3<T> {
            int i = supportVector(vertices1, d);
            int j = supportVector(vertices2, {-d[0], -d[1], -d[2]});
            return vertices1[i] - vertices2[j];
        };

        Vec3<T> d = {1, 0, 0};
        auto newPoint = support(d);
        size_t index = static_cast<size_t>(supportVector(vertices1, d)) * cols + static_cast<size_t>(supportVector(vertices2, {-d[0], -d[1], -d[2]}));
        visited[index] = true;
        cache[index] = newPoint;
        simplex.push_back(newPoint);
        d = {-newPoint[0], -newPoint[1], -newPoint[2]};

        for (size_t iter = 0; iter < maxIter; ++iter) {
            newPoint = support(d);
            index = static_cast<size_t>(supportVector(vertices1, d)) * cols + static_cast<size_t>(supportVector(vertices2, {-d[0], -d[1], -d[2]}));
            
            if (visited[index]) {
                auto vec = cache[index] - newPoint;
                auto normSquared = sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
                if (normSquared > tol) {
                    return false;
                }
            } else {
                visited[index] = true;
                cache[index] = newPoint;
            }

            if (dot(newPoint, d) < 0) {
                return false;
            }

            simplex.push_back(newPoint);

            if (handleSimplex(simplex, d, tol)) {
                return true;
            }
        }

        return false;
    }