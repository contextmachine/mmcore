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

// ---------------------------------------------------------------------------
// basic math helpers
// ---------------------------------------------------------------------------
template<typename T>
inline T dot(const Vec3<T>& a, const Vec3<T>& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template<typename T>
inline Vec3<T> cross(const Vec3<T>& a, const Vec3<T>& b)
{
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}

template<typename T>
inline Vec3<T> operator-(const Vec3<T>& a, const Vec3<T>& b)
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

// ---------------------------------------------------------------------------
// support point helpers
// ---------------------------------------------------------------------------
template<typename T>
inline int supportVector(const std::vector<Vec3<T>>& vertices,
                         const Vec3<T>&                 d)
{
    T   highest      = -std::numeric_limits<T>::max();
    int supportIndex = -1;

    for (size_t i = 0; i < vertices.size(); ++i)
    {
        T dotValue = dot(vertices[i], d);
        if (dotValue > highest)
        {
            highest      = dotValue;
            supportIndex = static_cast<int>(i);
        }
    }
    return supportIndex;
}

// ---------------------------------------------------------------------------
// Roundoff envelope for the SIGN of dot(u, v).
//
// The Voronoi-region tests below ask "is the origin on the positive side?",
// which is a SIGN question about a length^2 quantity.  They used to be
// compared against the caller's `tol` -- a LENGTH -- so the verdict depended
// on the model's size: for geometry of extent s the dot products are ~s^2,
// and once s^2 fell below tol every test took the wrong branch, the search
// cycled, and the driver fell out of its loop reporting "separated".
// Measured cliff: extent <= ~10*tol.  (The tetrahedron case below always
// used the scale-invariant `> 0`, which is why only some cases were wrong.)
//
// The correct bar is zero, widened by the roundoff of the dot itself:
// three products and two adds, each relative to |u||v|.  Scale-invariant by
// construction, and independent of any caller tolerance.
// ---------------------------------------------------------------------------
template<typename T>
inline T dotSignMargin(const Vec3<T>& u, const Vec3<T>& v)
{
    const T eps = std::numeric_limits<T>::epsilon();
    return T(8) * eps * std::sqrt(dot(u, u)) * std::sqrt(dot(v, v));
}

// ---------------------------------------------------------------------------
// simplex handling
// ---------------------------------------------------------------------------
// `tol` is retained in the signature for source compatibility but is NO
// LONGER used for the sign tests -- see dotSignMargin above.
template<typename T>
bool handleSimplex(std::vector<Vec3<T>>& simplex, Vec3<T>& d, T tol)
{
    switch (simplex.size())
    {
        //------------------------------------------------------------------
        // line segment
        //------------------------------------------------------------------
        case 2:
        {
            const auto& a  = simplex[1];
            const auto& b  = simplex[0];
            Vec3<T>     ab = b - a;
            Vec3<T>     ao = {-a[0], -a[1], -a[2]};

            if (dot(ab, ao) > dotSignMargin(ab, ao))
            {
                d = cross(cross(ab, ao), ab);
            }
            else
            {
                simplex.erase(simplex.begin());  // keep only A
                d = ao;
            }
            break;
        }

        //------------------------------------------------------------------
        // triangle
        //------------------------------------------------------------------
        case 3:
        {
            const auto& a  = simplex[2];
            const auto& b  = simplex[1];
            const auto& c  = simplex[0];
            Vec3<T>     ab = b - a;
            Vec3<T>     ac = c - a;
            Vec3<T>     ao = {-a[0], -a[1], -a[2]};
            Vec3<T>     abc = cross(ab, ac);

            // test region AC
            if (dot(cross(abc, ac), ao) > dotSignMargin(cross(abc, ac), ao))
            {
                if (dot(ac, ao) > 0)
                {
                    simplex.erase(simplex.begin() + 1);      // remove B
                    d = cross(cross(ac, ao), ac);
                }
                else
                {
                    simplex.erase(simplex.begin());           // remove C
                    return handleSimplex(simplex, d, tol);    // recurse to segment case
                }
            }
            else
            {
                // test region AB
                if (dot(cross(ab, abc), ao) > dotSignMargin(cross(ab, abc), ao))
                {
                    simplex.erase(simplex.begin());           // remove C
                    return handleSimplex(simplex, d, tol);
                }
                else
                {
                    // origin is above or below the triangle
                    if (dot(abc, ao) > dotSignMargin(abc, ao))
                    {
                        d = abc;
                    }
                    else             // below: flip winding
                    {
                        std::swap(simplex[0], simplex[1]);
                        d = {-abc[0], -abc[1], -abc[2]};
                    }
                }
            }
            break;
        }

        //------------------------------------------------------------------
        // tetrahedron
        //------------------------------------------------------------------
        case 4:
        {
            const auto& a       = simplex[3];
            const auto& b       = simplex[2];
            const auto& c       = simplex[1];
            const auto& dPoint  = simplex[0];

            Vec3<T> ab  = b - a;
            Vec3<T> ac  = c - a;
            Vec3<T> ad  = dPoint - a;
            Vec3<T> ao  = {-a[0], -a[1], -a[2]};

            Vec3<T> abc = cross(ab, ac);
            Vec3<T> acd = cross(ac, ad);
            Vec3<T> adb = cross(ad, ab);

            if (dot(abc, ao) > 0)
            {
                // outside ABC – keep A,B,C  (drop D)
                simplex.erase(simplex.begin());       // remove index 0 (dPoint)
                return handleSimplex(simplex, d, tol);
            }
            else if (dot(acd, ao) > 0)
            {
                // outside ACD – keep A,C,D  (drop B)
                simplex.erase(simplex.begin() + 2);   // *** FIX: remove index 2 (b) ***
                return handleSimplex(simplex, d, tol);
            }
            else if (dot(adb, ao) > 0)
            {
                // outside ADB – keep A,D,B  (drop C)
                simplex.erase(simplex.begin() + 1);   // *** FIX: remove index 1 (c) ***
                return handleSimplex(simplex, d, tol);
            }
            else
            {
                // origin is inside the tetrahedron
                return true;
            }
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// GJK driver
// ---------------------------------------------------------------------------
template<typename T>
bool gjk_collision_detection(const std::vector<Vec3<T>>& vertices1,
                             const std::vector<Vec3<T>>& vertices2,
                             T                           tol,
                             size_t                      maxIter = 0)
{
    if (vertices1.empty() || vertices2.empty())
        throw std::invalid_argument("Input vertex sets cannot be empty");

    if (maxIter == 0)
        maxIter = std::min(vertices1.size() * vertices2.size(),
                           static_cast<size_t>(std::numeric_limits<size_t>::max()));

    const size_t rows = vertices1.size();
    const size_t cols = vertices2.size();
    std::vector<bool>      visited(rows * cols, false);
    std::vector<Vec3<T>>   cache(rows * cols);

    std::vector<Vec3<T>> simplex;
    simplex.reserve(4);

    // ------------------------------------------------------------------
    // local support functor
    // ------------------------------------------------------------------
    auto support = [&](const Vec3<T>& dVec) -> Vec3<T>
    {
        int i = supportVector(vertices1, dVec);
        int j = supportVector(vertices2, {-dVec[0], -dVec[1], -dVec[2]});
        return vertices1[i] - vertices2[j];
    };

    // ------------------------------------------------------------------
    // initial search direction and first point
    // ------------------------------------------------------------------
    Vec3<T> d = {1, 0, 0};
    Vec3<T> newPoint = support(d);

    size_t i0 = supportVector(vertices1, d);
    size_t j0 = supportVector(vertices2, {-d[0], -d[1], -d[2]});
    size_t idx = i0 * cols + j0;

    visited[idx] = true;
    cache[idx]   = newPoint;

    simplex.push_back(newPoint);
    d = {-newPoint[0], -newPoint[1], -newPoint[2]};

    // ------------------------------------------------------------------
    // main loop
    // ------------------------------------------------------------------
    for (size_t iter = 0; iter < maxIter; ++iter)
    {
        newPoint = support(d);

        size_t i = supportVector(vertices1, d);
        size_t j = supportVector(vertices2, {-d[0], -d[1], -d[2]});
        idx      = i * cols + j;

        if (visited[idx])
        {
            // The support pair (i, j) repeated.  For a DISJOINT pair this is
            // the normal convergence signal -- the support optimum has been
            // reached -- and the separating-axis test below is what turns it
            // into a sound negative, so this must fall through rather than
            // conclude anything on its own.  (The former `len > tol` test
            // here was unreachable in practice: repeating (i, j) reproduces
            // the same difference point, so len is 0.)
        }
        else
        {
            visited[idx] = true;
            cache[idx]   = newPoint;
        }

        if (dot(newPoint, d) < 0)
            return false;    // support point failed to pass origin

        simplex.push_back(newPoint);

        if (handleSimplex(simplex, d, tol))
            return true;     // origin is inside the simplex → collision
    }
    // Ran out of iterations: UNKNOWN, not "separated".  The only sound
    // negative this routine can return is the separating-axis certificate
    // above (dot(newPoint, d) < 0).  Exhaustion used to return false, so a
    // starved max_iter reported deeply overlapping hulls as separated --
    // measured 200/200 false negatives at max_iter 1-2.  "Unknown => not
    // separated" keeps every caller's `if (!gjk(...)) prune;` sound.
    return true;
}