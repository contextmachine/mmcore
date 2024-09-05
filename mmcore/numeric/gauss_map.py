# mmcore/numeric/gauss_map.py
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from mmcore.geom.nurbs import NURBSSurface, split_surface_v, split_surface_u, subdivide_surface
from mmcore.numeric.algorithms.gjk import gjk_collision_detection as gjk_collision_detection
from mmcore.numeric.algorithms.quicksort import unique
from mmcore.numeric.monomial import bezier_to_monomial, monomial_to_bezier
from mmcore.numeric.vectors import unit, cartesian_to_spherical, spherical_to_cartesian, scalar_dot, scalar_norm
from mmcore.numeric.algorithms.cygjk import gjk


def is_flat(surf, u_min, u_max, v_min, v_max, tolerance=1e-3):

    corner_points = [surf(u_min, v_min), surf(u_min, v_max), surf(u_max, v_min), surf(u_max, v_max)]
    center_point = surf((u_min + u_max) / 2, (v_min + v_max) / 2)

    # Compute the plane defined by the three corner points
    normal = np.cross(corner_points[1] - corner_points[0], corner_points[2] - corner_points[0])
    normal = normal / scalar_norm(normal)
    d = -scalar_dot(normal, corner_points[0])

    # Check the distance of the center point from the plane
    distance = np.abs(scalar_dot(normal, center_point) + d)
    # Define an appropriate tolerance for flatness

    return distance < tolerance


def decompose_surface(surface, decompose_dir="uv"):
    def decompose_direction(srf, idx):
        srf_list = []
        knots = srf.knots_u if idx == 0 else srf.knots_v
        degree = srf.degree[idx]
        unique_knots = sorted(set(knots[degree + 1: -(degree + 1)]))

        while unique_knots:
            knot = unique_knots[0]
            if idx == 0:
                srfs = split_surface_u(srf, knot)
            else:
                srfs = split_surface_v(srf, knot)
            srf_list.append(srfs[0])
            srf = srfs[1]
            unique_knots = unique_knots[1:]
        srf_list.append(srf)
        return srf_list

    if not isinstance(surface, NURBSSurface):
        raise ValueError("Input must be an instance of NURBSSurface class")

    surf = surface.copy()

    if decompose_dir == "u":
        return decompose_direction(surf, 0)
    elif decompose_dir == "v":
        return decompose_direction(surf, 1)
    elif decompose_dir == "uv":
        multi_surf = []
        surfs_u = decompose_direction(surf, 0)
        for sfu in surfs_u:
            multi_surf += decompose_direction(sfu, 1)
        return multi_surf
    else:
        raise ValueError(
            f"Cannot decompose in {decompose_dir} direction. Acceptable values: u, v, uv"
        )


def compute_partial_derivative(coeffs, variable):
    """Compute partial derivative of monomial coefficients."""
    n, m, dim = coeffs.shape
    deriv = np.zeros((n, m, dim))
    if variable == "u":
        for i in range(1, n):
            deriv[i - 1, :, :] = i * coeffs[i, :, :]
    elif variable == "v":
        for j in range(1, m):
            deriv[:, j - 1, :] = j * coeffs[:, j, :]
    return deriv


def cross_product(a, b):
    """Compute cross product of two 3D polynomial patches."""
    n, m, _ = a.shape
    result = np.zeros((2 * n - 1, 2 * m - 1, 3))
    for i in range(n):
        for j in range(m):
            for k in range(n):
                for l in range(m):
                    result[i + k, j + l, 0] += (
                            a[i, j, 1] * b[k, l, 2] - a[i, j, 2] * b[k, l, 1]
                    )
                    result[i + k, j + l, 1] += (
                            a[i, j, 2] * b[k, l, 0] - a[i, j, 0] * b[k, l, 2]
                    )
                    result[i + k, j + l, 2] += (
                            a[i, j, 0] * b[k, l, 1] - a[i, j, 1] * b[k, l, 0]
                    )

    return result


def is_bezier(surface: NURBSSurface):
    kv, ku = unique(surface.knots_u).shape[0], unique(surface.knots_u).shape[0]
    if kv.shape[0] < 2 or ku.shape[0] < 2:
        raise ValueError("Degenerated patch")

    return kv.shape[0] == 2 and ku.shape[0] == 2


def normalize_polynomial(v, epsilon=1e-10):
    """Normalize a 3D vector polynomial with improved stability."""
    norm_squared = v[:, :, 0] ** 2 + v[:, :, 1] ** 2 + v[:, :, 2] ** 2
    max_norm = np.max(norm_squared)
    if max_norm < epsilon:
        return np.zeros_like(v)
    norm = np.sqrt(norm_squared / max_norm)
    return v / (norm[:, :, np.newaxis] * np.sqrt(max_norm))


"""
# Example usage and verification
bezier_patch = np.array(
    [
        [[0, 0, 0], [0, 1, 0], [0, 2, 0]],
        [[1, 0, 1], [1, 1, 2], [1, 2, 1]],
        [[2, 0, 0], [2, 1, 0], [2, 2, 0]],
    ]
)

gauss_map, normalized_gauss_map = compute_gauss_map(bezier_patch)
print("Gauss map control points:")
print(gauss_map)

"""


def compute_gauss_mapw(control_points, weights=None):
    """Compute the Gauss map for a rational Bézier patch."""
    if weights is not None:
        # Convert to homogeneous coordinates
        control_points = control_points * weights[:, :, np.newaxis]
        control_points = np.concatenate([control_points, weights[:, :, np.newaxis]], axis=-1)

    F = bezier_to_monomial(control_points)
    Fu = compute_partial_derivative(F, "u")
    Fv = compute_partial_derivative(F, "v")

    if weights is not None:
        # Handle rational case
        w = F[:, :, -1]
        Fu = Fu[:, :, :3] * w[:, :, np.newaxis] - F[:, :, :3] * Fu[:, :, -1:]
        Fv = Fv[:, :, :3] * w[:, :, np.newaxis] - F[:, :, :3] * Fv[:, :, -1:]
        Fu = Fu / (w ** 2)[:, :, np.newaxis]
        Fv = Fv / (w ** 2)[:, :, np.newaxis]

    N = cross_product(Fu, Fv)
    #N_normalized = normalize_polynomial(N)
    gauss_map = monomial_to_bezier(N)

    return gauss_map


def compute_gauss_map(control_points):
    """Compute the Gauss map for a Bézier patch with degree elevation."""
    F = bezier_to_monomial(control_points)
    Fu = compute_partial_derivative(F, "u")
    Fv = compute_partial_derivative(F, "v")

    N = cross_product(Fu, Fv)

    # N_normalized = normalize_polynomial(N)
    # print(N_normalized)
    # N_normalized[np.isnan(N_normalized)]=0.
    gauss_map = monomial_to_bezier(N)

    return gauss_map


class GaussMap:
    def __init__(self, mp: NURBSSurface, surf: NURBSSurface):
        self.surface = surf
        self._map = mp

        self._polar_map = None
        self._polar_convex_hull = None
        self.children = []
        self.bezier_patches = []
        self.compute()

    @classmethod
    def from_surf(cls, surf):
        _map = compute_gauss_map(np.array(surf.control_points))
        #print((_map.tolist(),np.array(surf.control_points).tolist()))
        # Compute convex hull
        return cls(NURBSSurface(np.array(unit(_map.reshape((-1, 3)))).reshape(_map.shape),
                                (_map.shape[0] - 1, _map.shape[1] - 1)), surf)

    def subdivide(self):
        srf = subdivide_surface(self.surface)
        mp = subdivide_surface(self._map)
        if len(self.children)==0:
            self.children = []
            for i in range(4):
                f = mp[i]
                s = srf[i]

                f.normalize_knots()
                s.normalize_knots()
                self.children.append(GaussMap(f, s))

            return  self.children
        else:
            return self.children

    def compute(self):
        # Convert NURBS to Bézier patches

        #self.bezier_patches = decompose_surface(self.surface)
        #_map=compute_gauss_map(np.array(self.surface.control_points))
        #self._map=NURBSSurface(_map,(_map.shape[0]-1,_map.shape[1]-1))
        #_polar_map = cartesian_to_spherical(unit(_map.control_points_flat))
        #_polar_convex_hull = ConvexHull(_polar_map, qhull_options='QJ')
        # Compute Gauss map for each Bézier patch
        #gauss_maps = []
        #for patch in self.bezier_patches:
        #    gm = compute_gauss_map(np.array(patch.control_points))
        #    gm=np.array(unit(gm.reshape((-1,3))))
        #    gauss_maps.append(gm)

        # Combine Gauss maps

        # Compute polar representation
        #self._polar_map = cartesian_to_spherical(unit(self._map.control_points_flat))

        # Compute convex hull
        self._polar_convex_hull = ConvexHull(np.array(unit(self._map.control_points_flat)), qhull_options='QJ')

    def bounds(self):
        """Compute bounds on the Gauss map."""
        return np.array(self._polar_convex_hull.points[self._polar_convex_hull.vertices])

    def intersects(self, other: GaussMap):
        """Check if this Gauss map intersects with another."""

        return gjk(self.bounds(), other.bounds())


def linear_program_solver(c, A_ub, b_ub, A_eq, b_eq):
    """
    Solve a linear programming problem using scipy's linprog function.
    """
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='highs')
    return res.x if res.success else None


def separate_gauss_maps(gm1: GaussMap, gm2: GaussMap):
    """
    Attempt to find separating vectors P1 and P2 for two Gauss maps.

    Returns:
        P1, P2: np.ndarray or None, None if separation is not possible
    """
    N1 = gm1.bounds()
    N2 = gm2.bounds()

    # First, try to find P1
    P1 = find_separating_vector(N1, N2)
    if P1 is None:
        return None, None

    # If P1 is found, try to find P2
    P2 = find_common_side_vector(N1, N2)
    if P2 is None:
        return None, None

    return P1, P2


def find_separating_vector(N1, N2):
    """
    Find a vector P1 that satisfies:
    P1 · n1 > 0 for all n1 in N1
    P1 · n2 < 0 for all n2 in N2
    """
    m, n = len(N1), len(N2)

    # Set up the linear programming problem
    c = [0, 0, 0, 1]  # Minimize epsilon

    A_ub = np.zeros((m + n, 4))
    A_ub[:m, :3] = -N1  # For N1: -P1 · n1 + epsilon <= 0
    A_ub[:m, 3] = -1
    A_ub[m:, :3] = N2  # For N2: P1 · n2 + epsilon <= 0
    A_ub[m:, 3] = -1

    b_ub = np.zeros(m + n)

    A_eq = np.array([[0, 0, 0, 1]])  # epsilon <= 0
    b_eq = np.array([0])

    # Solve the linear programming problem
    result = linear_program_solver(c, A_ub, b_ub, A_eq, b_eq)

    if result is not None:
        P1 = result[:3]
        epsilon = result[3]
        if scalar_norm(P1) > 1e-6 and epsilon < 0:  # Check if the solution is valid
            return P1 / scalar_norm(P1)  # Normalize P1

    return None


def find_common_side_vector(N1, N2):
    """
    Find a vector P2 that satisfies:
    P2 · n1 > 0 for all n1 in N1
    P2 · n2 > 0 for all n2 in N2
    """
    m, n = len(N1), len(N2)

    # Set up the linear programming problem
    c = [0, 0, 0, 1]  # Minimize epsilon

    A_ub = np.zeros((m + n, 4))
    A_ub[:m, :3] = -N1  # For N1: -P2 · n1 + epsilon <= 0
    A_ub[:m, 3] = -1
    A_ub[m:, :3] = -N2  # For N2: -P2 · n2 + epsilon <= 0
    A_ub[m:, 3] = -1

    b_ub = np.zeros(m + n)

    A_eq = np.array([[0, 0, 0, 1]])  # epsilon <= 0
    b_eq = np.array([0])

    # Solve the linear programming problem
    result = linear_program_solver(c, A_ub, b_ub, A_eq, b_eq)

    if result is not None:
        P2 = result[:3]
        epsilon = result[3]
        if scalar_norm(P2) > 1e-6 and epsilon < 0:  # Check if the solution is valid
            return P2 / scalar_norm(P2)  # Normalize P2

    return None


# Usage example:
# gm1 = GaussMap(surface1)
# gm2 = GaussMap(surface2)
# P1, P2 = separate_gauss_maps(gm1, gm2)
# if P1 is not None and P2 is not None:
#     print("Gauss maps can be separated")
#     print("P1:", P1)
#     print("P2:", P2)
# else:
#     print("Gauss maps cannot be separated")


from mmcore.numeric.aabb import aabb
from mmcore.geom.bvh import BoundingBox, Object3D, build_bvh, intersect_bvh_objects


class DebugTree:

    def __init__(self, data=None, layer=0):
        self.layer = layer
        self.data = data
        self.chidren = []

    def subd(self, count):
        for i in range(count):
            self.chidren.append(DebugTree(layer=self.layer + 1))
        return self.chidren


def _detect_intersections_deep(g1, g2, chs:dict,tol=0.1, dbg: DebugTree = None):
    """
    Подпрограмма процедуры detect_intersections. Принимает карты гаусса патча безье и выполняет рекурсивное подразбиение.
    Существует три варианта завершения:
    1. Патчи являются разделимыми (вернет пустой список)
    2. Патчи имеют одно тривиальное пересечение (граница одного патча явно пересекается с другим)
    3. Патчи пересекаются в одной точке. (В этом случае при релаксации будет найдено конкретное положение этой точки)

    :param g1:
    :param g2:
    :param tol:
    :param dbg:
    :return:
    """
    bb1, bb2 = BoundingBox(*np.array(g1.surface.bbox())), BoundingBox(
        *np.array(g2.surface.bbox())
    )
    dddd = [False, False, False, False, False]
    dbg.data = (g1, g2, dddd)

    if not bb1.intersect_disjoint(bb2):
        # ББокы не пересекаются
        dddd[0] = True

        return []

    if scalar_norm(bb1.max_point - bb1.min_point) < tol:
        dddd[1] = True
        # Бокс стал пренебрежительно маленьким, мы в сингулярной точке.
        return [(g1.surface, g2.surface)]
    if min(bb1.intersection(bb2).dims) < tol:
        # Бокс не маленький, но очень плоский. объекты не пересекаются
        dddd[2] = True

        return []
    #if is_flat(g1.surface.evaluate_v2, 0.,1.,0.,1.) and is_flat(g2.surface.evaluate_v2, 0.,1.,0.,1.):
    #    print('f')
    #    return [g1.surface, g2.surface]
    if id(g1.surface) not in chs:
        chs[id(g1.surface)] =  ConvexHull(g1.surface.control_points_flat)
    if id(g2.surface) not in chs:
        chs[id(g2.surface)] =  ConvexHull(g2.surface.control_points_flat)
    h1, h2 = chs[id(g1.surface)], chs[id(g2.surface)]
    if not gjk(h1.points[h1.vertices], h2.points[h2.vertices], 1e-8, 25):
        # Поверхности не пересекаются
        dddd[3] = True
        return []

    bb11, bb21 = BoundingBox(*np.array(aabb(g1.bounds()))), BoundingBox(
        *np.array(aabb(g2.bounds()))
    )
    if not bb11.intersect(bb21):
        # Поверхности вероятнее всего пересекаются и не содержать петель
        dddd[3] = True
        return [(g1.surface, g2.surface)]

    intersections = []
    #ss = g1.intersects(g2)
    n1, n2 = separate_gauss_maps(g1,g2)
    if (n1 is not None) and (n2 is not None):

        return []
    #if not ss:
    #    # Поверхности вероятнее всего пересекаются и не содержать петель (для тех кто провалил прошлый тест)

    #    dddd[4] = True
    #    return [(g1.surface, g2.surface)]
    # Все тесты провалены, новый этап подразбиения
    g11 = g1.subdivide()
    g12 = g2.subdivide()

    dbg1 = dbg.subd(16)
    ii = 0
    for gg in g11:
        for gh in g12:
            res = _detect_intersections_deep(gg, gh, chs,tol=tol, dbg= dbg1[ii])
            ii += 1

            intersections.extend(res)
    return intersections


class NURBSObject(Object3D):
    def __init__(self, surface: NURBSSurface):
        self.surface = surface
        super().__init__(BoundingBox(*np.array(self.surface.bbox())))


def detect_intersections(surf1, surf2, debug_tree: DebugTree) -> list[tuple[NURBSSurface, NURBSSurface]]:
    """
    Detects intersections between two NURBS surfaces by using a combination of surface decomposition into Bezier patches,
    bounding volume hierarchy (BVH) traversal, convex hull checks, and Gauss map analysis. The function efficiently finds
    intersecting subpatches of the surfaces.

    Algorithm Overview:
        1. Decomposes the input NURBS surfaces into smaller Bezier patches using a well-known algorithm.
        2. Constructs a BVH (Bounding Volume Hierarchy) for both decomposed surfaces to enable efficient pairwise testing.
        3. For each pair of potentially intersecting Bezier patches (as determined by BVH intersection tests), checks if
           their convex hulls intersect using the GJK (Gilbert-Johnson-Keerthi) algorithm.
        4. If convex hulls intersect, builds Gauss maps for further testing, including potential recursive subdivision
           for patches that cannot be easily separated.
        5. Records the state of the algorithm at each recursion step using the provided `debug_tree`.

    Bezier Patch Decomposition:
        The initial decomposition of the NURBS surface is not merely into smaller patches but specifically into **Bezier patches**.
        This step is critical because the algorithm for constructing a Gauss map requires the surface to be expressed in the
        monomial basis. NURBS surfaces, by default, are expressed in the non-monomial B-spline basis, but Bezier surfaces are
        represented in the Bernstein basis, which can be decomposed into the monomial form. Therefore, decomposing the NURBS surface
        into Bezier patches is a necessary precondition before constructing the Gauss map and performing the intersection tests.

    :param surf1: The first NURBS surface.
    :type surf1: NURBSSurface
    :param surf2: The second NURBS surface.
    :type surf2: NURBSSurface
    :param debug_tree: A debugging tree that stores the state of the algorithm at every recursive step. This is used for
        tracing and debugging the intersection detection process. It is mainly for development purposes and may be removed
        in the final implementation.
    :type debug_tree: DebugTree

    :returns: A list of tuples, where each tuple contains two NURBS surfaces (subpatches) that intersect.
    :rtype: list[tuple[NURBSSurface, NURBSSurface]]

    Detailed Workflow:
        1. **Bezier Patch Decomposition:**
           The input NURBS surfaces `surf1` and `surf2` are decomposed into **Bezier patches** using a well-known
           decomposition algorithm. This decomposition is necessary because the Gauss map construction relies on the
           Bernstein basis, which is a property of Bezier surfaces. The decomposition ensures that the patches can be
           represented in monomial form for further geometric processing.

        2. **Building BVH:**
           A BVH is built for each set of Bezier patches using `build_bvh()`. This hierarchical structure allows for efficient
           pruning of patch pairs that are unlikely to intersect based on their bounding volumes.

        3. **BVH Intersection:**
           The BVH structures for the two surfaces are traversed using `intersect_bvh_objects()`, which efficiently identifies
           pairs of Bezier patches that have overlapping bounding volumes.

        4. **Convex Hull Check (GJK Algorithm):**
           For each pair of Bezier patches returned by the BVH intersection test, the convex hulls of the control points are
           computed. The GJK algorithm is then used to check whether the convex hulls intersect. If the convex hulls do not
           intersect, the patches are discarded.

        5. **Gauss Map and Deep Intersection Check:**
           If the convex hulls intersect, Gauss maps are constructed for both patches using `GaussMap.from_surf()`. These maps
           help in further checking the intersection geometry. If the Gauss maps cannot be separated, the patches are considered
           to intersect. In cases where additional refinement is necessary, the algorithm calls `_detect_intersections_deep()`
           to perform recursive subdivision and deeper checks on the patches.

    Example Usage:

    .. code-block:: python

        surface1 = NURBSSurface(...)  # Create first NURBS surface
        surface2 = NURBSSurface(...)  # Create second NURBS surface
        debug_tree = DebugTree()      # Initialize the debug tree for tracing

        # Detect intersections between the two surfaces
        intersections = detect_intersections(surface1, surface2, debug_tree)

        # Process and output the results
        for surf_pair in intersections:
            surf1, surf2 = surf_pair
            print(f"Intersecting surfaces: {surf1}, {surf2}")

    Performance Considerations:
        - The BVH structure significantly reduces the number of patch pairs that need to be checked, improving performance
          compared to a brute-force approach.
        - The convex hull and Gauss map checks allow for early elimination of non-intersecting patches, further speeding
          up the process.

    Notes:
        - This method is designed for use with NURBS surfaces and leverages Bezier patch decomposition, which is crucial
          for the construction of Gauss maps in the monomial basis.
        - The current implementation strikes a balance between performance and robustness, but further optimization may
          be possible, particularly in reducing the number of recursive subdivisions.

    """
    s1d = decompose_surface(surf1)
    s2d = decompose_surface(surf2)

    subs = debug_tree.subd(len(s1d) * len(s2d))
    index = 0
    intersections = []
    for _ in s1d:
        _.normalize_knots()
    for _ in s2d:
        _.normalize_knots()
    tree1 = build_bvh([NURBSObject(s) for s in s1d])
    tree2 = build_bvh([NURBSObject(s) for s in s2d])
    gauss_maps=dict()
    for obj1, obj2 in intersect_bvh_objects(tree1, tree2):

        f = obj1.object.surface
        s = obj2.object.surface
        dddd = [False, False, False]
        subs[index].data = (f, s, dddd)

        h1, h2 = ConvexHull(f.control_points_flat), ConvexHull(
            s.control_points_flat
        )
        chs=dict()

        if gjk(h1.points[h1.vertices], h2.points[h2.vertices], 1e-5, 25):
            # Convex Hulls пересекаются
            # Строим карты гаусса для дальнейших проверок
            if id(f) not in gauss_maps:
                gauss_maps[id(f)]=GaussMap.from_surf(f)
            if id(s) not in gauss_maps:
                gauss_maps[id(s)]=GaussMap.from_surf(s)
            ss, ff =gauss_maps[id(f)], gauss_maps[id(s)]
            dddd[1] = True

            #p1, p2 = separate_gauss_maps(ff, ss)

            #if (p1 is None) or (p2 is None):
                # Карты не могут быть разделены, запускаем глубокую проверку для данных патчей
            dddd[2] = True
            sbb = subs[index].subd(1)
            intersections.extend(_detect_intersections_deep(ss, ff, chs,0.1, sbb[0]))

        index += 1
    return intersections


if __name__ == "__main__":
    from mmcore._test_data import ssx as td

    S1, S2 = td[2]
    dtr = DebugTree()
    import time
    s=time.time()
    res = detect_intersections(S1, S2, dtr)
    print(time.time()-s)
    fff = []
    for i, j in res:
        ip = np.array(i.control_points)
        jp = np.array(j.control_points)

        if np.any(np.isnan(ip.flatten())) or np.any(np.isnan(jp.flatten())):
            import warnings

            warnings.warn("NAN")
        else:
            fff.append((ip.tolist(), jp.tolist()))

    with open('../../tests/norm2.txt', 'w') as f:
        print(fff, file=f)


    def get_first_layer_dbg(dbg: DebugTree):
        cnds = []
        for ch in dbg.chidren:
            if ch.data:

                if all(ch.data[-1]):
                    cnds.append(
                        [np.array(ch.data[0].control_points).tolist(), np.array(ch.data[1].control_points).tolist()])

        return cnds
