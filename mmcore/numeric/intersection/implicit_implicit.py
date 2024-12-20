import numpy as np

from scipy.spatial import KDTree

from mmcore.geom.implicit.implicit import Intersection3D
from mmcore.numeric.vectors import scalar_norm, scalar_unit, scalar_cross, scalar_dot
from mmcore.numeric.algorithms.implicit_point import intersection_curve_point
from mmcore.numeric.aabb import aabb
from mmcore.numeric.marching import (
    marching_intersection_curve_points,

)
from mmcore.geom.implicit import Implicit3D

DEFAULT_STEP = 0.2


def mgrid3d(bounds, x_count, y_count, z_count):
    # Создаем линейные пространства
    (minx, miny, minz), (maxx, maxy, maxz) = bounds
    x = np.linspace(minx, maxx, x_count)
    y = np.linspace(miny, maxy, y_count)
    z = np.linspace(minz, maxz, z_count)
    # Создаем 3D сетку точек
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    # Объединяем координаты в один массив
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    return points


class ImplicitIntersectionCurve(Implicit3D):
    def __init__(self, surf1: Implicit3D, surf2: Implicit3D, tol=1e-6):
        super().__init__()
        self.surf1 = surf1
        self.surf2 = surf2
        self.tol = tol
        self._intersection = Intersection3D(surf1, surf2)

    def build_tree(self, depth=3):
        self._intersection.build_tree(depth=depth)
        self._tree = self._intersection.tree

    def bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        return self._intersection.bounds()

    def implicit(self, pt):
        return scalar_norm(pt - self.closest_point(pt))

    def tangent(self, pt):
        cpt = self.closest_point(pt)
        return scalar_unit(np.cross(self.surf1.gradient(cpt), self.surf2.gradient(cpt)))

    def gradient(self, xyz):
        return xyz - self.closest_point(xyz)

    def plane(self, pt):
        cpt = self.closest_point(pt)
        pln = np.zeros((4, 3))
        pln[0] = cpt
        n = scalar_unit(pt - cpt)
        pln[2] = n
        n1, n2 = self.surf1.gradient(cpt), self.surf2.gradient(cpt)
        #n3=(n1+ n2)/2
        tang = scalar_unit(scalar_cross(n1, n2))
        pln[1] = tang
        self.surf1.gradient(cpt)
        pln[2] = scalar_unit(n1 - tang * scalar_dot(n1, tang))
        pln[3] = scalar_cross(tang, pln[2])
        return pln

    def closest_point(self, v):
        return intersection_curve_point(self.surf1.implicit, self.surf2.implicit, v, self.surf1.gradient,
                                        self.surf2.gradient, tol=self.tol)

    def __iter__(self):
        return ImplicitIntersectionCurveIterator(self, step=DEFAULT_STEP, workers=-1)


class ImplicitIntersectionCurveIterator:
    """
            Find all intersection curves between two implicit bodies.

            Note
            ----
            When tracing the intersection curves of two implicit pipes (a cylinder with a thick wall) it shows a speed of ~ 18.0 sms,
            (on my machine this is about 1.5-2.5 times slower than Rhinoceros8, which of course can be considered a victory,
            since we achieve this speed in python at runtime).
            If we just trace in a loop from the desired points it will be ~7.5 ms, which is roughly identical to Rhinoceros8.

            KDtree adds about 1 millisecond in the test case.
            But the initial tree construction seems to cost more than 10.0 ms., and it will be much worse on large intersections.
            I tried uniform sampling, but it makes everything much slower, so I'll probably have to give up on that idea.


            """

    def __init__(self, crv: ImplicitIntersectionCurve, step=0.2, workers=-1, debug=None, clear_debug=True, **kwargs):
        self.crv = crv

        self.step = step
        self._kws = kwargs
        self._kdtree = None
        self.pop_queue = set()
        self.workers = workers
        self.traces = []
        self.points = self.build_points(debug=debug, clear=clear_debug)

        self.initial_point = self.rebuild_tree()

    def build_points(self, debug=None, clear=True):
        #TODO сделать более адекватное и простое семплирование точек, можно просто грид по bounds.
        # Построение дерева кажется пока пустой тратой времени особенно для деталей с тонкими стенками и множеством пересечений

        arr = []
        if debug is not None:

            for pt in np.average(self.crv.tree.border, axis=1):
                dbg = []
                success, p = intersection_curve_point(surf1=self.crv.surf1.implicit,
                                                      surf2=self.crv.surf2.implicit,
                                                      q0=pt,
                                                      grad1=self.crv.surf1.gradient, grad2=self.crv.surf2.gradient,
                                                      no_err=True,
                                                      max_iter=8,
                                                      tol=self.crv.tol)
                if success:
                    arr.append(p)
                    if clear:
                        debug[:] = dbg
                    else:
                        debug.append(dbg)
        else:
            for pt in np.average(self.crv.tree.border, axis=1):
                success, p = intersection_curve_point(self.crv.surf1.implicit,
                                                      self.crv.surf2.implicit,
                                                      pt,
                                                      self.crv.surf1.gradient, self.crv.surf2.gradient,
                                                      no_err=True,
                                                      max_iter=8,
                                                      tol=self.crv.tol)
                if success:
                    arr.append(p)

        return np.array(arr)

    def trace_curve_point(self, point, step):
        res = self._kdtree.query_ball_point(point, step, workers=self.workers)
        self.pop_queue.update({*res})

    def rebuild_tree(self):

        ixs = np.array(tuple(self.pop_queue), dtype=int)
        self.points = np.delete(self.points, ixs, axis=0)
        self.initial_point = None

        if len(self.points) > 0:

            self.initial_point = self.points[0, :]
            self.points = self.points[1:, :]
            if len(self.points) > 0:
                self._kdtree = KDTree(np.copy(self.points))
            else:
                self._kdtree = None

        else:
            self._kdtree = None
        self.pop_queue.clear()
        return self.initial_point

    def __next__(self):

        if self.initial_point is not None:

            res = marching_intersection_curve_points(self.crv.surf1.implicit, self.crv.surf2.implicit,
                                                     start_point=self.initial_point,
                                                     grad_f1=self.crv.surf1.gradient,
                                                     grad_f2=self.crv.surf2.gradient,
                                                     step=self.step,
                                                     point_callback=lambda pt: self.trace_curve_point(pt,
                                                                                                      step=self.step) if self._kdtree is not None else None,
                                                     **self._kws).tolist()
            self.initial_point = self.rebuild_tree()

            return res

        else:
            raise StopIteration

    def __iter__(self):
        return self


def intersection_curve(surf1, surf2):
    return ImplicitIntersectionCurve(surf1, surf2)


def iterate_curves(curve: ImplicitIntersectionCurve, step=0.1, workers=-1, return_nurbs=False, debug=None, **kwargs):
    if return_nurbs:
        return iterate_curves_as_nurbs(curve, step, workers=workers, **kwargs)
    return ImplicitIntersectionCurveIterator(curve, step=step, workers=workers, debug=debug, **kwargs)


from mmcore.geom.curves.knot import interpolate_curve


def iterate_curves_as_nurbs(curve: ImplicitIntersectionCurve, step=0.1, degree=3, workers=-1, **kwargs):
    for pts in ImplicitIntersectionCurveIterator(curve, step=step, workers=workers, **kwargs):
        yield interpolate_curve(pts, degree=degree)


def surface_iii(surf1, surf2, tol=1e-6):
    return ImplicitIntersectionCurve(surf1, surf2, tol=tol)


if __name__ == '__main__':
    import numba


    # Поистине огромным преимуществом является то что не сложная пользовательская реализация всего пары implicit и normal с использованием numba,
    # дает практически тот же результат что и примитивы реализованные на cython.
    # Это значит что действительно можно  получать создавать пользовательские объекты, не прибегая к написанию расширений на C,
    # и при этом иметь производительность сравнимую с коммерческим САПР.

    # The really great advantage is that a simple custom implementation of just a pair of implicit and normal using numba,
    # gives pretty much the same result as primitives implemented in cython.
    # This means that it is indeed possible to create custom objects without having to write C extensions,
    # and still have performance comparable to commercial CAD.

    @numba.njit(cache=True)
    def closest_point_on_ray2(ray, point):
        start, b = ray
        a = (point[0] - start[0], point[1] - start[1], point[2] - start[2])
        bn = (b[0] ** 2 + b[1] ** 2 + b[2] ** 2)
        # return start + vector_projection(point - start, direction)
        return (a[0] * b[0] * b[0] / bn + a[1] * b[0] * b[1] / bn + a[2] * b[0] * b[2] / bn + start[0],
                a[0] * b[0] * b[1] / bn + a[1] * b[1] * b[1] / bn + a[2] * b[1] * b[2] / bn + start[1],
                a[0] * b[0] * b[2] / bn + a[1] * b[1] * b[2] / bn + a[2] * b[2] * b[2] / bn + start[2])


    @numba.jit(cache=True)
    def cyl_normal(origin, axis, x, y, z):
        px, py, pz = closest_point_on_ray2((origin, axis), (x, y, z))
        n = np.array([x - px, y - py, z - pz])

        N = n / np.linalg.norm(n)

        if np.allclose(N, 0.0):
            return np.zeros(3, dtype=float)
        else:
            return n / N


    @numba.jit(cache=True)
    def cyl_implicit(origin, axis, r, x, y, z) -> float:
        px, py, pz = closest_point_on_ray2((origin, axis), (x, y, z))
        n = np.array([x - px, y - py, z - pz])
        return np.linalg.norm(n) - r


    @numba.jit(cache=True)
    def tub_implicit(origin, axis, r, thickness, x, y, z) -> float:
        ii = cyl_implicit(origin, axis, r, x, y, z)
        return abs(ii) - thickness / 2


    @numba.jit(cache=True)
    def tub_normal(origin, axis, r, thickness, x, y, z) -> float:

        res = np.zeros(3, dtype=float)
        res[0] = (tub_implicit(origin, axis, r, thickness, x + 0.001, y, z) - tub_implicit(origin, axis, r, thickness,
                                                                                           x - 0.001, y, z)) / 2 / 0.001
        res[1] = (tub_implicit(origin, axis, r, thickness, x, y + 0.001, z) - tub_implicit(origin, axis, r, thickness,
                                                                                           x, y - 0.001, z)) / 2 / 0.001
        res[2] = (tub_implicit(origin, axis, r, thickness, x, y, z + 0.001) - tub_implicit(origin, axis, r, thickness,
                                                                                           x, y, z - 0.001)) / 2 / 0.001

        return res / np.linalg.norm(res)


    class Cylinder(Implicit3D):
        def __init__(
                self, origin=np.array([0.0, 0.0, 0.0]), r=1, axis=np.array([0.0, 0.0, 1.0])
        ):
            super().__init__(autodiff=False)
            self.axis = np.array(axis)
            self.start = self.origin = np.array(origin)
            self.r = r
            self.end = np.array(self.origin + self.axis)

        def implicit(self, v) -> float:
            return cyl_implicit(self.origin, self.axis, self.r, v[0], v[1], v[2])

        def gradient(self, v) -> float:
            return cyl_normal(self.origin, self.axis, v[0], v[1], v[2])

        def bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
            return aabb(np.array(
                [self.start + self.r ** 2, self.start - self.r ** 2, self.end + self.r ** 2, self.end - self.r ** 2]))


    class Tube(Cylinder):
        def __init__(self, thickness, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.thickness = thickness

        def implicit(self, pt):
            return tub_implicit(self.origin, self.axis, self.r, self.thickness, pt[0], pt[1], pt[2])

        def gradient(self, v):
            return tub_normal(self.origin, self.axis, self.r, self.thickness, v[0], v[1], v[2])


    x, y, v, u, z = [[[12.359112840551504, -7.5948049557495425, 0.0], [2.656625109045951, 1.2155741170561933, 0.0]],
                     [[7.14384241216015, -6.934735074711716, -0.1073366304415263],
                      [7.0788761013028365, 10.016931402130641, 0.8727530304189204]],
                     [[8.072688942425103, -2.3061831591019826, 0.2615779273274319],
                      [7.173685617288537, -3.4427234423361512, 0.4324928834164773],
                      [7.683972288682133, -2.74630545102506, 0.07413871667321925],
                      [7.088944240699163, -4.61458155002528, -0.22460509818398067],
                      [7.304629277158477, -3.9462033818505433, 0.8955725109783643],
                      [7.304629277158477, -3.3362864951018985, 0.8955725109783643],
                      [7.304629277158477, -2.477065729786164, 0.7989970582016114],
                      [7.304629277158477, -2.0988672326949933, 0.7989970582016114]], 0.72648, 1.0]

    aa = np.array(x)
    bb = np.array(y)

    cl2 = Cylinder(bb[0], u, bb[1] - bb[0])
    cl1 = Cylinder(aa[0], z, aa[1] - aa[0])
    t1 = Tube(0.2, aa[0], z, aa[1] - aa[0])
    t2 = Tube(0.2, bb[0], u, bb[1] - bb[0])
    t1.implicit(np.array((1., 1., 1)))  # compile
    t1.gradient(np.array((1., 1., 1)))  #compile
    vv = np.array(v)

    import time

    s = time.perf_counter_ns()
    try:
        res = []
        for i in range(len(vv)):
            res.append(
                marching_intersection_curve_points(
                    t1.implicit,
                    t2.implicit,
                    t1.gradient,
                    t2.gradient,
                    vv[i],
                    max_points=200,
                    step=0.2,
                    tol=1e-5,
                ).tolist()
            )


    except ValueError as err:
        print(err)
    print("numba primitives speed:", (time.perf_counter_ns() - s) * 1e-6, 'ms.')
    crv = ImplicitIntersectionCurve(t1, t2)
    crv.build_tree()
    s = time.perf_counter_ns()
    res2 = []

    for item in iterate_curves(crv):
        res2.append(item)
    print("numba primitives speed (full):", (time.perf_counter_ns() - s) * 1e-6, 'ms.')

    print(len(res2))
    from mmcore.geom.primitives import Tube

    t11 = Tube(aa[0], aa[1], z, 0.2)
    t21 = Tube(bb[0], bb[1], u, 0.2)

    s = time.perf_counter_ns()
    res1 = []
    for item in iterate_curves(crv):
        res2.append(item)

    print("mmcore builtin primitives speed:", (time.perf_counter_ns() - s) * 1e-6, 'ms.')
