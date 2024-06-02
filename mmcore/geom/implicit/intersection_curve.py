
import numpy as np
from scipy.spatial import KDTree

from mmcore.geom.implicit.implicit import Implicit, Intersection3D
from mmcore.geom.implicit.marching import intersection_curve_point

from mmcore.geom.vec.vec_speedups import scalar_norm, scalar_unit
from mmcore.numeric.closest_point import closest_point_on_line, closest_point_on_ray

from mmcore.numeric.aabb import aabb

from mmcore.geom.implicit.marching import (
    marching_intersection_curve_points,

)
from mmcore.geom.implicit import Implicit3D


class IntersectionImplicitCurve(Implicit):
    def __init__(self, surf1: Implicit3D, surf2: Implicit3D, tol=1e-6):
        super().__init__(autodiff=False)
        self.surf1 = surf1
        self.surf2 = surf2
        self.tol = tol
        self._intersection = Intersection3D(surf1, surf2)

    def build_tree(self, depth=3):
        self._intersection.build_tree(depth=depth)
        self._tree = self._intersection.tree

    def bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        return self._intersection.bounds()

    def implicit(self, v):
        return scalar_norm(self._normal_not_unit(v))

    def _normal_not_unit(self, point):
        return point - self.closest_point(point)

    def normal(self, point):
        return scalar_unit(self._normal_not_unit(point))

    def closest_point(self, v):
        return intersection_curve_point(self.surf1, self.surf2, v, self.surf1.normal, self.surf2.normal, tol=self.tol)




class TraceIntersectionImplicitCurve:
    def __init__(self, crv: IntersectionImplicitCurve, workers=-1):
        self.crv = crv

        self.points = self.build_points()
        self._kdtree = None
        self.pop_queue = []
        self.workers = workers
        self.traces = []
        self.initial_point = self.rebuild_tree()

    def build_points(self):
        return np.array([self.crv.closest_point(i) for i in np.average(self.crv.tree.border, axis=1)])

    def trace_curve_point(self, point, step):

        self.add_to_pop(point, step)

    def add_to_pop(self, point, radius):
        res = self._kdtree.query_ball_point(point, radius, workers=self.workers)

        sorted_ixs = np.sort(res)
        self.pop_queue.extend(sorted_ixs)


    def rebuild_tree(self):

        ixs = np.array(np.unique(self.pop_queue), dtype=int)
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

    def _build_one(self, step=0.1, **kwargs):

        if self.initial_point is not None:
            self.traces.append(marching_intersection_curve_points(self.crv.surf1.implicit, self.crv.surf2.implicit,
                                                                  start_point=self.initial_point,
                                                                  grad_f1=self.crv.surf1.normal,
                                                                  grad_f2=self.crv.surf2.normal,
                                                                  step=step,
                                                                  point_callback=lambda pt: self.trace_curve_point(pt,
                                                                                                                   step=step) if self._kdtree is not None else None,
                                                                  **kwargs).tolist())
            self.initial_point = self.rebuild_tree()

    def build(self, step=0.1, **kwargs):
        """
        Find all intersection curves between two implicit bodies.

        Note
        ----
        When tracing the intersection curves of two implicit pipes (a cylinder with a thick wall) it shows a speed of ~0.150 sec,
        (on my machine this is about 10 times slower than Rhinoceros8, which of course can be considered a victory,
        since we achieve this speed in python at runtime with classes defined on the fly).
        If we just trace in a loop from the desired points it will be ~0.130 sec.
        This means that replacing KDTree with a more specialized data structure will give ~ -0.020 to speed at best.
        This is not insignificant, but most of the time is spent on marching, not on querying points and rebuilding the tree.


        :param step:
        :param kwargs:
        :return:
        """
        while self.initial_point is not None:
            self._build_one(step=step, **kwargs)
        return self.traces
if __name__ == '__main__':


    class Cylinder(Implicit3D):
        def __init__(
                self, origin=np.array([0.0, 0.0, 0.0]), r=1, axis=np.array([0.0, 0.0, 1.0])
        ):
            super().__init__(autodiff=False)
            self.axis = np.array(axis)
            self.start = self.origin = np.array(origin)
            self.r = r
            self.end = np.array(self.origin + self.axis)


        def _normal(self, v):
            pt = closest_point_on_ray((self.origin, self.axis), v)
            n = v - pt
            N = scalar_norm(n)

            if np.allclose(N, 0.0):
                return np.zeros(3, dtype=float)
            else:
                return n / N

        def implicit(self, v) -> float:
            pt = closest_point_on_ray((self.origin, self.axis), v)
            return scalar_norm(pt - v) - self.r

        def bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:

            return aabb(np.array(
                [self.start + self.r ** 2, self.start - self.r ** 2, self.end + self.r ** 2, self.end - self.r ** 2]))


    class Tube(Cylinder):
        def __init__(self, thickness, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.thickness = thickness
            self.normal = self.normal_from_function(self.implicit)

        def implicit(self, pt):
            ii = super().implicit(pt)
            return abs(ii) - self.thickness / 2


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
    vv = np.array(v)
    print(cl2.normal(vv[0]))
    import time

    s = time.time()
    try:
        res = []
        for i in range(len(vv)):
            res.append(
                marching_intersection_curve_points(
                    t1.implicit,
                    t2.implicit,
                    t1.normal,
                    t2.normal,
                    vv[i],
                    max_points=200,
                    step=0.1,
                    tol=1e-5,
                )
            )


    except ValueError as err:
        print(err)
    print(time.time() - s)

    crv = IntersectionImplicitCurve(t1, t2)
    crv.build_tree()
    s = time.time()

    trace = TraceIntersectionImplicitCurve(crv)
    res=trace.build()
    print(time.time() - s)
    print(len(res))