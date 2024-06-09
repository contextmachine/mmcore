import warnings

import numpy as np

from scipy.spatial import KDTree

from mmcore.geom.implicit.implicit import Implicit, Intersection3D
from mmcore.geom.implicit.marching import intersection_curve_point

from mmcore.geom.vec.vec_speedups import scalar_norm, scalar_unit


from mmcore.numeric.aabb import aabb

from mmcore.geom.implicit.marching import (
    marching_intersection_curve_points,

)
from mmcore.geom.implicit import Implicit3D


def mgrid3d(bounds, x_count,y_count,z_count):
    # Создаем линейные пространства
    (minx, miny, minz), (maxx, maxy, maxz) =bounds
    x = np.linspace(minx,maxx, x_count)
    y = np.linspace(miny,maxy, y_count)
    z = np.linspace(minz,maxz,z_count)
    # Создаем 3D сетку точек
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    # Объединяем координаты в один массив
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    return points


class ImplicitIntersectionCurve(Implicit):
    def __init__(self, surf1: Implicit3D, surf2: Implicit3D, tol=1e-6):
        super().__init__()
        self.surf1 = surf1
        self.surf2 = surf2
        self.tol = tol
        self._intersection = Intersection3D(surf1, surf2)

    def build_tree(self, depth=3):
        self._intersection.build_tree(depth=depth)
        self._tree = self._intersection.tree

    def sample(self, step=None,x_cnt=15, y_cnt=15, z_cnt=15):

        (minx, miny, minz), (maxx, maxy, maxz) = self.bounds()
        print(self.bounds)
        if step is not None:
            x_cnt, y_cnt, z_cnt = np.array([np.ceil((maxx - minx) / step), np.ceil((maxy - miny) / step), np.ceil(
            (maxz - minz) / step)], dtype=int)
        return mgrid3d(((minx, miny, minz), (maxx, maxy, maxz)), x_cnt, y_cnt, z_cnt)

    def bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        return self._intersection.bounds()

    def implicit(self, v):
        return scalar_norm(self._normal_not_unit(v))

    def _normal_not_unit(self, point):
        return point - self.closest_point(point)

    def normal(self, point):
        return scalar_unit(self._normal_not_unit(point))

    def closest_point(self, v):
        return intersection_curve_point(self.surf1.implicit, self.surf2.implicit, v, self.surf1.normal, self.surf2.normal, tol=self.tol)


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

    def __init__(self, crv: ImplicitIntersectionCurve, step=0.1, workers=-1, **kwargs):
        self.crv = crv

        self.points = self.build_points()
        self._kdtree = None
        self.pop_queue = set()
        self.workers = workers
        self.traces = []
        self.initial_point = self.rebuild_tree()
        self.step = step
        self._kws = kwargs

    def build_points(self, method=0):
        #TODO сделать более адекватное и простое семплирование точек, можно просто грид по bounds.
        # Построение дерева кажется пока пустой тратой времени особенно для деталей с тонкими стенками и множеством пересечений
        if hasattr(self.crv, '_tree'):
            return np.array([self.crv.closest_point(i) for i in
                         np.average(self.crv.tree.border + self.crv.tree.full, axis=1)])
        else:
            if method == 0:
                warnings.warn("Method 0 cannot be used because one of the bodies does not have the `tree` attribute. Using method 1")
            l=[]

            for i in self.crv.sample():

                d=abs(self.crv.implicit(i))
                if d<=0.2:
                    print(d,i)
                    l.append(self.crv.closest_point(i))


            return np.array(l)
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
                                                     grad_f1=self.crv.surf1.normal,
                                                     grad_f2=self.crv.surf2.normal,
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


def iterate_curves(curve: ImplicitIntersectionCurve, step=0.1, workers=-1, **kwargs):
    return ImplicitIntersectionCurveIterator(curve, step=step, workers=workers, **kwargs)


from mmcore.geom.curves.knot import interpolate_curve


def iterate_curves_as_nurbs(curve: ImplicitIntersectionCurve, step=0.1, degree=3, workers=-1, **kwargs):
    for pts in ImplicitIntersectionCurveIterator(curve, step=step, workers=workers, **kwargs):
        yield interpolate_curve(pts, degree=degree)


if __name__ == '__main__':
    import numba


    @numba.njit(cache=True)
    def py_vp(a, b):
        res = np.empty((3,))
        bn = (b[0] ** 2 + b[1] ** 2 + b[2] ** 2)

        res[0] = a[0] * b[0] * b[0] / bn + a[1] * b[0] * b[1] / bn + a[2] * b[0] * b[2] / bn
        res[1] = a[0] * b[0] * b[1] / bn + a[1] * b[1] * b[1] / bn + a[2] * b[1] * b[2] / bn
        res[2] = a[0] * b[0] * b[2] / bn + a[1] * b[1] * b[2] / bn + a[2] * b[2] * b[2] / bn
        return res
    @numba.njit(cache=True)
    def closest_point_on_ray2(ray, point):
        start, b = ray
        a=(point[0] - start[0], point[1] - start[1], point[2] - start[2])

        bn = (b[0] ** 2 + b[1] ** 2 + b[2] ** 2)



        # return start + vector_projection(point - start, direction)
        return  (a[0] * b[0] * b[0] / bn + a[1] * b[0] * b[1] / bn + a[2] * b[0] * b[2] / bn+start[0],
         a[0] * b[0] * b[1] / bn + a[1] * b[1] * b[1] / bn + a[2] * b[1] * b[2] / bn+start[1],
         a[0] * b[0] * b[2] / bn + a[1] * b[1] * b[2] / bn + a[2] * b[2] * b[2] / bn+start[2])

    @numba.jit(cache=True)
    def cyl_normal(origin, axis, x,y,z):
        px,py,pz = closest_point_on_ray2((origin, axis), (x,y,z))
        n = np.array([x-px ,y-py,z-pz])

        N = n / np.linalg.norm( n )

        if np.allclose(N, 0.0):
            return np.zeros(3, dtype=float)
        else:
            return n / N


    @numba.jit(cache=True)
    def cyl_implicit(origin, axis, r,x,y,z) -> float:
        px,py,pz = closest_point_on_ray2((origin, axis), (x,y,z))
        n = np.array([x - px, y - py, z - pz])
        return np.linalg.norm(n) - r
    @numba.jit(cache=True)
    def tub_implicit(origin, axis, r, thickness,x,y,z) -> float:
        ii = cyl_implicit(origin, axis, r, x,y,z)
        return abs(ii) - thickness / 2
    @numba.jit(cache=True)
    def tub_normal(origin, axis, r, thickness, x,y,z) -> float:

        res = np.zeros(3, dtype=float)
        res[0]=( tub_implicit(origin, axis, r, thickness,x+0.001,y,z)-        tub_implicit(origin, axis, r, thickness, x-0.001,y,z))/2/0.001
        res[1] =(tub_implicit(origin, axis, r, thickness,x,y+0.001,z)-        tub_implicit(origin, axis, r, thickness, x,y-0.001,z))/2/0.001
        res[2] =(tub_implicit(origin, axis, r, thickness,x,y,z+0.001)-        tub_implicit(origin, axis, r, thickness, x,y,z-0.001))/2/0.001


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

            return cyl_implicit(self.origin,self.axis,self.r,v[0],v[1],v[2])

        def normal(self, v) -> float:
            return cyl_normal(self.origin, self.axis,v[0],v[1],v[2])
        def bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:

            return aabb(np.array(
                [self.start + self.r ** 2, self.start - self.r ** 2, self.end + self.r ** 2, self.end - self.r ** 2]))


    class Tube(Cylinder):
        def __init__(self, thickness, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.thickness = thickness


        def implicit(self, pt):


            return  tub_implicit(self.origin, self.axis, self.r, self.thickness,pt[0],pt[1],pt[2])
        def normal(self, v):
            return tub_normal(self.origin, self.axis, self.r, self.thickness,v[0],v[1],v[2])

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
    t1.normal(np.array((1.,1.,1))) #compile
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
                    t1.normal,
                    t2.normal,
                    vv[i],
                    max_points=200,
                    step=0.2,
                    tol=1e-5,
                ).tolist()
            )


    except ValueError as err:
        print(err)

    #print("numba primitives speed (full):", (time.perf_counter_ns() - s)*1e-6,'ms.')


    crv = ImplicitIntersectionCurve(t1, t2)
    crv.build_tree()
    s = time.perf_counter_ns()
    res2 = []
    for item in iterate_curves(crv):
        res2.append(item)
    #trace = ImplicitIntersectionCurveIterator(crv)

    print("numba primitives speed (full):", (time.perf_counter_ns() - s) * 1e-6, 'ms.')

    print(len(res2))
    from mmcore.geom.implicit._implicit import CylinderPipe

    t11 = CylinderPipe(aa[0],aa[1] , z,0.2 )
    t21 = CylinderPipe(bb[0],bb[1] , u,0.2 )

    s = time.perf_counter_ns()
    res1 = []
    for item in iterate_curves(crv):
        res2.append(item)



    print("mmcore builtin primitives speed:",(time.perf_counter_ns() - s)*1e-6,'ms.')