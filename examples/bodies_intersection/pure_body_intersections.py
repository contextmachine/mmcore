
import numpy as np

from mmcore.geom.implicit.marching import marching_intersection_curve_points
from mmcore.geom.implicit.implicit import Implicit3D
from mmcore.geom.implicit.intersection_curve import ImplicitIntersectionCurve, iterate_curves
from mmcore.geom.vec.vec_speedups import scalar_norm
from mmcore.numeric.closest_point import closest_point_on_ray


def min_vec(a, b):
    return np.array([min(a[0], b[0]), min(a[1], b[1]), min(a[2], b[2])])


def max_vec(a, b):
    return np.array((max(a[0], b[0]), max(a[1], b[1]), max(a[2], b[2])))


def cylinder_aabb(pa, pb, ra):
    a = pb - pa
    e = ra * np.sqrt(1.0 - a * a / np.dot(a, a))
    return (min_vec(pa - e, pb - e), max_vec(pa + e, pb + e))


def cone_aabb(pa, pb, ra, rb):
    a = pb - pa
    e = np.sqrt(1.0 - a * a / np.dot(a, a))
    return (min_vec(pa - e * ra, pb - e * rb), max_vec(pa + e * ra, pb + e * rb))


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

        return cylinder_aabb(self.start, self.end, self.r)


class Tube(Cylinder):
    def __init__(self, thickness, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.thickness = thickness
        self.normal = self.normal_from_function(self.implicit)

    def implicit(self, pt):
        ii = super().implicit(pt)
        return abs(ii) - self.thickness / 2

    def bounds(self):
        return cylinder_aabb(self.start,self.end,self.r+self.thickness/2)

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
t1.normal(np.random.random(3))
t2.normal(np.random.random(3))
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

crv = ImplicitIntersectionCurve(t1, t2)
crv.build_tree()
s = time.time()
res = []
for item in iterate_curves(crv):
    res.append(item)
#trace = ImplicitIntersectionCurveIterator(crv)

print(time.time() - s)

print(len(res))
