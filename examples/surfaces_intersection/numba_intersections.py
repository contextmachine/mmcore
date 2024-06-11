import numba
import numpy as np
import time
from mmcore.geom.implicit.implicit import Implicit3D
from mmcore.geom.implicit.intersection_curve import ImplicitIntersectionCurve, iterate_curves


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
    res[1] = (tub_implicit(origin, axis, r, thickness, x, y + 0.001, z) - tub_implicit(origin, axis, r, thickness, x,
                                                                                       y - 0.001, z)) / 2 / 0.001
    res[2] = (tub_implicit(origin, axis, r, thickness, x, y, z + 0.001) - tub_implicit(origin, axis, r, thickness, x, y,
                                                                                       z - 0.001)) / 2 / 0.001

    return res / np.linalg.norm(res)


def cylinder_aabb(pa, pb, ra):
    a = pb - pa
    e = ra * np.sqrt(1.0 - a * a / scalar_dot(a, a))
    return (np.minimum(pa - e, pb - e), np.maximum(pa + e, pb + e))


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

    def _normal(self, v) -> float:
        return cyl_normal(self.origin, self.axis, v[0], v[1], v[2])

    def bounds(self):
        return cylinder_aabb(self.start, self.end, self.r)


class Tube(Cylinder):
    def __init__(self, thickness, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.thickness = thickness

    def implicit(self, pt):
        return tub_implicit(self.origin, self.axis, self.r, self.thickness, pt[0], pt[1], pt[2])

    def _normal(self, v):
        return tub_normal(self.origin, self.axis, self.r, self.thickness, v[0], v[1], v[2])

    def bounds(self):
        return cylinder_aabb(self.start, self.end, self.r + self.thickness / 2)


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

t1 = Tube(0.2, aa[0], z, aa[1] - aa[0])
t2 = Tube(0.2, bb[0], u, bb[1] - bb[0])

t1.normal(np.random.random(3))
t2.normal(np.random.random(3))

crv = ImplicitIntersectionCurve(t1, t2)
crv.build_tree()

s = time.time()
res = []
for item in iterate_curves(crv):
    res.append(item)

print(time.time() - s)

print(len(res))
