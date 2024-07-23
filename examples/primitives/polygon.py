import numpy as np
from mmcore.geom.curves.curve import Curve
from mmcore.geom.curves.bspline import NURBSpline
from mmcore.geom.implicit import Implicit2D
from mmcore.numeric.aabb import aabb


class Polygon(Implicit2D, Curve):
    def __init__(self, vertices):
        super().__init__()
        self._vertices = np.array(vertices)[...,:2]
        cpts=np.zeros((len(vertices),3))
        cpts[...,:2]=self._vertices
        min_, max_ = aabb(self._vertices)
        self._bounds = tuple(min_), tuple(max_)
        self._parametric_form = NURBSpline(cpts, degree=1)
    def is_periodic(self):
        return True
    def point_inside(self, pt):
        return super().point_inside(pt)

    def evaluate(self, x):
        return self._parametric_form.evaluate(x)

    def evaluate_multi(self, x):
        return self._parametric_form.evaluate_multi(x)

    def derivative(self, x):
        return self._parametric_form.derivative(x)

    def second_derivative(self, x):
        return self._parametric_form.second_derivative(x)

    def normal(self, x):
        return self._parametric_form.normal(x)

    def tangent(self, x):
        return self._parametric_form.tangent(x)

    def curvature(self, x):
        return self._parametric_form.curvature(x)

    def interval(self):
        return self._parametric_form.interval()

    def plane_at(self, x):
        return self._parametric_form.plane_at(x)

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, v):
        self._vertices = np.array(v)[...,:2]
        min_, max_ = aabb(self._vertices)
        self._bounds = tuple(min_), tuple(max_)
        cpts = np.zeros((len(self._vertices), 3))
        cpts[..., :2] = self._vertices



        self._parametric_form.control_points = cpts

    def bounds(self):
        return self._bounds

    def implicit(self, p):
        d = np.dot(p - self.vertices[0], p - self.vertices[0])
        s = 1.0
        N = len(self.vertices)
        for i in range(N):
            j = (i - 1) % N
            e = self.vertices[j] - self.vertices[i]
            w = p - self.vertices[i]
            b = w - e * max(min(np.dot(w, e) / np.dot(e, e), 1.0), 0.0)
            d = min(d, np.dot(b, b))
            c = np.array(
                [p[1] >= self.vertices[i][1], p[1] < self.vertices[j][1], e[0] * w[1] > e[1] * w[0]], dtype=bool
            )
            if c.all() or not c.any():
                s *= -1.0
        return s * np.sqrt(d)