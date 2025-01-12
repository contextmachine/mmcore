import numpy as np

from mmcore.geom.curves import NURBSpline
from mmcore.geom.implicit import Implicit2D
from examples.primitives.rectangle import Rectangle


class RoundedRectangle(Rectangle):
    def __init__(self, origin, width, height, r=(1.0, 1.0, 1.0, 1.0)):
        super().__init__(origin, width, height)
        self.r = np.array(r, dtype=float)

    def points(self, max_points: int = None, step: float = None, delta: float = 0.001):
        return Implicit2D.points(self, max_points=max_points, step=step, delta=delta)

    def _control_points(self):
        pts = (

            [self.origin[0], self.origin[1] + self.r[0], 0.],
            [self.origin[0], self.origin[1], 0.],
            [self.origin[0] + self.r[0], self.origin[1], 0.],

            [self.origin[0] + self.width - self.r[1], self.origin[1], 0.],

            [self.origin[0] + self.width, self.origin[1], 0.],

            [self.origin[0] + self.width, self.origin[1] + self.r[2], 0.],
            [self.origin[0] + self.width, self.origin[1] + self.height - self.r[3], 0.],

            [self.origin[0] + self.width, self.origin[1] + self.height, 0.],

            [self.origin[0] + self.width - self.r[3], self.origin[1] + self.height, 0.],
            [self.origin[0] + self.r[0], self.origin[1] + self.height, 0.],

            [self.origin[0], self.origin[1] + self.origin[1] + self.height, 0.],

            [self.origin[0], self.origin[1] + self.height - self.r[3], 0.],

            [self.origin[0], self.origin[1] + self.r[0], 0.]

        )

        return np.array(pts)

    # TODO: Make a NURBS curve with line and arc segments
    def to_bspline(self, degree=1, step: float = 0.5):

        return NURBSpline(self._control_points(), degree=2)

    def implicit(self, p) -> float:
        b = self._b
        # Adjust the point p relative to the origin
        p_adjusted = p - (self.origin + b)

        # Original sdRoundedBox logic
        if p_adjusted[0] > 0.0:
            rx, ry = self.r[0], self.r[1]
        else:
            rx, ry = self.r[2], self.r[3]

        if p_adjusted[1] > 0.0:
            rx = rx
        else:
            rx = ry

        q = np.abs(p_adjusted) - b + rx
        return min(max(q[0], q[1]), 0.0) + np.linalg.norm(np.maximum(q, 0.0)) - rx

    def gradient(self, p):
        return Implicit2D.gradient(self, p)
