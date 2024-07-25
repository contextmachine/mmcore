import numpy as np

from mmcore.geom.curves import NURBSpline
from mmcore.geom.implicit import Implicit2D
from mmcore.numeric.vectors import scalar_norm


class Rectangle(Implicit2D):
    def __init__(self, origin, width, height):
        super().__init__()
        self.origin = np.array(origin, dtype=float)
        self._width = width
        self._height = height
        self._b= np.array([self._width / 2.0, self._height / 2.0])
    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, v):
        self._width = v
        self._b[1] = self._width / 2
    @property
    def height(self):
        return self._width

    @height.setter
    def height(self,v):
        self._height=v
        self._b[1]=self._height/2
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
            return (
                (self.origin[0], self.origin[1]),
                (self.origin[0] + self.width, self.origin[1] + self.height))

    def points(self, max_points: int = None, step: float = None, delta: float = 0.001):
        if max_points is not None and step is not None:
            return Implicit2D.points(self, max_points=max_points,step=step,delta=delta)
        elif step is not None:
            return Implicit2D.points(self, step=step, delta=delta)
        return np.array([self.origin,  (self.origin[0]+self.width, self.origin[1]),  (self.origin[0]+self.width, self.origin[1]+self.height)+ (self.origin[0], self.origin[1]+self.height)])
    def implicit(self, p) -> float:
        b = self._b
        # Adjust the point p relative to the origin
        p_adjusted = p - (self.origin + b)

        q = np.abs(p_adjusted) - b
        return min(max(q[0], q[1]), 0.0) + scalar_norm(np.maximum(q, 0.0))

    def to_bspline(self, degree=1, step: float = 0.5):
        pts=self.points()
        return NURBSpline(pts, degree=degree)

    def gradient(self, p):
        # Calculate the half-dimensions of the box
        b = self._b
        # Adjust the point p relative to the origin
        p_adjusted = p - (self.origin + b)

        # Original sdgBox logic
        w = np.abs(p_adjusted) - b
        s = np.array([(-1 if p_adjusted[0] < 0.0 else 1), (-1 if p_adjusted[1] < 0.0 else 1)])
        g = max(w[0], w[1])
        q = np.maximum(w, 0.0)
        l = scalar_norm(q)

        #distance = l if g > 0.0 else g
        grad = s * (q / l if g > 0.0 else (np.array([1, 0]) if w[0] > w[1] else np.array([0, 1])))

        return np.array(grad)

