import numpy as np

from mmcore.geom.implicit import Implicit2D


class RoundedRectangle(Implicit2D):
    def __init__(self, origin, width, height, r=(1.0, 1.0, 1.0, 1.0)):
        super().__init__()
        self.origin = np.array(origin, dtype=float)
        self.width = width
        self.height = height
        self.r = np.array(r, dtype=float)

    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (
            (self.origin[0], self.origin[1]),
            (self.origin[0] + self.width, self.origin[1] + self.height),
        )

    def implicit(self, p) -> float:
        b = np.array([self.width / 2.0, self.height / 2.0])
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
