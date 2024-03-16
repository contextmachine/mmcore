import numpy as np

from mmcore.geom.bspline import NURBSpline


def nurbs_curve_split(self: NURBSpline, t: float) -> tuple[NURBSpline, NURBSpline]:
    p = self.degree
    knots_to_insert = [t] * p
    new_knots = np.insert(self.knots, np.searchsorted(self.knots, t), knots_to_insert)
    left_points = self.control_points[:np.searchsorted(self.knots, t) + 1]
    left_weights = self.weights[:np.searchsorted(self.knots, t) + 1]
    right_points = self.control_points[np.searchsorted(self.knots, t):]
    right_weights = self.weights[np.searchsorted(self.knots, t):]
    for _ in range(p):
        points = []
        weights = []
        for j in range(1, len(left_points)):
            if self.knots[j + p] - self.knots[j] != 0:
                alpha = (t - self.knots[j]) / (self.knots[j + p] - self.knots[j])
            else:
                alpha = 0  # or any other appropriate value
            points.append(alpha * left_points[j] + (1 - alpha) * left_points[j - 1])
            weights.append(alpha * left_weights[j] + (1 - alpha) * left_weights[j - 1])
        left_points = points
        left_weights = weights
    left_curve = NURBSpline(np.array(left_points), np.array(left_weights), self.degree,
                            new_knots[:len(left_points) + p + 1])
    right_curve = NURBSpline(np.array(right_points), np.array(right_weights), self.degree,
                             new_knots[-len(right_points) - p - 1:])
    return left_curve, right_curve
