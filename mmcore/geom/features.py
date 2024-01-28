from enum import IntEnum

import numpy as np


class PointsOrder(IntEnum):
    COLLINEAR = -1
    CW = 0
    CCW = 1


def points_order(points, close=True) -> PointsOrder:
    if len(points) < 3:
        raise ValueError(f"At least 3 points expected! \n{points}")
    if close:
        points = np.concatenate([points, [points[0]]])

    determinant = sum(
            (points[i + 1][0] - points[i][0]) * (points[i + 1][1] + points[i][1]) for i in range(len(points) - 1)
            )
    if determinant > 0:
        return PointsOrder.CW
    elif determinant < 0:
        return PointsOrder.CCW
    else:
        return PointsOrder.COLLINEAR
