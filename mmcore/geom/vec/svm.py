import sys
from typing import Any

import numpy as np
from scipy.optimize._minimize import minimize_scalar, _minimize_scalar_bounded

from mmcore.geom.vec import dot
from mmcore.func import vectorize


@vectorize(signature='(i, j),(j)->(j)')
def support_vector(vertices: np.ndarray[Any, np.dtype[float]], d: np.ndarray[Any, np.dtype[float]]) -> np.ndarray[
    Any, np.dtype[float]]:
    """Support Vector Method

    :param vertices: An array of vertices
    :type vertices: np.ndarray[Any, np.dtype[float]]

    :param d: A vector
    :type d: np.ndarray[Any, np.dtype[float]]

    :return: The support vector
    :rtype: np.ndarray[Any, np.dtype[float]]

    Illustration:
    -----------

            vertices
      +  +
     +      +
     +        +
      +         +
       +         +
        +      + │
          +  +   │
                 │    d
    ─────────────┴─── -->

    """
    highest = -sys.float_info.max
    support = np.zeros(d.shape, dtype=d.dtype)

    for v in vertices:
        dot_value = dot(v, d)

        if dot_value > highest:
            highest = dot_value
            support = v

    return support


def curve_support_vector(curve, bounds=np.array([0, 1]), **props):
    """

    :param curve:
    :type curve:
    :param bounds:
    :type bounds:
    :param props: Additional properties. see
    :type props:
    :return:
    :rtype:
    """

    @vectorize(signature='(i)->()')
    def wrap(vector):
        def objective(t):
            return -1 * dot(vector, curve(t))

        return minimize_scalar(objective,
                               method='Bounded',
                               bounds=bounds, **props).x

    return wrap

