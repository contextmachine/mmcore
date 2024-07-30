from typing import Any
import numpy as np
from numpy.typing import ArrayLike
from mmcore.geom.curves.curve import Curve
from mmcore.geom.curves.bspline import NURBSpline

from mmcore.geom.curves.bspline_utils import (
    calc_b_spline_point,
    calcNURBSDerivatives,
    calc_bspline_derivatives,
    calc_rational_curve_derivatives,
)


from mmcore.geom.curves.knot import (
    find_span_binsearch,
    find_multiplicity,
    knot_insertion,
)




class Circle(Curve):
    def __init__(self, radius, origin=np.array([0.0, 0.0, 0.0])):
        super().__init__()
        self.r = radius
        self.origin = origin

    @property
    def a(self):
        return self.r

    @property
    def b(self):
        return self.origin[0]

    @property
    def c(self):
        return self.origin[1]

    def fx(self, x):
        _ = np.sqrt(self.a ** 2 - (x - self.b) ** 2)
        return np.array([self.c + _, self.c - _])

    def fy(self, y):
        _ = np.sqrt(self.a ** 2 - (y - self.c) ** 2)
        return np.array([self.b + _, self.b - _])

    def implict(self, xy):
        return (
                (xy[0] - self.origin[0]) ** 2 + (xy[1] - self.origin[1]) ** 2 - self.r ** 2
        )

    def intersect_with_circle(self, circle):
        ...

    def evaluate(self, t: float) -> ArrayLike:
        return np.array(
            [self.r * np.cos(t) + self.origin[0], self.r * np.sin(t) + self.origin[1]]
        )




if __name__ == "__main__":
    import numpy as np

    a1 = NURBSpline(
        np.array(
            [
                (30.184638404201344, -18.216164837439184, 0.0),
                (15.325025552531345, -49.500456857454566, 0.0),
                (0.33619867606420506, -38.000408650509947, 0.0),
                (2.2915627545368258, -10.800856430713994, 0.0),
                (34.577915247303785, -29.924532100689298, 0.0),
                (24.771126815705877, -44.396502877967905, 0.0),
                (8.7351102878776850, -27.081823555152429, 0.0),
                (0.60796701514639295, -28.615956860732620, 0.0),
            ]
        )
    )
    a2 = NURBSpline(
        np.array(
            [
                (7.2648314876233702, -17.952160046548514, 0.0),
                (2.1216889176987861, -39.948793039632903, 0.0),
                (15.124018315255334, -10.507711766165173, 0.0),
                (44.907234268165922, -36.066799609839038, 0.0),
                (-6.5507389082519225, -35.613653473099788, 0.0),
            ]
        )
    )

    import time

    s = time.time()

    print(a1.intersect_with_curve(
        a2
    ))
    print(divmod(time.time() - s, 60))
