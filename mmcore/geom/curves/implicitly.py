import numpy as np

from mmcore.numeric.vectors import dot

from mmcore.numeric import scalar_norm
from mmcore.numeric.algorithms.point_inversion import point_inversion_curve

from collections import namedtuple
__all__=['implicitly']

ImplicitlyParametricCurveResult=namedtuple("ImplicitlyParametricCurveResult",['dist', 'grad','t'])


class ImplicitlyCurve:
    def __init__(self, curve):
        self.curve=curve
        self._build()
    def _build(self):
        self._t = np.linspace(*self.curve.interval(), 100)
        self._pts = self.curve(self._t)
    def __call__(self, xyz)->ImplicitlyParametricCurveResult:
        drs = xyz - self._pts
        dots = dot(drs, drs)
        ix = np.argmin(dots)
        t_on_curve = point_inversion_curve(self.curve, xyz, float(self._t[ix]), 1e-6, 1e-6)
        pt_on_curve = self.curve.evaluate(t_on_curve)
        grad = pt_on_curve - xyz
        dist = scalar_norm(grad)
        return ImplicitlyParametricCurveResult(dist, grad, t_on_curve)

def implicitly(crv):
    """
    Transforms a parametric curve into its implicit form, enabling distance and gradient calculation
    from any point in space to the nearest point on the curve.

    :param crv: A curve object that supports parametric evaluation. The curve object must implement the following methods:
                - `interval()`: Returns the interval (start and end) of the curve parameterization.
                - `__call__(t)`: Evaluates the curve at parameter `t`, returning the corresponding point in space.
                - `evaluate(t)`: Evaluates the curve at parameter `t` and returns the corresponding point in space.

    :return: A function that takes a point in space (as a NumPy array `xyz`) and returns an `ImplicitlyParametricCurveResult` object,
             which includes:
             - `dist` (float): The shortest distance from the point `xyz` to the curve.
             - `grad` (np.ndarray): The gradient vector pointing from the point `xyz` to the nearest point on the curve.
             - `t` (float): The parameter value on the curve corresponding to the nearest point.

    :rtype: callable

    :usage:
    ```
    crv = SomeCurveClass(...)  # A curve object with necessary methods implemented
    implicit_function = implicitly(crv)

    point = np.array([x, y, z])
    result = implicit_function(point)

    print("Distance to curve:", result.dist)
    print("Gradient to curve:", result.grad)
    print("Parameter on curve:", result.t)
    ```

    :notes:
    - The function computes the distance and gradient by finding the nearest point on the curve to the input point.
    - The nearest point on the curve is determined by evaluating the curve over a sampled parameter range and refining
      the closest match using a point inversion algorithm.

    :example:
    Consider a curve object `crv` with a parameter interval of [0, 1]. Given a point `xyz` in space, the function can be used
    as follows:

    ```
    implicit = implicitly(crv)
    result = implicit(np.array([1.0, 2.0, 3.0]))

    print(f"Distance to curve: {result.dist}")
    print(f"Gradient: {result.grad}")
    print(f"Nearest point on curve at parameter: {result.t}")
    ```

    This will print the distance from the point `[1.0, 2.0, 3.0]` to the curve, the gradient vector pointing to the nearest
    point on the curve, and the parameter value on the curve at that nearest point.
    """

    return ImplicitlyCurve(crv)
