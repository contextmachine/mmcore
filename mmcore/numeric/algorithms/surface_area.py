import numpy as np
from scipy.integrate import dblquad
from mmcore.numeric.fdm import DEFAULT_H


def _partial_derivatives_fun(surface, u, v, h=DEFAULT_H):
    # Compute partial derivatives numerically
    x, y, z = surface((u, v))
    xu, yu, zu = surface((u + h, v))
    xv, yv, zv = surface((u, v + h))
    r_u = ((xu - x) / h, (yu - y) / h, (zu - z) / h)
    r_v = ((xv - x) / h, (yv - y) / h, (zv - z) / h)
    return np.array(r_u), np.array(r_v)


def _partial_derivatives_cls(surface, u, v):
    uv = np.array([u, v])
    return surface.derivative_u(uv), surface.derivative_v(uv)


# Define the domain of u and v
u_min, u_max = 0, np.pi * 2
v_min, v_max = -1.0, 1.0
from mmcore.numeric.vectors import scalar_cross, scalar_norm


def surface_rect_area(surface, bounds=None):
    """

    :param surface:
    :param bounds:
    :return:

    Example
    -----
    >>> def cylinder(uv):
    ...     u, v = uv
    ...     x = np.cos(u)
    ...     y = np.sin(u)
    ...     z = v
    ...     return np.array((x, y, z))



res1=surface_rect_area(cylinder, ((0.0, 2 * np.pi), (-1.0, 1.0)))
    """
    if hasattr(surface, "derivative_u") and hasattr(surface, "derivative_v"):

        def integrand(u, v):
            r_u, r_v = _partial_derivatives_cls(surface, u, v)
            cross_product = scalar_cross(r_u, r_v)
            return scalar_norm(cross_product)

    else:

        def integrand(u, v):
            r_u, r_v = _partial_derivatives_fun(surface, u, v)
            cross_product = scalar_cross(r_u, r_v)
            return scalar_norm(cross_product)

    if bounds is None:
        bounds = surface.interval()
    (u_min, u_max), (v_min, v_max) = bounds
    # Compute the area by integrating the integrand over the domain
    return dblquad(integrand, v_min, v_max, u_min, u_max)



