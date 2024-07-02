from mmcore.numeric.vectors import scalar_norm, scalar_dot, scalar_cross
import numpy as np


def frenet_serret_frame(curve, curve_prime, curve_double_prime, t):
    """
    Calculate the Frenet-Serret frame for a given parametric curve and its derivatives.
    Compute the Frenet-Serret frame for a given curve at a specified parameter value.

    :param curve: A function that takes a float t and returns a 3D point (x, y, z).
    :type curve: function
    :param curve_prime: function representing the first derivative of the curve
    :type curve_prime: function
    :param curve_double_prime: function representing the second derivative of the curve
    :type curve_double_prime: function
    :param t: parameter value at which to evaluate the Frenet-Serret frame
    :type t: float
    :return: array containing the position vector, tangent vector, normal vector, and binormal vector
    :rtype: numpy.ndarray

    """
    # Compute the derivatives
    r_t = curve(t)
    r_prime_t = curve_prime(t)
    r_double_prime_t = curve_double_prime(t)
    # Compute the tangent vector T(t) and normalize it
    r_prime_norm=scalar_norm(r_prime_t)
    T = r_prime_t /   r_prime_norm
    # Compute the normal vector N(t)
    T_prime = (r_double_prime_t - scalar_dot(r_double_prime_t, T) * T) /   r_prime_norm

    N = T_prime / scalar_norm(T_prime)
    # Compute the binormal vector B(t)
    B = scalar_cross(T, N)
    return np.array([r_t, T, N, B])


if __name__ == '__main__':
    pts = np.array(
        [(-9.1092663228073292, -12.711321277810857, -0.77093266173210928),
         (-1.5012583168504101, -15.685662924609387, -6.6022178296290024),
         (0.62360921189203689, -15.825362292273830, 2.9177845739234654),
         (7.7822965141636233, -14.858282311330257, -5.1454157090841059)])
    from mmcore.geom.bspline import NURBSpline

    nc = NURBSpline(pts)
    t = 0.3
    pln = np.asarray(nc.plane_at(t))
    tnb_frame = frenet_serret_frame(nc.evaluate, nc.derivative, nc.second_derivative, t)
    print(np.allclose(pln, tnb_frame))
    print(pln, tnb_frame)
