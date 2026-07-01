"""Closest point from the CENTER of a sphere: the whole patch is the answer.

Every point of the octant is exactly at distance R from the center, so there
is no meaningful "closest point" — the solver recognizes this algebraically
(the rational Bernstein ratio F_i/W_i is constant, so the certificate fires at
the root cell, in a single pop) and returns one ``degenerate_surface`` entity
instead of an arbitrary implementation-chance point.
"""
import numpy as np

from mmcore.numeric._bez_closest_point import bez_surface_closest_points


def sphere_octant(radius=1.0):
    """Rational biquadratic octant of a sphere, homogeneous (3,3,4) net."""
    s = np.sqrt(2) / 2
    cp = radius * np.array([
        [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
        [[1, 0, 1], [1, 1, 1], [0, 1, 1]],
        [[1, 0, 0], [1, 1, 0], [0, 1, 0]],
    ], dtype=float)
    w = np.array([[1.0, s, 1.0], [s, 0.5, s], [1.0, s, 1.0]])
    return np.concatenate([cp * w[:, :, None], w[:, :, None]], axis=2)


if __name__ == "__main__":
    import time

    R = 2.0
    surf = sphere_octant(R)
    center = np.zeros(3)

    t0 = time.perf_counter()
    res = bez_surface_closest_points(surf, center, atol=1e-6, rational=True)
    dt = time.perf_counter() - t0

    print(f"query at sphere center, R={R}  ({dt * 1000:.2f} ms)")
    for e in res:
        print(f"  kind={e['kind']}  distance={e['distance']:.9f}  "
              f"u_range={e['u_range']}  v_range={e['v_range']}")
    assert res[0]["kind"] == "degenerate_surface"
    print("-> the ENTIRE patch is equidistant: no single closest point exists.")
