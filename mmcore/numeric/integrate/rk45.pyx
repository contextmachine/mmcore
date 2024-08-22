import numpy as np



def rk45_vector(f, t0, y0, t1, tol=1e-6, hmax=0.1, hmin=1e-6):
    def rk_step(f, t, y, h):
        k1 = h * f(t, y)
        k2 = h * f(t + 1 / 4 * h, y + 1 / 4 * k1)
        k3 = h * f(t + 3 / 8 * h, y + 3 / 32 * k1 + 9 / 32 * k2)
        k4 = h * f(
            t + 12 / 13 * h, y + 1932 / 2197 * k1 - 7200 / 2197 * k2 + 7296 / 2197 * k3
        )
        k5 = h * f(
            t + h, y + 439 / 216 * k1 - 8 * k2 + 3680 / 513 * k3 - 845 / 4104 * k4
        )
        k6 = h * f(
            t + 1 / 2 * h,
            y
            - 8 / 27 * k1
            + 2 * k2
            - 3544 / 2565 * k3
            + 1859 / 4104 * k4
            - 11 / 40 * k5,
        )

        y_rk4 = y + 25 / 216 * k1 + 1408 / 2565 * k3 + 2197 / 4104 * k4 - 1 / 5 * k5
        y_rk5 = (
            y
            + 16 / 135 * k1
            + 6656 / 12825 * k3
            + 28561 / 56430 * k4
            - 9 / 50 * k5
            + 2 / 55 * k6
        )

        return y_rk4, y_rk5

    vt = []
    vy = []

    h = hmax
    t = t0
    y = y0

    while t < t1:
        if t + h > t1:
            h = t1 - t

        y_rk4, y_rk5 = rk_step(f, t, y, h)
        error = np.linalg.norm(y_rk5 - y_rk4)

        if error <= tol:
            t += h
            y = y_rk5
            vt.append(t)
            vy.append(y)

        h = min(hmax, max(hmin, 0.9 * h * (tol / error) ** 0.25))

    return np.array(vt), np.array(vy)


def differential_equation(t: float, uv: np.ndarray) -> np.ndarray:
    # Example system: dx/dt = y, dy/dt = -x (Simple harmonic oscillator)
    x, y = uv
    return np.array([y, -x])


# Example usage:
t0, y0 = 0.0, np.array([1.0, 0.0])  # Initial condition (x=1, y=0 at t=0)
t1 = 10.0  # Final time
vt, vy = rk45_vector(differential_equation, t0, y0, t1)

# Display the results
for ti, yi in zip(vt, vy):
    print(f"t: {ti:.5f}, y: {yi}")
