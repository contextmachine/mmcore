"""A NON-PLANAR equidistant closest-point curve: the general "Pull" tracer.

Every closest-point set lies on the sphere of radius d_min about the query,
but it need NOT be a circle. This example constructs an exact witness:

* take a non-planar rational spherical curve γ(u) — the stereographic image
  of a PARABOLA (stereographic projection maps lines/circles to circles,
  i.e. plane sections; a parabola is neither, so its image is provably
  non-planar);
* rule a band along the sphere-tangent direction ``e_z × γ`` (which is
  perpendicular to the radial direction γ):

      S(u, v) = γ(u) + (2v − 1)·h·(e_z × γ(u))

  Since ``(e_z × γ) ⊥ γ``, the squared distance from the sphere center is
  EXACTLY ``R² + (2v−1)²·h²·‖e_z × γ‖²`` — minimal on the whole line v = ½,
  where the surface touches the sphere along γ.

The closest-point set is therefore the entire non-planar curve γ. The
solver's exact-circle certification must DECLINE (the curve is not planar),
and the general equidistant tracer — the seed of full Curve Pulling —
produces the answer as a traced polyline on the sphere.
"""
import numpy as np
from math import comb


def _mono_to_bern(a, n):
    """Monomial coefficients (low->high, len<=n+1) -> Bernstein, degree n."""
    a = np.pad(np.asarray(a, dtype=float), (0, n + 1 - len(a)))
    return np.array([sum(comb(i, k) / comb(n, k) * a[k] for k in range(i + 1))
                     for i in range(n + 1)])


def nonplanar_band(radius=2.0, a=0.8, c=0.3, h=0.5):
    """Rational Bézier band (degree 4 x 1, homogeneous (5,2,4) net) tangent to
    the sphere of ``radius`` about the origin along a non-planar curve.

    ``a``/``c`` shape the parabola ``q(t) = (t, c + a t²)``, ``t = 2u − 1``,
    whose stereographic image is the touching curve; ``h`` is the half-width
    of the band along the sphere-tangent ruling.
    """
    R = radius
    x = np.array([-1.0, 2.0])                      # t = 2u - 1
    y = np.array([c + a, -4 * a, 4 * a])           # c + a(2u-1)^2
    n2 = np.convolve(x, x)
    n2 = np.pad(n2, (0, 2)) + np.convolve(y, y)    # x^2 + y^2 (degree 4)
    w = n2.copy(); w[0] += 1.0                     # denominator x^2+y^2+1
    zn = n2.copy(); zn[0] -= 1.0                   # z numerator x^2+y^2-1
    two_x = np.pad(2 * x, (0, 3))
    two_y = np.pad(2 * y, (0, 2))
    # rows γ ∓ h·(e_z × γ); e_z × γ has numerator (-2y, 2x, 0) over the same w
    r0x = R * (two_x + h * two_y); r0y = R * (two_y - h * two_x)
    r1x = R * (two_x - h * two_y); r1y = R * (two_y + h * two_x)
    B = lambda p: _mono_to_bern(p, 4)
    bw = B(w); bz = R * B(zn)
    net = np.zeros((5, 2, 4))
    net[:, 0] = np.column_stack([B(r0x), B(r0y), bz, bw])
    net[:, 1] = np.column_stack([B(r1x), B(r1y), bz, bw])
    return net


if __name__ == "__main__":
    import time
    from mmcore.numeric._bez_closest_point import bez_surface_closest_points, eval_surface

    R = 2.0
    net = nonplanar_band(radius=R)
    query = np.zeros(3)

    t0 = time.perf_counter()
    res = bez_surface_closest_points(net, query, atol=1e-6, rational=True)
    dt = time.perf_counter() - t0

    print(f"non-planar equidistant band  ({dt * 1000:.1f} ms)")
    for e in res:
        print(f"  kind={e['kind']}  distance={e['distance']:.9f}  "
              f"circle-certified={'circle' in e}")
    curve = [e for e in res if e["kind"] == "degenerate_curve"][0]
    X = curve["points"]
    sphere_err = np.abs(np.linalg.norm(X, axis=1) - R).max()
    s = np.linalg.svd(X - X.mean(0), compute_uv=False)
    print(f"  traced {len(X)} points: on-sphere to {sphere_err:.2e}, "
          f"planarity ratio s_min/s_max = {s[-1] / s[0]:.4f} (0 would be a circle)")
    print("-> a genuinely non-planar spherical closest set: only the general "
          "Pull tracer can produce this.")

    # --- render (optional) ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit(0)
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    uu = np.linspace(0, 1, 60); vv = np.linspace(0, 1, 12)
    G = np.array([[eval_surface(net, u, v, rational=True) for v in vv] for u in uu])
    ax.plot_surface(G[..., 0], G[..., 1], G[..., 2], alpha=0.35, color="tab:blue",
                    linewidth=0, antialiased=True)
    th, ph = np.meshgrid(np.linspace(0, 2 * np.pi, 36), np.linspace(0, np.pi, 18))
    ax.plot_wireframe(R * np.cos(th) * np.sin(ph), R * np.sin(th) * np.sin(ph),
                      R * np.cos(ph), color="gray", alpha=0.15, linewidth=0.5)
    ax.plot(X[:, 0], X[:, 1], X[:, 2], color="red", linewidth=3,
            label="equidistant closest set (non-planar)")
    ax.scatter(*query, color="black", s=40, label="query point")
    ax.set_box_aspect((1, 1, 1)); ax.legend(); ax.set_title(
        "Band tangent to the sphere along a non-planar spherical curve")
    out = "nonplanar_equidistant_band.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"figure saved to {out}")
