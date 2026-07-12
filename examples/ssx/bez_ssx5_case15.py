"""case 15 (ledger L25, 2026-07-12): edge-graze — transversal arc hugging a domain edge.

S1 is the graph of z(s,t) = h*s - c*(t - t_star)^2 - h*d0 over (s,t) in [0,1]^2 —
an exact degree-(1,2) Bezier patch. S2 is the plane z=0. The true SSI is the
parabola

    s(t) = (c/h) * (t - t_star)^2 + d0,   xyz = (s, t, 0),   t in [0,1]

whose closest approach to S1's s=0 domain edge is d0 (in both parameter and xyz
units): d0 = 0 grazes the edge exactly (tangent at one point), d0 > 0 passes
just INSIDE without touching, d0 < 0 dips outside (two transversal s=0 boundary
crossings). The intersection is TRANSVERSAL everywhere in 3D (surface normals
at closest approach are 45 degrees apart) — the (near-)tangency is between the
SSI curve and the parameter-domain EDGE, not between the surfaces.

The module-level pair is the L25 repro: an INSIDE near-graze (d0 = 1e-4, i.e.
0.1*atol clearance, off-lattice t_star). Before the L25 fix the marcher's
fixed-face exit Newton stalled at the closest-approach witness (residual = d0,
NOT a root — the face carries no Psi zero), committed it as an exit vertex at
the atol-scale acceptance bar, and the strict path certificate then rejected
every fragment: the ENTIRE arc vanished (0 branches, reasons=['trace_unverified']).
At d0 ~ atol the exit was refused instead and the march broke off mid-curve:
two truncated branches with a silent gap, falsely complete.

EXPECTED: one transversal branch covering the whole parabola through the
near-graze; zero singularities; complete=True; full analytic coverage.
"""
import numpy as np

RATIONAL = False

# Default variant: off-lattice near-graze passing 0.1*atol inside the s=0 edge.
T_STAR = 0.37
C_SHARP = 1.0
H_SLOPE = 1.0
D_CLEAR = 1e-4


def build_graze_pair(t_star=T_STAR, c=C_SHARP, h=H_SLOPE, d0=D_CLEAR, scale=1.0):
    """S1 = graph of h*s - c*(t-t_star)^2 - h*d0 (degree (1,2)); S2 = plane z=0."""
    r0 = np.array([-c * t_star ** 2,
                   c * t_star * (1.0 - t_star),
                   -c * (1.0 - t_star) ** 2]) - h * d0
    S1 = np.array([
        [[0.0, 0.0, r0[0]], [0.0, 0.5, r0[1]], [0.0, 1.0, r0[2]]],
        [[1.0, 0.0, r0[0] + h], [1.0, 0.5, r0[1] + h], [1.0, 1.0, r0[2] + h]],
    ]) * scale
    S2 = np.array([
        [[-0.5, -0.5, 0.0], [-0.5, 1.5, 0.0]],
        [[1.5, -0.5, 0.0], [1.5, 1.5, 0.0]],
    ]) * scale
    return S1, S2


def analytic_curve(t_star=T_STAR, c=C_SHARP, h=H_SLOPE, d0=D_CLEAR, scale=1.0,
                   n=2001):
    """Exact SSI samples, clipped to S1's domain (s in [0,1])."""
    t = np.linspace(0.0, 1.0, n)
    s = (c / h) * (t - t_star) ** 2 + d0
    m = (s >= 0.0) & (s <= 1.0)
    return np.column_stack([s[m], t[m], np.zeros(int(m.sum()))]) * scale


S1, S2 = build_graze_pair()


if __name__ == "__main__":
    from mmcore.numeric.intersection.ssx._bez_ssx5 import bez_ssx
    from bez_ssx5_coverage_check import point_to_polyline_dist

    atol = 1e-3
    res = bez_ssx(S1, S2, atol, rational=RATIONAL)
    print(f"branches={len(res['branches'])} points={len(res['points'])} "
          f"singularities={[g.kind for g in res.get('singularities', [])]} "
          f"complete={res.get('complete')} "
          f"reasons={res.get('status', {}).get('reasons', [])}")
    polys = []
    for b in res["branches"]:
        xyz = np.asarray(b.curve[1])
        polys.append(xyz)
        print(f"  branch kind={getattr(b, 'kind', '?')} n={len(xyz)} "
              f"{np.round(xyz[0], 4).tolist()} -> {np.round(xyz[-1], 4).tolist()}")
    truth = analytic_curve()
    d = np.array([min((point_to_polyline_dist(p, poly) for poly in polys),
                      default=np.inf) for p in truth])
    missed = d > 5 * atol
    print(f"analytic coverage {int((~missed).sum())}/{len(truth)}"
          + (f"; MISSED t-ranges around {np.round(truth[missed][:, 1], 4).tolist()[:10]}"
             f" worst {float(d[missed].max()):.5f}" if missed.any() else ""))
