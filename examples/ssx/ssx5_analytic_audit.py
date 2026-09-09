"""Independent, deterministic SSX coverage audit with algebraically known zeros.

Every input is S(u,v)=(u,v,product(f_i(u,v))) against z=0.  The f_i
are explicitly prescribed line or circle factors; their real zero sets in
the unit square are the complete reference, independently of SSX and CSX.
Distinct circles below have disjoint zero sets and nonzero gradients, and
line roots are distinct, so all generated components are regular.

Coverage is a continuous bound, not just a sampled hit rate: distance to a
polyline is 1-Lipschitz and the reference's maximum arc spacing is known.
We add half that spacing to the maximum measured sample distance.  Output
soundness is checked on whole segments using exact line/circle distance
extrema; length, closure and component ownership detect duplicate traces,
fragmentation and accidental bridges independently of branch count.

Run a fresh process per case with an external watchdog::

    python examples/ssx/ssx5_analytic_audit.py --timeout 45 --output /tmp/audit.json
    python examples/ssx/ssx5_analytic_audit.py --case nested_circles --variant swap

No solver tolerance or work limits are adjusted to make individual cases pass.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

# Direct script execution must use this checkout, including isolated worktrees.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@dataclass(frozen=True)
class Component:
    kind: str
    data: tuple[float, ...]

    @property
    def length(self):
        return 1.0 if self.kind == "line" else 2.0 * math.pi * self.data[2]

    def reference(self, max_spacing):
        n = max(8, int(math.ceil(self.length / max_spacing)))
        if self.kind == "line":
            xyz = np.column_stack((np.full(n + 1, self.data[0]),
                                   np.linspace(0., 1., n + 1), np.zeros(n + 1)))
        else:
            cx, cy, radius = self.data
            theta = np.arange(n) * (2.0 * math.pi / n)
            xyz = np.column_stack((cx + radius * np.cos(theta),
                                   cy + radius * np.sin(theta), np.zeros(n)))
        return xyz, self.length / n

    def distance(self, xyz):
        xyz = np.asarray(xyz)
        if self.kind == "line":
            return np.sqrt((xyz[..., 0] - self.data[0]) ** 2 + xyz[..., 2] ** 2
                           + np.maximum(0., np.maximum(-xyz[..., 1], xyz[..., 1]-1.)) ** 2)
        cx, cy, radius = self.data
        radial = np.linalg.norm(xyz[..., :2] - [cx, cy], axis=-1) - radius
        return np.hypot(radial, xyz[..., 2])

    def segment_error_bound(self, xyz):
        if self.kind == "line" or len(xyz) < 2:
            # Distance to a closed convex line segment is convex.
            return float(self.distance(xyz).max(initial=0.))
        cx, cy, radius = self.data
        a, b = xyz[:-1, :2] - [cx, cy], xyz[1:, :2] - [cx, cy]
        delta = b - a
        den = np.einsum("ij,ij->i", delta, delta)
        t = np.clip(-np.einsum("ij,ij->i", a, delta) / np.maximum(den, 1e-300), 0., 1.)
        min_r = np.linalg.norm(a + t[:, None] * delta, axis=1)
        max_r = np.maximum(np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1))
        radial_error = np.maximum(np.abs(min_r-radius), np.abs(max_r-radius))
        max_z = np.maximum(np.abs(xyz[:-1, 2]), np.abs(xyz[1:, 2]))
        return float(np.hypot(radial_error, max_z).max(initial=0.))


def polynomial_product(a, b):
    out = np.zeros((a.shape[0] + b.shape[0] - 1,
                    a.shape[1] + b.shape[1] - 1))
    for i, j in np.ndindex(a.shape):
        out[i:i+b.shape[0], j:j+b.shape[1]] += a[i, j] * b
    return out


def power_to_bernstein(coefficients):
    """Exact power/Bernstein basis identity, rounded only in float arithmetic."""
    m, n = np.asarray(coefficients).shape
    u = np.zeros((m, m))
    v = np.zeros((n, n))
    for i in range(m):
        for k in range(i + 1):
            u[i, k] = math.comb(i, k) / math.comb(m - 1, k)
    for j in range(n):
        for k in range(j + 1):
            v[j, k] = math.comb(j, k) / math.comb(n - 1, k)
    return u @ coefficients @ v.T


def graph_pair(components):
    polynomial = np.ones((1, 1))
    for component in components:
        if component.kind == "line":
            factor = np.array([[-component.data[0]], [1.]])
        else:
            cx, cy, radius = component.data
            factor = np.array([[cx*cx + cy*cy - radius*radius, -2.*cy, 1.],
                               [-2.*cx, 0., 0.], [1., 0., 0.]])
        polynomial = polynomial_product(polynomial, factor)
    # Degree at least one in both axes so the graph remains a regular surface.
    polynomial = np.pad(polynomial,
                        ((0, max(0, 2-polynomial.shape[0])),
                         (0, max(0, 2-polynomial.shape[1]))))
    z = power_to_bernstein(polynomial)
    s1 = np.empty(z.shape + (3,))
    s1[..., 0] = np.linspace(0., 1., len(z))[:, None]
    s1[..., 1] = np.linspace(0., 1., z.shape[1])[None, :]
    s1[..., 2] = z
    # Containment has a margin, so only the graph's boundary owns line ends.
    s2 = np.array([[[-.25, -.25, 0.], [-.25, 1.25, 0.]],
                   [[1.25, -.25, 0.], [1.25, 1.25, 0.]]])
    return s1, s2


def case_components(name):
    cases = {
        "one_line": (Component("line", (3/8,)),),
        "two_lines": tuple(Component("line", (x,)) for x in (1/4, 3/4)),
        "four_lines": tuple(Component("line", (x,)) for x in (1/8, 3/8, 5/8, 7/8)),
        "nearby_lines": tuple(Component("line", (x,)) for x in (15/32, 17/32)),
        "tight_lines": tuple(Component("line", (x,)) for x in (127/256, 129/256)),
        "one_circle": (Component("circle", (1/2, 1/2, 1/4)),),
        "two_circles": (Component("circle", (1/4, 1/2, 3/16)),
                        Component("circle", (3/4, 1/2, 3/16))),
        "nested_circles": (Component("circle", (1/2, 1/2, 3/16)),
                           Component("circle", (1/2, 1/2, 3/8))),
        "nearby_circles": (Component("circle", (1/2, 1/2, 7/32)),
                           Component("circle", (1/2, 1/2, 9/32))),
    }
    return cases[name]


CASES = ("one_line", "two_lines", "four_lines", "nearby_lines", "tight_lines",
         "one_circle", "two_circles", "nested_circles", "nearby_circles")
VARIANTS = ("identity", "swap", "reverse_u", "reverse_v", "transpose", "split_u", "split_uv")


def _nurbs(net):
    from mmcore.nurbs._nurbs_eval import NURBSSurfaceTuple
    m, n = net.shape[:2]
    return NURBSSurfaceTuple(order_u=m, order_v=n,
                            knot_u=np.array([0.]*m + [1.]*m),
                            knot_v=np.array([0.]*n + [1.]*n),
                            control_points=net, weights=np.ones((m, n)))


def _split_axis(surf, axis):
    from mmcore.numeric.bern import de_casteljau_split_nd
    from mmcore.nurbs._nurbs_eval import NURBSSurfaceTuple
    left, right = de_casteljau_split_nd(surf.control_points, axis=axis, t=.5)
    cp = (np.concatenate((left, right[1:]), axis=0) if axis == 0
          else np.concatenate((left, right[:, 1:]), axis=1))
    order = surf.order_u if axis == 0 else surf.order_v
    knot = np.array([0.]*order + [.5]*(order-1) + [1.]*order)
    return NURBSSurfaceTuple(order_u=surf.order_u, order_v=surf.order_v,
                            knot_u=knot if axis == 0 else surf.knot_u,
                            knot_v=knot if axis == 1 else surf.knot_v,
                            control_points=cp, weights=np.ones(cp.shape[:2]))


def solve_case(name, variant="identity", atol=1e-3):
    from mmcore.numeric.intersection.ssx._bez_ssx5 import bez_ssx
    s1, s2 = graph_pair(case_components(name))
    if variant.startswith("split_"):
        from mmcore.numeric.intersection.ssx._nssx5 import nurbs_ssx
        a, b = _split_axis(_nurbs(s1), 0), _nurbs(s2)
        if variant == "split_uv":
            a = _split_axis(a, 1)
        return nurbs_ssx(a, b, atol=atol)
    if variant == "swap":
        s1, s2 = s2, s1
    elif variant == "reverse_u":
        s1 = s1[::-1].copy()
    elif variant == "reverse_v":
        s1 = s1[:, ::-1].copy()
    elif variant == "transpose":
        s1 = s1.transpose(1, 0, 2).copy()
    return bez_ssx(s1, s2, atol=atol, rational=False)


def distances_to_polylines(points, polylines):
    best = np.full(len(points), np.inf)
    for polyline in polylines:
        if len(polyline) == 1:
            best = np.minimum(best, np.linalg.norm(points-polyline[0], axis=1))
        for a, b in zip(polyline[:-1], polyline[1:]):
            delta = b-a
            den = float(delta @ delta)
            if den == 0.:
                continue
            t = np.clip((points-a) @ delta / den, 0., 1.)
            best = np.minimum(best, np.linalg.norm(points-a-t[:, None]*delta, axis=1))
    return best


def audit_result(name, result, atol=1e-3):
    components = case_components(name)
    polylines = [np.asarray(b.curve[1]) for b in result["branches"]]
    component_polylines = [[] for _ in components]
    branch_details = []
    failures = []
    for index, (branch, xyz) in enumerate(zip(result["branches"], polylines)):
        errors = [component.segment_error_bound(xyz) for component in components]
        owner = int(np.argmin(errors))
        component_polylines[owner].append(xyz)
        length = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
        branch_details.append({"branch": index, "component": owner, "vertices": len(xyz),
                               "length": length, "closed": bool(branch.closed),
                               "endpoint_gap": float(np.linalg.norm(xyz[-1]-xyz[0])),
                               "soundness_upper": errors[owner]})
        if errors[owner] > 4.*atol:
            failures.append(f"branch {index} crosses components or leaves the exact zero set")
        if components[owner].kind == "circle" and not branch.closed:
            failures.append(f"branch {index} lacks closed metadata for a closed component")
        if components[owner].kind == "line" and branch.closed:
            failures.append(f"branch {index} incorrectly marks an open component closed")
    component_details = []
    for index, (component, owned) in enumerate(zip(components, component_polylines)):
        ref, spacing = component.reference(max_spacing=atol/2.)
        distances = distances_to_polylines(ref, owned)
        coverage_upper = float(distances.max()) + spacing/2.
        coverage_lower = float(distances.max())
        length = sum(float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) for p in owned)
        detail = {"component": index, "kind": component.kind, "parameters": list(component.data),
                  "branches": len(owned), "coverage_lower": coverage_lower,
                  "coverage_upper": coverage_upper, "expected_length": component.length,
                  "output_length": length}
        if component.kind == "circle":
            center = component.data[:2]
            angular_travel = 0.
            for poly in owned:
                relative = poly[:, :2] - center
                angle = np.unwrap(np.arctan2(relative[:, 1], relative[:, 0]))
                angular_travel += float(np.abs(np.diff(angle)).sum())
            detail["angular_travel"] = angular_travel
            # Two endpoint neighborhoods each spend the 4*atol positional
            # envelope. Convert that to an angle, independent of case scale.
            angle_slack = 2.*math.asin(min(1., 4.*atol/component.data[2]))
            if abs(angular_travel-2.*math.pi) > angle_slack:
                failures.append(f"component {index} angular travel indicates loss or retrace")
        component_details.append(detail)
        if coverage_upper > 4.*atol:
            failures.append(f"component {index} lacks certified continuous coverage")
        if len(owned) != 1:
            failures.append(f"component {index} has {len(owned)} output branches instead of one")
        if abs(length-component.length) > 8.*math.pi*atol:
            failures.append(f"component {index} length indicates loss or retrace")
    if result.get("points") or result.get("singularities") or result.get("overlap_regions"):
        failures.append("regular analytic components acquired point, singularity, or overlap output")
    complete = bool(result.get("complete", False))
    return {"case": name, "atol": atol, "complete": complete,
            "reasons": result.get("status", {}).get("reasons", []),
            "work": result.get("status", {}).get("work", {}),
            "branch_count": len(polylines), "point_count": len(result.get("points", [])),
            "singularity_count": len(result.get("singularities", [])),
            "components": component_details, "branches": branch_details,
            "failures": failures, "silent_failure": complete and bool(failures),
            "passed": complete and not failures}


def run_case(name, variant, atol):
    start = time.monotonic()
    result = solve_case(name, variant=variant, atol=atol)
    report = audit_result(name, result, atol=atol)
    report.update(variant=variant, elapsed_s=time.monotonic()-start)
    return report


def _json_ready(value):
    """Represent unbounded distance (an absent component) with JSON null."""
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=CASES)
    parser.add_argument("--variant", action="append", choices=VARIANTS)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--timeout", type=float, default=45.)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    cases, variants = args.case or CASES, args.variant or ["identity"]
    if args.worker:
        print("SSX_AUDIT_JSON=" + json.dumps(_json_ready(run_case(cases[0], variants[0], args.atol)),
                                             allow_nan=False))
        return
    reports = []
    for name in cases:
        for variant in variants:
            cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--case", name,
                   "--variant", variant, "--atol", str(args.atol)]
            try:
                completed = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
                marker = next((line.removeprefix("SSX_AUDIT_JSON=")
                               for line in completed.stdout.splitlines()
                               if line.startswith("SSX_AUDIT_JSON=")), None)
                report = (json.loads(marker) if marker is not None else
                          {"case": name, "variant": variant, "error": completed.stderr,
                           "returncode": completed.returncode, "passed": False})
            except subprocess.TimeoutExpired:
                report = {"case": name, "variant": variant, "timeout_s": args.timeout, "passed": False}
            reports.append(report)
            print(json.dumps(report, allow_nan=False), flush=True)
            if args.output:
                args.output.write_text(json.dumps(reports, indent=2, allow_nan=False) + "\n")
    return 0 if all(report["passed"] for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
