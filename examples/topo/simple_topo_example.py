"""Build a BRep by slicing a NURBS surface with isocurves.

Demonstrates:
- Manual quad-face construction with geometry attachment
- Repeated split_face_by_curve to subdivide into strips
- brep_to_loop_wires for visualization
"""
import copy
import json
import numpy as np
from pathlib import Path

from mmcore.geom._nurbs_eval import NURBSSurfaceTuple
from mmcore.geom._nurbs_knots import trim_curve
from mmcore.geom.nurbs_iso import extract_isocurve
from mmcore.numeric.closest_point import nurbs_curve_closest_point
from mmcore.topo.brep import BRep

# ── Load surface from JSON ───────────────────────────────────────────────────

fp = Path(__file__).parent / "example_surf_01.json"
with open(fp) as f:
    val = NURBSSurfaceTuple(
        **{k: (np.asarray(v, dtype=float) if isinstance(v, list) else v) for k, v in json.load(f).items()}
    )

# ── Extract boundary and interior isocurves ──────────────────────────────────

(u_min, u_max), (v_min, v_max) = val.interval()
u_0 = extract_isocurve(val, v_min, direction="v")  # bottom boundary
u_1 = extract_isocurve(val, v_max, direction="v")  # top boundary

u_params = [
    0.0, 324.18864044415017, 648.37728088830033, 972.56592133245044,
    1296.7545617766007, 1620.9432022207507, 1945.1318426649009,
    2269.3204831090511, 2593.5091235532013,
]
v_lines = [extract_isocurve(val, u, direction="u") for u in u_params]

# ── Build initial quad face ──────────────────────────────────────────────────


def _compute_edge_params(brep, edge):
    """Set edge.param by projecting vertex points onto the curve."""
    crv = brep.G_CRV[edge.geom]
    t0, (dst1, *_) = nurbs_curve_closest_point(crv, brep.V[edge.v_start].point)
    t1, (dst2, *_) = nurbs_curve_closest_point(crv, brep.V[edge.v_end].point)
    edge.param = t0, t1


brep = BRep()

# 4 vertices, 4 edges → one quad face
v1, v2, e1, l1, f1, s1 = brep.MEVVLS(tuple(u_0.start().tolist()), tuple(u_0.end().tolist()))
e1.geom = brep.new_curve(u_0)
e1.param = u_0.interval()

v3, e2 = brep.MEV(l1.id, v2.id, p_new=tuple(v_lines[-1].end().tolist()))
e2.geom = brep.new_curve(v_lines[-1])
e2.param = v_lines[-1].interval()

v4, e3 = brep.MEV(l1.id, v3.id, p_new=tuple(u_1.start().tolist()))
e3.geom = brep.new_curve(u_1)
e3.param = u_1.interval()

e4, l2, f2 = brep.MELF(l1.id, v4.id, v1.id)
e4.geom = brep.new_curve(v_lines[0])
e4.param = v_lines[0].interval()

# Assign surface geometry to the face
f2.surf = brep.new_surface(val)

# Recompute exact param ranges from vertex projections
for e in brep.E.values():
    if e.geom is not None:
        _compute_edge_params(brep, e)

assert brep.validate() == [], brep.validate()

# ── Split the face with interior isocurves ───────────────────────────────────

steps = [copy.deepcopy(brep)]

for vline in v_lines[1:-1]:
    brep.split_face_by_curve(vline, face_id=f2.id)
    steps.append(copy.deepcopy(brep))
    print(brep.summary())

assert brep.validate() == [], brep.validate()
print(f"\nFinal: {len(brep.F)} faces, {len(brep.E)} edges, {len(brep.V)} vertices")


# ── Utility: extract trimmed curves for visualization ────────────────────────

def brep_to_loop_wires(brep: BRep):
    """Yield lists of trimmed NURBSCurveTuples, one list per loop."""
    for loop in brep.L.values():
        edgs = []
        for he_id in brep._loop_halfedges(loop.id):
            he = brep.HE[he_id]
            edge = brep.E[he.edge]
            if edge.geom is None:
                continue
            crv = brep.G_CRV[edge.geom]
            t0, t1 = sorted(edge.param)
            edgs.append(trim_curve(crv, t0, t1))
        yield edgs
