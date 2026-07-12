"""case 12 (USER 2026-07-10): coplanar partial overlap — 2D overlap REGION.

Two planar bilinear patches in z=0 sharing one full edge (the x=11.805 row)
and one corner; their interiors overlap in a 2D region. Both surfaces come
from single-span NURBS (knot spans irrelevant for Bezier extraction).

REPORTED WRONG OUTPUT (two runs, same pair):
  1.1  two branches found on the coinciding edges — but the true answer is
       an OVERLAP (2D region), not curve branches;
  1.2  a branch + a tangent_point singularity + an isolated point.
EXPECTED (semantics TBD with user): a surface-overlap REGION result — the
paper's C2 sub-case #(Delta)=inf, "partially overlap" (Fig. 8 bottom row).
Current output schema has no overlap-region concept: design decision needed
before coding.

RESOLVED 2026-07-12 (ledger L28, approved Option C): the result now carries
result['overlap_regions'] — one SSXOverlapRegion with a closed rim loop of
4 kind='overlap' branches (2 shared-edge + 2 interior curved-preimage rims),
paired sample-synchronized uv loops on both surfaces, a certified interior
witness, and normal_agreement; complete=True with status.reasons == [].

Note RATIONAL=False here (weights all 1); z=0 exactly for every control point.
"""
import numpy as np

RATIONAL = False

S1 = np.array([[[30.45075084, -31.4974516, 0.], [33.62638337, -51.52443823, 0.]],
               [[11.8052607, -31.4974516, 0.], [11.8052607, -51.52443823, 0.]]])

S2 = np.array([[[30.45075084, -31.4974516, 0.], [27.67935951, -47.01458062, 0.]],
               [[11.8052607, -31.4974516, 0.], [11.8052607, -51.52443823, 0.]]])


if __name__ == "__main__":
    from mmcore.numeric.intersection.ssx._bez_ssx5 import bez_ssx
    res = bez_ssx(S1, S2, 1e-3, rational=False)
    print(f"branches={len(res['branches'])} points={len(res['points'])} "
          f"singularities={[g.kind for g in res.get('singularities', [])]}")
    for b in res["branches"]:
        xyz = np.asarray(b.curve[1])
        print(f"  branch kind={getattr(b, 'kind', '?')} n={len(xyz)} "
              f"{np.round(xyz[0], 3).tolist()} -> {np.round(xyz[-1], 3).tolist()}")
