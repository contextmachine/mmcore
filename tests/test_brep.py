import numpy as np
from mmcore.topo.brep import BRep, box


def assert_valid(brep: BRep, context: str = ""):
    """Assert that brep.validate() returns no errors."""
    errors = brep.validate()
    if errors:
        msg = f"BRep validation failed ({context}):\n" + "\n".join(f"  - {e}" for e in errors)
        raise AssertionError(msg)


# ── MEVVLS ───────────────────────────────────────────────────────────────────

def test_mevvls_basic():
    m = BRep()
    v1, v2, e, l, f, s = m.MEVVLS((0, 0, 0), (1, 0, 0))
    assert_valid(m, "after MEVVLS")
    assert len(m.V) == 2
    assert len(m.E) == 1
    assert len(m.HE) == 2
    assert len(m.L) == 1
    assert len(m.F) == 1
    assert len(m.S) == 1


# ── MEV / KEV ────────────────────────────────────────────────────────────────

def test_mev_basic():
    m = BRep()
    v1, v2, e, l, f, s = m.MEVVLS((0, 0, 0), (1, 0, 0))
    v3, e2 = m.MEV(l.id, v2.id, (1, 1, 0))
    assert_valid(m, "after MEV")
    assert len(m.V) == 3
    assert len(m.E) == 2


def test_mev_kev_roundtrip():
    m = BRep()
    v1, v2, e, l, f, s = m.MEVVLS((0, 0, 0), (1, 0, 0))
    v3, e2 = m.MEV(l.id, v2.id, (1, 1, 0))
    assert_valid(m, "after MEV")
    m.KEV(l.id, v3.id)
    assert_valid(m, "after KEV")
    assert len(m.V) == 2
    assert len(m.E) == 1


# ── MEL / KEL ────────────────────────────────────────────────────────────────

def test_mel_kel_roundtrip():
    m = BRep()
    v1, v2, e, l, f, s = m.MEVVLS((0, 0, 0), (1, 0, 0))
    v3, _ = m.MEV(l.id, v2.id, (1, 1, 0))
    v4, _ = m.MEV(l.id, v3.id, (0, 1, 0))

    e_new, l2 = m.MEL(l.id, v1.id, v4.id)
    assert_valid(m, "after MEL")
    assert len(m.L) == 2

    m.KEL(e_new.id, l2.id)
    assert_valid(m, "after KEL")
    assert len(m.L) == 1


# ── MELF / KELF ──────────────────────────────────────────────────────────────

def test_melf_face_tags():
    """MELF creates a new face and all HEs in loop2 point to it."""
    m = BRep()
    v1, v2, e, l, f, s = m.MEVVLS((0, 0, 0), (1, 0, 0))
    v3, _ = m.MEV(l.id, v2.id, (1, 1, 0))
    v4, _ = m.MEV(l.id, v3.id, (0, 1, 0))

    e_new, l2, f2 = m.MELF(l.id, v4.id, v1.id)
    assert_valid(m, "after MELF")

    # All HEs in loop2 must point to the new face
    for hid in m._loop_halfedges(l2.id):
        assert m.HE[hid].face == f2.id, f"HE {hid} has face={m.HE[hid].face}, expected {f2.id}"


def test_kelf_face_cleanup():
    """KELF removes a face and retags HEs correctly."""
    m = BRep()
    v1, v2, e, l, f, s = m.MEVVLS((0, 0, 0), (1, 0, 0))
    v3, _ = m.MEV(l.id, v2.id, (1, 1, 0))
    v4, _ = m.MEV(l.id, v3.id, (0, 1, 0))

    e_new, l2, f2 = m.MELF(l.id, v4.id, v1.id)
    deleted_face_id = f2.id
    m.KELF(e_new.id, l2.id)
    assert_valid(m, "after KELF")

    # No HE should reference the deleted face
    for he in m.HE.values():
        assert he.face != deleted_face_id, f"HE {he.id} still points to deleted face {deleted_face_id}"


def test_melf_kelf_roundtrip():
    """MELF then KELF returns to the original state (structurally)."""
    m = BRep()
    v1, v2, e, l, f, s = m.MEVVLS((0, 0, 0), (1, 0, 0))
    v3, _ = m.MEV(l.id, v2.id, (1, 1, 0))
    v4, _ = m.MEV(l.id, v3.id, (0, 1, 0))

    e_new, l2, f2 = m.MELF(l.id, v4.id, v1.id)
    assert_valid(m, "after MELF")
    assert len(m.F) == 2
    assert len(m.L) == 2

    m.KELF(e_new.id, l2.id)
    assert_valid(m, "after KELF roundtrip")
    assert len(m.F) == 1
    assert len(m.L) == 1


# ── MVE / KVE ────────────────────────────────────────────────────────────────

def test_mve_single_face():
    """MVE on an edge within a single face."""
    m = BRep()
    v1, v2, e1, l1, f1, s = m.MEVVLS((0, 0, 0), (1, 0, 0))
    v_mid, e2 = m.MVE(e1.id, (0.5, 0, 0))
    assert_valid(m, "after MVE single face")
    assert len(m.V) == 3
    assert len(m.E) == 2


def test_mve_shared_edge():
    """MVE on a shared edge correctly assigns loop/face to both sides."""
    m = BRep()
    v1, v2, e1, l1, f1, s = m.MEVVLS((0, 0, 0), (1, 0, 0))
    v3, _ = m.MEV(l1.id, v2.id, (1, 1, 0))
    v4, _ = m.MEV(l1.id, v3.id, (0, 1, 0))

    # Create two faces sharing e_shared
    e_shared, l2, f2 = m.MELF(l1.id, v4.id, v1.id)
    assert_valid(m, "before MVE on shared edge")

    # Split the shared edge
    v_mid, e_new = m.MVE(e_shared.id, (0.25, 0.75, 0))
    assert_valid(m, "after MVE on shared edge")


def test_mve_kve_roundtrip():
    """MVE then KVE returns to the original state."""
    m = BRep()
    v1, v2, e1, l1, f1, s = m.MEVVLS((0, 0, 0), (1, 0, 0))
    v3, _ = m.MEV(l1.id, v2.id, (1, 1, 0))
    v4, _ = m.MEV(l1.id, v3.id, (0, 1, 0))
    e_shared, l2, f2 = m.MELF(l1.id, v1.id, v4.id)

    v_mid, e_new = m.MVE(e_shared.id, (0.5, 0.5, 0))
    assert_valid(m, "after MVE")
    m.KVE(e_shared.id, v_mid.id)
    assert_valid(m, "after KVE")
    assert len(m.V) == 4


# ── Body ID counter ──────────────────────────────────────────────────────────

def test_body_id_counter():
    """Body uses _B_AUTOID, not _S_AUTOID."""
    m = BRep()
    v1, v2, e, l, f, s = m.MEVVLS((0, 0, 0), (1, 0, 0))
    b = m.new_body(shells=[s.id])
    # Body and Shell should have distinct IDs (no collision)
    assert b.id != s.id, f"Body id {b.id} collides with Shell id {s.id}"


# ── Edge.he maintained ────────────────────────────────────────────────────────

def test_edge_he_maintained():
    """Edge.he is set after creation operators."""
    m = BRep()
    v1, v2, e, l, f, s = m.MEVVLS((0, 0, 0), (1, 0, 0))
    assert e.he is not None, "Edge.he should be set after MEVVLS"
    assert e.he in m.HE, f"Edge.he={e.he} not in HE dict"

    v3, e2 = m.MEV(l.id, v2.id, (1, 1, 0))
    assert e2.he is not None, "Edge.he should be set after MEV"

    v4, _ = m.MEV(l.id, v3.id, (0, 1, 0))
    e_mel, l2 = m.MEL(l.id, v1.id, v4.id)
    assert e_mel.he is not None, "Edge.he should be set after MEL"


# ── KEMH / MEKH ──────────────────────────────────────────────────────────────

def test_mekh_kemh_roundtrip():
    """MEKH bridges a hole into an outer loop; KEMH undoes it."""
    m = BRep()
    # Build outer quad face
    v1, v2, e1, l1, f1, s = m.MEVVLS((0, 0, 0), (4, 0, 0))
    v3, _ = m.MEV(l1.id, v2.id, (4, 4, 0))
    v4, _ = m.MEV(l1.id, v3.id, (0, 4, 0))
    e_close, l_bottom, f_bottom = m.MELF(l1.id, v4.id, v1.id)
    assert_valid(m, "outer quad")

    # Build a hole using MEL (splits l1 into two loops on the same face)
    v5, _ = m.MEV(l1.id, v1.id, (1, 1, 0))
    v6, _ = m.MEV(l1.id, v5.id, (3, 1, 0))
    v7, _ = m.MEV(l1.id, v6.id, (3, 3, 0))
    v8, _ = m.MEV(l1.id, v7.id, (1, 3, 0))
    e_split, l_inner = m.MEL(l1.id, v8.id, v5.id)
    assert_valid(m, "after MEL for hole")

    # Mark the inner loop as a hole and register it with the face
    l_inner_obj = m.L[l_inner.id]
    l_inner_obj.is_outer = False
    m.F[l_inner_obj.face].inners.append(l_inner.id)

    # Now use MEKH to bridge outer and hole
    e_bridge = m.MEKH(l1.id, l_inner.id, v1.id, v5.id)
    assert_valid(m, "after MEKH")

    # Now use KEMH to undo it
    l_hole = m.KEMH(e_bridge.id)
    assert_valid(m, "after KEMH")
    assert not l_hole.is_outer


# ── MZEV / KZEV ─────────────────────────────────────────────────────────────

def test_mzev_basic():
    """MZEV creates a zero-length edge between two loops."""
    m = BRep()
    v1, v2, e1, l1, f1, s = m.MEVVLS((0, 0, 0), (1, 0, 0))
    v3, _ = m.MEV(l1.id, v2.id, (1, 1, 0))
    v4, _ = m.MEV(l1.id, v3.id, (0, 1, 0))
    e_close, l2, f2 = m.MELF(l1.id, v4.id, v1.id)
    assert_valid(m, "before MZEV")

    # MZEV between l1 and l2 at shared vertex v1
    e_zero, v_new = m.MZEV(l1.id, l2.id, v1.id)
    assert_valid(m, "after MZEV")
    assert m.V[v_new.id].point == m.V[v1.id].point


# ── Multiple MELF (box-like construction) ────────────────────────────────────

def test_box_mve_on_shared_edges():
    """Split multiple shared edges of a box and validate."""
    m = box(2, 2, 2)
    assert_valid(m, "box before splits")

    # Find all edges and split each one
    edge_ids = list(m.E.keys())
    split_count = 0
    for e_id in edge_ids[:4]:  # split first 4 edges
        e = m.E[e_id]
        p0 = np.array(m.V[e.v_start].point)
        p1 = np.array(m.V[e.v_end].point)
        mid = tuple(((p0 + p1) / 2).tolist())
        m.MVE(e_id, mid)
        split_count += 1
        assert_valid(m, f"after MVE #{split_count}")


# ── Box ──────────────────────────────────────────────────────────────────────

def test_box_valid():
    """The box() function produces a fully valid BRep."""
    m = box(1, 1, 1)
    assert_valid(m, "box(1,1,1)")
    assert m.topology_check() == 0, f"Euler formula check failed: {m.topology_check()}"


def test_box_counts():
    """Box has correct entity counts: 8V, 12E, 6F."""
    m = box(1, 1, 1)
    assert len(m.V) == 8
    assert len(m.E) == 12
    assert len(m.F) == 6
    assert len(m.HE) == 24


def test_box_face_tags():
    """Every HE in the box model has a face that exists in F."""
    m = box(1, 1, 1)
    for he_id, he in m.HE.items():
        assert he.face in m.F, f"HE {he_id} has face={he.face} not in F"


def test_box_edge_he_fields():
    """Every edge in the box has a valid he field."""
    m = box(1, 1, 1)
    for e_id, e in m.E.items():
        assert e.he is not None, f"Edge {e_id}: he is None"
        assert e.he in m.HE, f"Edge {e_id}: he {e.he} not in HE dict"
        assert m.HE[e.he].edge == e_id, f"Edge {e_id}: he points to wrong edge"


# ── Validate empty ───────────────────────────────────────────────────────────

def test_validate_empty():
    """An empty BRep validates cleanly."""
    m = BRep()
    assert m.validate() == []


# ── Task 1: Geometry dictionaries ─────────────────────────────────────────────

def test_geometry_dictionaries_exist():
    m = BRep()
    assert hasattr(m, 'G_CRV') and isinstance(m.G_CRV, dict)
    assert hasattr(m, 'G_PCRV') and isinstance(m.G_PCRV, dict)
    assert hasattr(m, 'G_SRF') and isinstance(m.G_SRF, dict)

def test_edge_geom_is_optional_int():
    m = BRep()
    v1, v2, e, l, f, s = m.MEVVLS((0, 0, 0), (1, 0, 0))
    assert e.geom is None

def test_face_surf_is_optional_int():
    m = BRep()
    v1, v2, e, l, f, s = m.MEVVLS((0, 0, 0), (1, 0, 0))
    assert f.surf is None

def test_halfedge_pcurve_is_optional_int():
    m = BRep()
    v1, v2, e, l, f, s = m.MEVVLS((0, 0, 0), (1, 0, 0))
    he = m.HE[e.he]
    assert he.pcurve is None


# ── Task 2: Geometry factory helpers ─────────────────────────────────────────

def test_new_curve():
    m = BRep()
    mock_crv = {"type": "line"}
    crv_id = m.new_curve(mock_crv)
    assert crv_id in m.G_CRV
    assert m.G_CRV[crv_id] is mock_crv

def test_new_pcurve():
    m = BRep()
    mock_pcrv = {"type": "line2d"}
    pcrv_id = m.new_pcurve(mock_pcrv)
    assert pcrv_id in m.G_PCRV
    assert m.G_PCRV[pcrv_id] is mock_pcrv

def test_new_surface():
    m = BRep()
    mock_srf = {"type": "plane"}
    srf_id = m.new_surface(mock_srf)
    assert srf_id in m.G_SRF
    assert m.G_SRF[srf_id] is mock_srf

def test_edge_references_curve():
    m = BRep()
    v1, v2, e, l, f, s = m.MEVVLS((0, 0, 0), (1, 0, 0))
    mock_crv = {"type": "line"}
    crv_id = m.new_curve(mock_crv)
    e.geom = crv_id
    assert m.G_CRV[e.geom] is mock_crv


# ── Task 3: validate() geometry reference checks ─────────────────────────────

def test_validate_catches_bad_geom_ref():
    m = BRep()
    v1, v2, e, l, f, s = m.MEVVLS((0, 0, 0), (1, 0, 0))
    e.geom = 99999
    errors = m.validate()
    assert any("Edge" in err and "G_CRV" in err for err in errors)

def test_validate_catches_bad_surf_ref():
    m = BRep()
    v1, v2, e, l, f, s = m.MEVVLS((0, 0, 0), (1, 0, 0))
    f.surf = 99999
    errors = m.validate()
    assert any("Face" in err and "G_SRF" in err for err in errors)

def test_validate_catches_bad_pcurve_ref():
    m = BRep()
    v1, v2, e, l, f, s = m.MEVVLS((0, 0, 0), (1, 0, 0))
    m.HE[e.he].pcurve = 99999
    errors = m.validate()
    assert any("HE" in err and "G_PCRV" in err for err in errors)
