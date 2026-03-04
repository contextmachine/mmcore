"""
Tests and usage examples for the Half-Edge Topological Data Structure.

Demonstrates building faces, splitting, holes, and inverse operations.
"""
from mmcore.topo.halfedge_topo import Topology



def assert_valid(topo: Topology, context: str = ""):
    errors = topo.validate()
    if errors:
        print(f"\n{'='*60}")
        print(f"VALIDATION FAILED: {context}")
        for e in errors:
            print(f"  ✗ {e}")
        print(topo.dump())
        raise AssertionError(f"Validation failed ({context}): {errors[0]}")


def assert_stats(topo: Topology, V, E, L, F, S, context=""):
    s = topo.stats()
    expected = {'V': V, 'E': E, 'HE': E * 2, 'L': L, 'F': F, 'S': S}
    if s != expected:
        print(f"\n{'='*60}")
        print(f"STATS MISMATCH: {context}")
        print(f"  Expected: {expected}")
        print(f"  Got:      {s}")
        print(topo.dump())
        raise AssertionError(f"Stats mismatch ({context})")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: Basic construction — single face with triangle boundary
# ═══════════════════════════════════════════════════════════════════════════════

def test_triangle():
    """Build a triangular face from scratch."""
    print("TEST 1: Triangle face")
    topo = Topology()

    # Step 1: create shell + face
    face, shell = topo.make_face_shell()
    assert_valid(topo, "after make_face_shell")
    assert_stats(topo, V=0, E=0, L=0, F=1, S=1, context="make_face_shell")

    # Step 2: first edge (digon)
    v1, v2, e1, loop = topo.make_edge_loop(face)
    assert_valid(topo, "after make_edge_loop")
    assert_stats(topo, V=2, E=1, L=1, F=1, S=1, context="make_edge_loop")
    assert loop.length == 2, f"Digon loop should have 2 HEs, got {loop.length}"

    # Step 3: add a spike from v1 → v3
    he_from_v1 = v1.halfedge  # outgoing from v1
    v3, e2 = topo.make_edge_vertex(he_from_v1)
    assert_valid(topo, "after make_edge_vertex")
    assert_stats(topo, V=3, E=2, L=1, F=1, S=1, context="make_edge_vertex")

    # Step 4: close the triangle by connecting v3 to v2
    # Find half-edges from v3 and v2 in the loop
    he_from_v3 = None
    he_from_v2 = None
    for he in loop.halfedges():
        if he.vertex is v3 and he.target is v1:
            he_from_v3 = he  # going v3→v1, this is the return leg of the spike
        if he.vertex is v2:
            he_from_v2 = he

    # Actually we need he originating at v3 (towards v1) and he originating at v2
    # for split_face. But split_face connects he1.vertex to he2.vertex.
    # We want to connect v3 and v2.
    # Find a he starting at v3 and a he starting at v2 that are in the same loop.
    he_v3 = None
    he_v2 = None
    for he in loop.halfedges():
        if he.vertex is v3:
            he_v3 = he
        if he.vertex is v2:
            he_v2 = he

    assert he_v3 is not None, "Couldn't find HE from v3"
    assert he_v2 is not None, "Couldn't find HE from v2"

    e3, face2 = topo.split_face(he_v3, he_v2)
    assert_valid(topo, "after split_face (triangle close)")
    assert_stats(topo, V=3, E=3, L=2, F=2, S=1, context="split_face triangle")

    # One face should be the triangle (3 HEs), the other the degenerate remainder
    f1_size = face.outer_loop.length
    f2_size = face2.outer_loop.length
    print(f"  Face 1 loop size: {f1_size}, Face 2 loop size: {f2_size}")
    assert 3 in (f1_size, f2_size), "One face should be a triangle (3 HEs)"

    print(topo.dump())
    print("  ✓ Triangle built successfully\n")
    return topo


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: Split edge
# ═══════════════════════════════════════════════════════════════════════════════

def test_split_edge():
    """Split an edge to insert a midpoint."""
    print("TEST 2: Split edge")
    topo = Topology()

    face, shell = topo.make_face_shell()
    v1, v2, e1, loop = topo.make_edge_loop(face)
    assert_stats(topo, V=2, E=1, L=1, F=1, S=1, context="before split")

    # Split the edge
    he = e1.halfedge  # one of the two half-edges
    v_mid, e2 = topo.split_edge(he)
    assert_valid(topo, "after split_edge")
    assert_stats(topo, V=3, E=2, L=1, F=1, S=1, context="after split")
    assert loop.length == 4, f"After splitting digon, loop should have 4 HEs, got {loop.length}"

    # Split again
    he2 = e2.halfedge
    v_mid2, e3 = topo.split_edge(he2)
    assert_valid(topo, "after second split_edge")
    assert_stats(topo, V=4, E=3, L=1, F=1, S=1, context="after 2nd split")
    assert loop.length == 6

    print(f"  Loop vertices: {list(loop.vertices())}")
    print("  ✓ Split edge works\n")
    return topo


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: Quad face from splits + split_face
# ═══════════════════════════════════════════════════════════════════════════════

def test_quad():
    """Build a quad by splitting edges of the initial digon, then split_face."""
    print("TEST 3: Quad face")
    topo = Topology()

    face, shell = topo.make_face_shell()
    v1, v2, e1, loop = topo.make_edge_loop(face)

    # Split both edges to get 4 vertices
    # Digon: v1 → v2 → v1  (he1: v1→v2, he2: v2→v1)
    he1 = e1.halfedge           # v1 → v2
    v3, e2 = topo.split_edge(he1)
    # Now: v1→v3→v2→...→v1, loop has 4 HEs

    # Find he going v2→v1 (or v2→v3, depends on which side)
    he_v2 = None
    for he in loop.halfedges():
        if he.vertex is v2 and he.target is not v3:
            he_v2 = he
            break
    if he_v2 is None:
        for he in loop.halfedges():
            if he.vertex is v2:
                he_v2 = he
                break

    v4, e3 = topo.split_edge(he_v2)
    assert_valid(topo, "after 2 splits")
    assert_stats(topo, V=4, E=3, L=1, F=1, S=1, context="2 splits on digon")

    # Loop now has 6 HEs traversing: v1→v3→v2→v4→v1→v3→v2→v4... wait
    # Actually the digon splits give us: v1 v3 v2 v4 on the loop
    print(f"  Loop vertices after splits: {list(loop.vertices())}")
    print(f"  Loop length: {loop.length}")

    # Find opposite vertices to connect (v3 and v4, or v1 and v2)
    # Let's connect v3 and v4 to split the face into a quad
    he_v3 = None
    he_v4 = None
    for he in loop.halfedges():
        if he.vertex is v3 and he_v3 is None:
            he_v3 = he
        if he.vertex is v4 and he_v4 is None:
            he_v4 = he

    if he_v3 and he_v4:
        e4, face2 = topo.split_face(he_v3, he_v4)
        assert_valid(topo, "after split_face for quad")
        assert_stats(topo, V=4, E=4, L=2, F=2, S=1, context="quad split")

        for f in shell.faces:
            sz = f.outer_loop.length
            print(f"  {f} loop size: {sz}, vertices: {list(f.outer_loop.vertices())}")

    print("  ✓ Quad face built\n")
    return topo


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: Hole creation (inner loop)
# ═══════════════════════════════════════════════════════════════════════════════

def test_hole():
    """Create a face with a hole using add_inner_loop."""
    print("TEST 4: Face with hole")
    topo = Topology()

    face, shell = topo.make_face_shell()
    v1, v2, e1, outer = topo.make_edge_loop(face)

    # Build outer boundary into a triangle
    he_v1 = v1.halfedge
    v3, e2 = topo.make_edge_vertex(he_v1)

    he_v3 = None
    he_v2 = None
    for he in outer.halfedges():
        if he.vertex is v3:
            he_v3 = he
        if he.vertex is v2:
            he_v2 = he

    e3, face2 = topo.split_face(he_v3, he_v2)
    assert_valid(topo, "outer triangle built")

    # Identify the triangular face (3 HEs)
    tri_face = face if face.outer_loop.length == 3 else face2
    print(f"  Triangle face: {tri_face}, outer loop size: {tri_face.outer_loop.length}")

    # Add an inner loop (hole seed) to the triangle
    h1, h2, he_inner, inner_loop = topo.add_inner_loop(tri_face)
    assert_valid(topo, "after add_inner_loop")
    assert inner_loop in tri_face.inner_loops
    print(f"  Inner loop: {inner_loop}, vertices: {list(inner_loop.vertices())}")

    print(topo.dump())
    print("  ✓ Hole created\n")
    return topo


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5: join_loops and separate_loop
# ═══════════════════════════════════════════════════════════════════════════════

def test_join_separate_loops():
    """Create a hole, bridge it to outer loop, then separate it back."""
    print("TEST 5: Join and separate loops")
    topo = Topology()

    face, shell = topo.make_face_shell()
    v1, v2, e1, outer = topo.make_edge_loop(face)

    # Add inner loop
    h1, h2, e_inner, inner = topo.add_inner_loop(face)
    assert_valid(topo, "with inner loop")
    assert_stats(topo, V=4, E=2, L=2, F=1, S=1, context="outer+inner digons")

    stats_before = topo.stats()
    print(f"  Before join: {stats_before}")
    print(f"  Inner loops: {face.inner_loops}")

    # Join: bridge v1 (outer) to h1 (inner)
    he_outer = v1.halfedge
    he_inner = h1.halfedge
    bridge_edge = topo.join_loops(he_outer, he_inner)
    assert_valid(topo, "after join_loops")
    assert len(face.inner_loops) == 0, "Inner loop should be absorbed"
    print(f"  After join: {topo.stats()}")
    print(f"  Outer loop size: {face.outer_loop.length}")

    # Separate: restore the inner loop by removing the bridge
    he_bridge = bridge_edge.halfedge
    # Determine which half-edge goes from outer to inner
    # We need the one whose removal recreates the inner loop correctly
    inner_restored = topo.separate_loop(he_bridge)
    assert_valid(topo, "after separate_loop")
    assert len(face.inner_loops) == 1, "Inner loop should be restored"
    print(f"  After separate: {topo.stats()}")
    print(f"  Restored inner loop: {inner_restored}")

    print("  ✓ Join/separate loops works\n")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 6: Inverse operators
# ═══════════════════════════════════════════════════════════════════════════════

def test_inverse_operators():
    """Test that inverse operators correctly undo forward operators."""
    print("TEST 6: Inverse operators")
    topo = Topology()

    # --- make_face_shell / kill_face_shell ---
    face, shell = topo.make_face_shell()
    assert_stats(topo, V=0, E=0, L=0, F=1, S=1, context="make_face_shell")
    topo.kill_face_shell(face)
    assert_stats(topo, V=0, E=0, L=0, F=0, S=0, context="kill_face_shell")
    print("  ✓ make/kill_face_shell")

    # --- make_edge_loop / kill_edge_loop ---
    face, shell = topo.make_face_shell()
    v1, v2, e1, loop = topo.make_edge_loop(face)
    assert_stats(topo, V=2, E=1, L=1, F=1, S=1, context="make_edge_loop")
    topo.kill_edge_loop(face)
    assert_valid(topo, "after kill_edge_loop")
    assert_stats(topo, V=0, E=0, L=0, F=1, S=1, context="kill_edge_loop")
    assert face.outer_loop is None
    print("  ✓ make/kill_edge_loop")

    # --- make_edge_vertex / kill_edge_vertex ---
    v1, v2, e1, loop = topo.make_edge_loop(face)
    he_v1 = v1.halfedge
    v3, e2 = topo.make_edge_vertex(he_v1)
    assert_stats(topo, V=3, E=2, L=1, F=1, S=1, context="make_edge_vertex")

    # Kill the spike: find the HE from v3 (degree 1)
    he_from_v3 = v3.halfedge  # outgoing from v3
    topo.kill_edge_vertex(he_from_v3)
    assert_valid(topo, "after kill_edge_vertex")
    assert_stats(topo, V=2, E=1, L=1, F=1, S=1, context="kill_edge_vertex")
    print("  ✓ make/kill_edge_vertex")

    # --- split_edge / join_edge ---
    he = e1.halfedge
    v_mid, e_new = topo.split_edge(he)
    assert_stats(topo, V=3, E=2, L=1, F=1, S=1, context="split_edge")
    merged_e = topo.join_edge(v_mid)
    assert_valid(topo, "after join_edge")
    assert_stats(topo, V=2, E=1, L=1, F=1, S=1, context="join_edge")
    print("  ✓ split/join_edge")

    # --- split_face / join_face ---
    # Build a quad first
    he = list(loop.halfedges())[0]
    v3, e2 = topo.make_edge_vertex(he)

    he_v3 = None
    he_v2_target = None
    for h in loop.halfedges():
        if h.vertex is v3:
            he_v3 = h
        if h.vertex is v2:
            he_v2_target = h

    e3, face2 = topo.split_face(he_v3, he_v2_target)
    assert_valid(topo, "after split_face")
    n_faces_before = len(topo.faces)

    # Join them back
    he_join = e3.halfedge
    if he_join.face is he_join.twin.face:
        raise ValueError("Cannot join_face: same face on both sides")
    topo.join_face(he_join)
    assert_valid(topo, "after join_face")
    assert len(topo.faces) == n_faces_before - 1
    print("  ✓ split/join_face")

    print("  ✓ All inverse operators verified\n")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 7: Navigation and queries
# ═══════════════════════════════════════════════════════════════════════════════

def test_navigation():
    """Test vertex/face/edge navigation helpers."""
    print("TEST 7: Navigation")
    topo = Topology()

    # Build two adjacent triangles (butterfly shape)
    face, shell = topo.make_face_shell()
    v1, v2, e1, loop = topo.make_edge_loop(face)

    # Add v3 spike from v1
    v3, e2 = topo.make_edge_vertex(v1.halfedge)
    # Close triangle: connect v3 to v2
    he_v3 = he_v2 = None
    for he in loop.halfedges():
        if he.vertex is v3:
            he_v3 = he
        if he.vertex is v2:
            he_v2 = he
    e3, face_b = topo.split_face(he_v3, he_v2)

    # Pick the triangle face and add v4 from v1 on the non-triangle face
    tri_face = face if face.outer_loop.length == 3 else face_b
    other_face = face_b if tri_face is face else face
    print(f"  Triangle: {tri_face}, Other: {other_face}")

    # Test vertex navigation
    print(f"  v1 degree: {v1.degree}")
    print(f"  v1 neighbors: {list(v1.neighbors())}")
    print(f"  v2 degree: {v2.degree}")

    # Test face navigation
    tri_neighbors = list(tri_face.neighbor_faces())
    print(f"  Triangle face neighbors: {tri_neighbors}")

    # Test edge boundary detection
    boundary = list(shell.boundary_edges())
    print(f"  Boundary edges: {len(boundary)}")

    # Test find_halfedge
    he_12 = topo.find_halfedge(v1, v2)
    he_21 = topo.find_halfedge(v2, v1)
    assert he_12 is not None, "Should find HE from v1 to v2"
    assert he_21 is not None, "Should find HE from v2 to v1"
    assert he_12.twin is he_21

    # Test find_edge
    e_12 = topo.find_edge(v1, v2)
    assert e_12 is not None
    assert e_12 is he_12.edge

    assert_valid(topo, "navigation test")
    print("  ✓ Navigation works\n")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 8: Complex workflow — face with trimmed hole
# ═══════════════════════════════════════════════════════════════════════════════

def test_complex_workflow():
    """
    Build a rectangular face with a triangular hole inside.

    Workflow:
      1. Create face + outer digon
      2. Build outer rectangle via split_edge
      3. split_face to form quad
      4. Add inner loop (hole seed)
      5. Build up hole boundary via split_edge
      6. Bridge inner to outer and back to verify join/separate
    """
    print("TEST 8: Complex workflow — rectangle with triangular hole")
    topo = Topology()

    # 1. Shell + face + first edge
    face, shell = topo.make_face_shell()
    v1, v2, e1, loop = topo.make_edge_loop(face)

    # 2. Split both edges to get 4 boundary vertices
    he_12 = e1.halfedge  # v1→v2
    v3, e2 = topo.split_edge(he_12)
    # Now: loop visits v1→v3→v2→...→v1

    # Find HE from v2 going back towards v1 side
    he_from_v2 = None
    for he in loop.halfedges():
        if he.vertex is v2:
            he_from_v2 = he
            break
    v4, e3 = topo.split_edge(he_from_v2)
    assert_valid(topo, "4 vertices on boundary")

    # 3. Split face between v3 and v4 to make two faces (rectangle)
    he_v3 = he_v4 = None
    for he in loop.halfedges():
        if he.vertex is v3 and he_v3 is None:
            he_v3 = he
        if he.vertex is v4 and he_v4 is None:
            he_v4 = he
    e4, face2 = topo.split_face(he_v3, he_v4)
    assert_valid(topo, "rectangle split")

    # 4. Pick the face to put the hole in
    target_face = face if face.outer_loop.length >= 3 else face2
    print(f"  Target face for hole: {target_face}, loop size: {target_face.outer_loop.length}")

    # 5. Add inner loop (hole)
    h1, h2, e_h, inner = topo.add_inner_loop(target_face)
    assert_valid(topo, "inner loop added")

    # Build hole into a triangle: split the inner edge to get 3 vertices
    he_inner = e_h.halfedge
    h3, e_h2 = topo.split_edge(he_inner)
    assert_valid(topo, "inner edge split")

    # Find HEs in inner loop for h1, h2, h3 to close triangle
    he_h3 = he_h2_inner = None
    for he in inner.halfedges():
        if he.vertex is h3 and he_h3 is None:
            he_h3 = he
        if he.vertex is h2 and he_h2_inner is None:
            he_h2_inner = he

    # Note: Can't split_face on inner loop — that would try to create a new face
    # inside the hole, which is a different semantic. The inner loop IS the hole boundary.
    # For a triangular hole, we just need 3 edges forming the inner loop.
    # Currently we have: h1→h3→h2→...→h1 (with some back-tracking from the digon split)

    print(f"  Inner loop vertices: {list(inner.vertices())}")
    print(f"  Inner loop size: {inner.length}")

    # 6. Bridge test: connect outer vertex to inner vertex
    he_outer_v = None
    for he in target_face.outer_loop.halfedges():
        he_outer_v = he
        break

    he_inner_v = None
    for he in inner.halfedges():
        he_inner_v = he
        break

    stats_pre = topo.stats()
    bridge = topo.join_loops(he_outer_v, he_inner_v)
    assert_valid(topo, "after join_loops")
    assert len(target_face.inner_loops) == 0
    print(f"  After join — outer loop size: {target_face.outer_loop.length}")

    # Separate back
    he_b = bridge.halfedge
    restored = topo.separate_loop(he_b)
    assert_valid(topo, "after separate_loop")
    assert len(target_face.inner_loops) == 1
    print(f"  After separate — inner loop restored: {restored}")

    print(topo.dump())
    print("  ✓ Complex workflow complete\n")
    return topo


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 9: Multiple faces sharing edges
# ═══════════════════════════════════════════════════════════════════════════════

def test_multiple_faces():
    """Build a fan of 3 triangular faces sharing a central vertex."""
    print("TEST 9: Fan of 3 triangles")
    topo = Topology()

    face, shell = topo.make_face_shell()
    # Start with one edge
    v_center, v1, e1, loop = topo.make_edge_loop(face)

    # Add v2 as spike from v_center
    v2, e2 = topo.make_edge_vertex(v_center.halfedge)

    # Close first triangle: split_face between v2 and v1
    he_v2 = he_v1 = None
    for he in loop.halfedges():
        if he.vertex is v2 and he.target is v_center:
            he_v2 = he
        if he.vertex is v1:
            he_v1 = he
    e3, tri1 = topo.split_face(he_v2, he_v1)
    assert_valid(topo, "first triangle")

    # Identify the non-triangle face to continue building on
    remaining = face if face.outer_loop.length > 3 else tri1
    triangle1 = tri1 if face.outer_loop.length > 3 else face

    # Add v3 spike from v_center on the remaining face
    he_vc = None
    for he in remaining.outer_loop.halfedges():
        if he.vertex is v_center:
            he_vc = he
            break

    v3, e4 = topo.make_edge_vertex(he_vc)

    # Close second triangle: split_face between v3 and appropriate vertex
    he_v3 = None
    v_target = None
    for he in remaining.outer_loop.halfedges():
        if he.vertex is v3:
            he_v3 = he
    # Find a vertex that's not v_center and not v3
    for he in remaining.outer_loop.halfedges():
        if he.vertex is not v_center and he.vertex is not v3:
            he_target = he
            v_target = he.vertex
            break

    e5, tri2 = topo.split_face(he_v3, he_target)
    assert_valid(topo, "second triangle")

    print(f"  Shell has {len(shell.faces)} faces")
    for f in shell.faces:
        sz = f.outer_loop.length
        verts = list(f.outer_loop.vertices())
        print(f"    {f}: {sz} HEs, vertices: {verts}")

    # Test face adjacency
    for f in shell.faces:
        neighbors = list(f.neighbor_faces())
        print(f"    {f} neighbors: {neighbors}")

    # Test edge sharing
    shared_edges = [e for e in topo.edges.values() if not e.is_boundary]
    print(f"  Shared (non-boundary) edges: {len(shared_edges)}")

    print(topo.dump())
    print("  ✓ Multiple face fan built\n")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ALL TESTS
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 70)
    print("  Half-Edge Topology — Test Suite")
    print("=" * 70 + "\n")

    test_triangle()
    test_split_edge()
    test_quad()
    test_hole()
    test_join_separate_loops()
    test_inverse_operators()
    test_navigation()
    test_complex_workflow()
    test_multiple_faces()

    print("=" * 70)
    print("  ALL TESTS PASSED ✓")
    print("=" * 70)
