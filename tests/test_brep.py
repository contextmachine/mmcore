from mmcore.topo.brep import BRep


def test_1():
    m=BRep()
    v1, v2, edge1, loop1, face, shell = m.MEVVLS((0, 0, 0), (1, 0, 0))

    v3, edge2 = m.MEV(loop1.id, v2.id, (1, 1, 0))  # start with a 2‑vertex wire shell

    v4, edge3 = m.MEV(loop1.id, v3.id, (0, 1, 0))  # start with a 2‑vertex wire shell
    edge4,loop2=m.MEL(loop1.id,v1.id,v4.id)

    print( m.summary())
    print([m.HE[i] for i in m._loop_halfedges(loop1.id)])
    print([m.V[m.HE[i].vert ].point for i in m._loop_halfedges(loop1.id)])
    print([m.HE[i] for i in m._loop_halfedges(loop2.id)])
    print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(loop2.id)])
