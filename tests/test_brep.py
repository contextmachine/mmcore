from mmcore.topo.brep import BRep


def test_1():
    m = BRep()
    v1, v2, edge1, loop1, face, shell = m.MEVVLS((0, 0, 0), (1, 0, 0))
    print(m.topology_check())
    v3, edge2 = m.MEV(loop1.id, v2.id, (1, 1, 0))  # start with a 2‑vertex wire shell
    print(m.topology_check())
    v4, edge3 = m.MEV(loop1.id, v3.id, (0, 1, 0))  # start with a 2‑vertex wire shell
    print(m.topology_check())
    ids = list(m._loop_halfedges(loop1.id))
    print(ids)
    print([m.HE[i] for i in m._loop_halfedges(loop1.id)])
    print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(loop1.id)])

    edge4, loop2 = m.MEL(loop1.id, v1.id, v4.id)

    print("\nMEL->", m.summary())
    print(m.topology_check())

    print([m.HE[i] for i in m._loop_halfedges(loop1.id)])
    print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(loop1.id)])
    print([m.HE[i] for i in m._loop_halfedges(loop2.id)])
    print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(loop2.id)])
    print(list(m._loop_halfedges(loop1.id)))
    print(list(m._loop_halfedges(loop2.id)))

    m.KEL(edge4.id, loop2.id)

    print("\nKEL->", m.summary())
    print(m.topology_check())
    print(list(m._loop_halfedges(loop1.id)))
    print(m.V)
    print([m.HE[i] for i in m._loop_halfedges(loop1.id)])
    print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(loop1.id)])

def test_2():

    m = BRep()
    v1, v2, edge1, loop1, face, shell = m.MEVVLS((0, 0, 0), (1, 0, 0))
    print(m.topology_check())
    v3, edge2 = m.MEV(loop1.id, v2.id, (1, 1, 0))  # start with a 2‑vertex wire shell
    print(m.topology_check())
    v4, edge3 = m.MEV(loop1.id, v3.id, (0, 1, 0))  # start with a 2‑vertex wire shell
    print(m.topology_check())
    ids = list(m._loop_halfedges(loop1.id))
    print([m.HE[i] for i in m._loop_halfedges(loop1.id)])
    print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(loop1.id)])

    edge4, loop2, f2 = m.MELF(loop1.id, v1.id, v4.id)

    print("\nMELF->", m.summary())
    print(m.topology_check())

    print([m.HE[i] for i in m._loop_halfedges(loop1.id)])
    print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(loop1.id)])
    print([m.HE[i] for i in m._loop_halfedges(loop2.id)])
    print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(loop2.id)])
    print(list(m._loop_halfedges(loop1.id)))
    print(list(m._loop_halfedges(loop2.id)))

    m.KELF(edge4.id, loop2.id)

    print("\nKELF->", m.summary())
    print(m.topology_check())
    print(list(m._loop_halfedges(loop1.id)))
    print(m.V)
    print([m.HE[i] for i in m._loop_halfedges(loop1.id)])
    print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(loop1.id)])


import numpy as np
def test_3():
    m=BRep()
    v1, v2, edge1, loop1, face, shell = m.MEVVLS((0, 0, 0), (1, 0, 0))
    print(m.topology_check())
    v3, edge2 = m.MEV(loop1.id, v2.id, (1, 1, 0))  # start with a 2‑vertex wire shell
    print(m.topology_check())
    v4, edge3 = m.MEV(loop1.id, v3.id, (0, 1, 0))  # start with a 2‑vertex wire shell
    print(m.topology_check())
    ids=list(m._loop_halfedges(loop1.id))
    print([m.HE[i] for i in m._loop_halfedges(loop1.id)])
    print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(loop1.id)])

    edge4,loop2,f2=m.MELF(loop1.id,v1.id,v4.id)

    print("\nMELF->", m.summary())

    print(m.topology_check())
    print(list(m._loop_halfedges(loop1.id)))
    print(list(m._loop_halfedges(loop2.id)))

    v5, edge5 = m.MVE(
        edge4.id,tuple( ((np.array(v1.point) - np.array(v4.point)) / 2 + np.array(v4.point)).tolist())
    )
    print('\nMVE->',m.summary())
    print(m.topology_check())
    print(m.V)

    print(m.E)
    print(list(m._loop_halfedges(loop1.id)))
    print(list(m._loop_halfedges(loop2.id)))
    m.KVE(edge4.id,v5.id)
    print('\nKVE->',m.summary())

    print(m.E)
    print(m.V)
    print(list(m._loop_halfedges(loop1.id)))
    print(list(m._loop_halfedges(loop2.id)))
  
