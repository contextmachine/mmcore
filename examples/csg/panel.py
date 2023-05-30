import ujson
from mmcore.geom.csg import CSG
from mmcore.base.geom import utils, MeshData
vn=CSG.cylinder(dict(start=[0,0,0], end=[600,0,0], radius=10))
vn=CSG.cylinder(**dict(start=[0,0,0], end=[600,0,0], radius=10))
vn2=CSG.cylinder(**dict(start=[0,0,0], end=[600,0,0], radius=5))
vn3=vn-vn2
from mmcore.base import A, AGroup
grp=AGroup(name="Union")
for p in vn3.toPolygons():
    grp.add(p.mesh())
grp.dump("testgrp.json")
from mmcore.base.sharedstate import serve
serve.start()
cb2=CSG.cube(dict(center=[300,10,0.5], radius=[350,5,0.5]))
from mmcore.base import A, AGroup
grp2=AGroup(name="Union")
for p in cb2.toPolygons():
    grp2.add(p.mesh())
grp2.dump("testgrp.json")

cb2=CSG.cube(dict(center=[300,5,0.5], radius=[350,5,15]))
from mmcore.base import A, AGroup
grp2=AGroup(name="Union")
for ch in grp2.children:
    ch.dispose()
for p in cb2.toPolygons():
    grp2.add(p.mesh())
grp2.dump("testgrp.json")
vn4 = vn3-cb2

grp2.dump("testgrp.json")
grp2.dump("tube.json")
with open("/Users/andrewastakhov/PycharmProjects/mmcore/tests/panel.json", "r") as f:
    geom = ujson.load(f)
    msh = MeshData.from_buffer_geometry(geom["geometries"][0])

