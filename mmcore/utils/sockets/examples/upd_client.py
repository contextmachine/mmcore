from mmcore.baseitems import Matchable
from mmcore.services.service import RhinoIronPython, Serviceable

class IronPyCommand(Serviceable):
    __match_args__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def injection(cls):
        return cls.__doc__


class BrepExtruder(IronPyCommand):
    """
# You can use "Inject language reference" in Pycharm to continue using the hints & intelligence.
# It is simple and native.
# Maybe we could also probably use something like dockets ...

import Rhino.Geometry as rg

# Create 4 corner surf
# noinspection PyUnresolvedReferences
mybrep=rg.Brep.CreateFromCornerPoints(self.pt1,self.pt2,self.pt3,self.pt4,
                                        tolerance=self.tolerance)

# Make brep offset
a,b,c = rg.Brep.CreateOffsetBrep(mybrep, 1.0, True, True, 0.1)
offset_brep=list(a)


    """
    __match_args__ = "mybrep", "offset_brep"


import Rhino.Geometry as rgg

rgg.Brep.CreateOffsetBrep()


class Sweep(IronPyCommand):
    """
# You can use "Inject language reference" in Pycharm to continue using the hints & intelligence.
# It is simple and native.
# Maybe we could also probably use something like dockets ...

import Rhino.Geometry as rg
# noinspection PyUnresolvedReferences
rail = rg.NurbsCurve.CreateControlPointCurve(input_msg['points'], input_msg['degree'])
_, pln = rail.FrameAt(0.0)
# noinspection PyUnresolvedReferences
section=input_msg['section']
plnn = rg.Plane(pln.Origin, pln.YAxis,pln.ZAxis)
section_transform = rg.Transform.PlaneToPlane(rg.Plane.WorldXY, plnn)
section.Transform(section_transform)
swp = rg.SweepOneRail()
brp = list(swp.PerformSweep(rail, section))
b
    """
    __match_args__ = "brp", "swp"

    @classmethod
    def injection(cls):
        return cls.__doc__

class Sweep(IronPyCommand):
    """
# You can use "Inject language reference" in Pycharm to continue using the hints & intelligence.
# It is simple and native.
# Maybe we could also probably use something like dockets ...

import Rhino.Geometry as rg
import rhinoscriptsyntax as rs
surf = rs.AddPlanarSrf(self.x)
surface = rs.coercegeometry(surf)

    """
    __match_args__ = "surface", "area"

    @classmethod
    def injection(cls):
        return cls.__doc__

@RhinoIronPython(('localhost', 10081), bytesize=1024 ** 10)
class StopSignal(Matchable):
    """stop"""
    __match_args__ = ()


scr = """

import System.Collections.Generic
pts=System.Collections.Generic.List[rg.Point3d](ctx.points)
rail = rg.NurbsCurve.CreateControlPointCurve(pts, degree=ctx.degree)

_, pln = rail.FrameAt(0.0)
# noinspection PyUnresolvedReferences
section=ctx.section
plnn = rg.Plane(pln.Origin, pln.YAxis,pln.ZAxis)
section_transform = rg.Transform.PlaneToPlane(rg.Plane.WorldXY, plnn)
section.Transform(section_transform)

swp = rg.SweepOneRail()
brp = list(swp.PerformSweep(rail, section))

import json

with open("/tmp/rhsock/brp.cxm", "wb") as fl:
    fl.write(cxd.CxmData(brp))


"""
scr2 = """
import Rhino.Geometry as rg
import rhinoscriptsyntax as rs
surf = rs.AddPlanarSrf(self.x)
surface = rs.coercegeometry(surf)
"""
from mmcore.addons import ModuleResolver
with ModuleResolver() as rsl:
    import rhino3dm
import rhino3dm as rg

a = rg.Point3d(650.03, -1031.64, 378.658)
b = rg.Point3d(832.03, -86.6376, 0)
c = rg.Point3d(-74.4714, 700.864, 0)
sec = rg.NurbsCurve.CreateControlPointCurve(
    [rg.Point3d(-10.609331, 20.60585, 0), rg.Point3d(13.520195, 20.60585, 0), rg.Point3d(13.520195, -9.038997, 0),
     rg.Point3d(7.621866, -9.038997, 0), rg.Point3d(7.621866, -16.775766, 0), rg.Point3d(17.120474, -16.775766, 0),
     rg.Point3d(17.120474, -22.367688, 0), rg.Point3d(-8.847493, -22.367688, 0), rg.Point3d(-8.847493, 17.771588, 0),
     rg.Point3d(8.158078, 17.771588, 0)], degree=1)

inp = dict(points=[a, b, c], section=sec, degree=3)

import cxmdata

model = rg.File3dm()
import time

server_address = ('localhost', 10085)
import socket
import numpy as np

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s = time.time()
crvn = rg.GeometryBase.Decode({
                                  "version": 10000, "archive3dm": 70, "opennurbs": -1879014330,
                                  "data": "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"}
                              )
for i in range(300):
    inp["points"] = list(map(lambda x: rg.Point3d(*x), np.random.random((3, 3)) * 1000))
    inp["section"] = crvn.ToNurbsCurve()

    inp["degree"] = 3
    msg = {}
    msg['input'] = inp
    msg['output'] = ['brp']
    msg['py'] = scr
    sock.sendto(cxmdata.CxmData(msg), server_address)
    aaa, bbb = sock.recvfrom(65507)
    ans = cxmdata.CxmData(aaa).decompress()
    print(ans)

    with open("/tmp/rhsock/brp.cxm", "rb") as fl:
        ans2 = cxmdata.CxmData(fl.read()).decompress()
        print(i, ans2)
        model.Objects.AddBrep(ans2[0])
r = time.time() - s
print(r)
model.Write("/tmp/rhsock/model77.3dm", 7)
