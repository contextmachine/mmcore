scr = """
import Rhino.Geometry as rg
import rhinoscriptsyntax as rs
surf = rs.AddPlanarSrf(self.x)
surface = rs.coercegeometry(surf)\import json

with open("/tmp/rhsock/brp.cxm", "wb") as fl:
    fl.write(cxd.CxmData(brp))

"""

from mmcore.addons import ModuleResolver
with ModuleResolver() as rsl:
    import rhino3dm
import rhino3dm as rg

a = rg.Point3d(650.03, -1031.64, 378.658)
b = rg.Point3d(832.03, -86.6376, 0)
c = rg.Point3d(-74.4714, 700.864, 0)
a = [[-17.910662824207499, 7.6080691642651352], [-8.0043227665706080, -18.306916426512974],
     [29.481268011527391, -21.873198847262255], [26.945244956772349, 0.0], [47.391930835734897, -10.936599423631124],
     [74.810913118549806, 11.033619606638331], [36.517288254747072, 44.356437396774368],
     [42.408615156870596, 12.874659263551926], [14.056604440401230, 24.841417033490281],
     [5.9560299499814278, -8.2972967909543964], [-17.910662824207499, 7.6080691642651352]]
b = [[13.906810790584643, 1.1434817034108491], [13.711129999322676, -14.706662388808182],
     [23.593009958051837, -16.956991488320753], [20.657798089122384, 1.5348432859347749],
     [13.906810790584643, 1.1434817034108491]]

c = [[54.350284322718444, 21.148292487536807], [41.249867706885297, 3.8075231591941883],
     [48.429922368848302, -2.4520116743120308], [67.155760572846958, 13.992385259557480],
     [54.350284322718444, 21.148292487536807]]

from mmcore.addons import ModuleResolver
with ModuleResolver() as rsl:
    import rhino3dm
import rhino3dm as rg


def from_pt(a): return rg.Polyline([rg.Point3d(*o, 0) for o in a]).ToNurbsCurve()


x = [from_pt(a) for pt in (a, b, c)]

inp = {}
import cxmdata

model = rg.File3dm()
import time

server_address = ('localhost', 10085)
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s = time.time()
inp["x"] = x
msg = {}
msg['input'] = inp
msg['output'] = ['surface']
msg['py'] = scr

sock.sendto(cxmdata.CxmData(msg), server_address)
aaa, bbb = sock.recvfrom(65507)
ans = cxmdata.CxmData(aaa).decompress()
print(ans)

with open("/tmp/rhsock/brp.cxm", "rb") as fl:
    ans2 = cxmdata.CxmData(fl.read()).decompress()

    model.Objects.AddBrep(ans2[0])

model.Write("/tmp/rhsock/model77.3dm", 7)
