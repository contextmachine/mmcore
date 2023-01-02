import os
import sys
import socket
from typing import Protocol

import dill
import httpx

from mmcore.cxmdata import CxmData
from mmcore.services.service import Service

class Injection(Protocol):
    """
        """

    def __init__(self, *args, **kwargs):
        ...

    @property
    def injection(self) -> str:
        return self.__class__.__doc__

    def __call__(self, *args, **results):
        ...


class Inputs:
    def __init__(self, srv):
        self.srv = srv

    def __set__(self, instance, owner):
        ...


class RhinoIronPython(Service):
    def __init__(self, address=('localhost', 10000), bytesize=4096 * 2, **kwargs):
        """

        @param address:
        @param bytesize:
        @param kwargs:
        """
        super().__init__()
        self.server_address = address
        self.bytesize = bytesize
        self.extra_kwargs = kwargs
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def solve(self, msg):
        try:

            # Send data
            # print(sys.stderr, 'sending "%s"' % message, flush=True)
            sent = self.sock.sendto(bytes(msg), self.server_address)
            # Receive response
            print(sys.stderr, 'waiting to receive', flush=True)
            data, server = self.sock.recvfrom(self.bytesize)
            print(data)
            return CxmData(data).decompress()


        except Exception as err:
            print(err)
import Rhino.Geometry as rg
rg.Brep.CreateFromLoftRebuild()


class RhinoCompute(Service):

    def __init__(self, address, apikey=os.getenv("RHIBOCOMPUTE"), **kwargs):
        super().__init__(address, *kwargs)
        self.apikey=apikey
        self.extra_kwargs = kwargs
        self.client = httpx.AsyncClient(headers="http://{}/")


    _headers ={
        "User-Agent": "compute.rhino3d.py/1.2.0",
        "Accept": "application/json",
        "Content-Type": ("application/json","application/binary","application/text"),

        }


    def solve(self, msg) -> dict:
        from compute_rhino3d import Util
        Util.url = self.server_address
        Util.apiKey = self.apikey
        dill.source.likely_import()
        self._method = meth
        self._lines = dill.source.getsourcelines(self._method)[0]
        # print(self._lines)
        self.inputs = set(extract_inputs(self._method, True))
        self.out = set(extract_out(self._lines))
        self.intern = set(self._method.__code__.co_varnames) - self.inputs.union(self.out)
        inp = kwargs

        out = list(self.out)
        script = ""

        for line in self._lines[2:-1]:
            script += line.replace("        ", "")
        print(script)
        print(inp)
        print(out)
        return Util.PythonEvaluate(script, inp, out)


    @property
    def headers(self):
        self._headers["RhinoComputeKey"]= self.apikey
        return self._headers

    @headers.setter
    def headers(self, value):
        self._headers = value

class A:
    def __init__(self, pln):
        self.polyline = pln

    @ComputeBinder
    def rhcmp(self, polyline: rhino3dm.Polyline = None, x=None, y=None, z=None):
        import Rhino.Geometry as rg

        rail = rg.NurbsCurve.CreateControlPointCurve \
            ([rg.Point3d(xx, yy, zz) for xx, yy, zz in zip(eval(x), eval(y), eval(z))], 2)
        _, pln = rail.FrameAt(0.0)

        plnn = rg.Plane(pln.Origin, pln.YAxis,
                        pln.ZAxis)
        polyline.Transform(rg.Transform.PlaneToPlane(rg.Plane.WorldXY, plnn))
        swp = rg.SweepOneRail()
        ans = swp.PerformSweep(rail, polyline)
        return ans

    def sweep(self, rail):
        x, y, z = rail
        return self.rhcmp(polyline=self.polyline, x=f'{x}', y=f'{y}', z=f'{z}')["ans"][0]

result = Util.PythonEvaluate("import Rhino.Geometry as rg\nres=rg.NurbsCurve.CreateControlPointCurve([rg.Point3d(xx,yy,zz) for xx,yy,zz in zip(eval(x),eval(y),eval(z))], 3)",
                                 {
                                     "x": '[0.0,1.0,2.0,3.0,4.0,5.0,6.0]',
                                     "y": '[0.0,1.0,2.0,3.0,4.0,5.0,6.0]',
                                     "z": '[0.0,1.0,2.0,3.0,4.0,5.0,6.0]'
                                 }, ["res"])"""