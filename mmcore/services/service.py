import socket
from functools import wraps
from types import TracebackType
from typing import ContextManager
from typing import Type

from mmcore.baseitems import Matchable


class Serviceable(Matchable):
    __match_args__ = ()
    _injection = None

    def __init__(self, *args, **kwargs):
        self._injection = None
        super().__init__(*args, **kwargs)

    @classmethod
    def injection(cls):
        return cls._injection


from cxmdata import CxmData
import rhino3dm as rg


def serve(serv):
    @wraps(serv)
    def sserv(*ar, **kws):
        def wrp(obj):
            with serv(obj, *ar, **kws) as obbj:
                return obbj

        return wrp

    return sserv


class SocketService(ContextManager):
    _server_address = None

    server_address: tuple[str, int] | str
    bytesize: int | None
    extra_kwargs: dict

    def __init__(self, obj, server_address, bytesize, **kwargs):
        super().__init__()
        self.obj = obj
        self.outputs = obj.__match_args__
        self.server_address = server_address
        self.bytesize = bytesize
        self.__dict__ |= kwargs

    def solve(self, msg) -> dict: ...

    def __enter__(self):
        return self

    def __call__(self, *args, **kwargs):
        compressed = CxmData.compress(self.request(**kwargs))
        cxm_out = self.solve(compressed)
        self.inst = self.obj(*cxm_out)
        return self.inst

    def request(self, **parameters) -> dict:
        return dict(
            input=parameters,
            py=self.obj.injection(),
            output=self.outputs
            )


class RhinoIronPython(SocketService):

    def __exit__(self, __exc_type: Type[BaseException] | None, __exc_value: BaseException | None,
                 __traceback: TracebackType | None) -> bool | None:
        return False

    def __init__(self, *args, **kwargs):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        super().__init__(*args, **kwargs)

    def solve(self, msg):
        try:
            sent = self.sock.sendto(bytes(msg), self.server_address)
            data, server = self.sock.recvfrom(self.bytesize)
            return CxmData(data).decompress()
        except Exception as err:
            print(err)


class DRhinoIronPython(RhinoIronPython):
    """
    >>> @DRhinoIronPython(('localhost', 10081), bytesize=1024 * 1024)
    ... class BrepExtruder(IronPyCommand):
    ...     __doc__ = "..."
    ...     __match_args__ = "mybrep", "offset_brep"
    ...
    ... a=rg.Point3d(650.03, -1031.64, 378.658)
    ... b=rg.Point3d(832.03, -86.6376, 0)
    ... c=rg.Point3d(-74.4714, 700.864, 0)
    ... d=rg.Point3d(-431.472, -793.639, 0)
    ... brp = BrepExtruder(pt1=a,pt2=b,pt3=c, pt4=d, tolerance=0.001)
    >>> brp
    BrepExtruder(mybrep=<rhino3dm._rhino3dm.Brep object at 0x1571f05f0>,
    offset_brep=[<rhino3dm._rhino3dm.Brep object at 0x1572104b0>])
    """
