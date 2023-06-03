from __future__ import absolute_import, annotations

import abc
import functools
import importlib
import os
import socket
import subprocess
import typing
from enum import Enum
from functools import wraps
from typing import ContextManager

import dill
from cxmdata import CxmData

from mmcore.addons import ModuleResolver

with ModuleResolver() as rsl:
    pass
import rhino3dm as rg


def serve(serv):
    @wraps(serv)
    def sserv(*ar, **kws):
        def wrp(obj):
            with serv(obj, *ar, **kws) as obbj:
                obbj.s

        return wrp

    return sserv


class SocketService(ContextManager):
    _server_address = None

    server_address: tuple[str, int] | str
    bytesize: int | None
    extra_kwargs: dict = dict()

    def __init__(self, obj, server_address, bytesize, **kwargs):
        super().__init__()
        self.obj = obj
        self.outputs = obj.__match_args__
        self.server_address = server_address
        self.bytesize = bytesize
        self.extra_kwargs |= kwargs

    @abc.abstractmethod
    def solve(self, msg) -> dict: ...

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __call__(self, **kwargs):
        print(kwargs)
        compressed = CxmData.compress(self.request(**kwargs))
        cxm_out = self.solve(compressed)
        return self.obj(*cxm_out)

    def request(self, **parameters) -> dict:
        return dict(
            input=parameters,
            py=self.obj.__doc__,
            output=self.outputs
        )


class RhinoStartsEnum(str,Enum):
    RHINO_CODE = f"-_ScriptEditor Run {os.getcwd()}/bin/rhpyc_classic.py"
    IRONPYTHON = f"-_RunPythonScript {os.getcwd()}/bin/app.py"


class RhinoRunner(ContextManager):
    """
    Use only on host with rhinoceros installation!
    """
    def __init__(self, path, command: typing.Union[RhinoStartsEnum, str] = RhinoStartsEnum.RHINO_CODE, **kwargs):
        if os.getenv("IS_RHINO_RUNNING") == 'True':
            subprocess.Popen(['sudo', 'kill', os.getenv("RHINO_PID")])

        super().__init__()
        self.path = path
        self.command = command
        self.kwargs = kwargs
        self.conn=None

    def __enter__(self):
        self.proc = subprocess.Popen(
            [self.path, "-nosplash", "-runscript", self.command],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE, **self.kwargs)

        os.environ["IS_RHINO_RUNNING"] = 'True'
        os.environ["RHINO_PID"] = str(self.proc.pid)
        self.proc.communicate()
        if self.command == RhinoStartsEnum.RHINO_CODE:
            import rpyc
            self.conn = rpyc.connect("localhost", 7778)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.kill()
        os.environ["IS_RHINO_RUNNING"] = 'False'

    def __call__(self, *args, **kwargs):
        return self.proc.communicate(*args, **kwargs)

    def kill(self):

        self.proc.kill()
        try:
            del os.environ["IS_RHINO_RUNNING"]
        except:
            pass


class RhinoIronPython(SocketService):
    server_address: tuple[str, int] | str
    bytesize: int | None
    extra_kwargs: dict

    def __init__(self, obj, server_address, bytesize, **kwargs):
        super().__init__(obj, server_address, bytesize, **kwargs)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def __call__(self, **kwargs):
        return super().__call__(**kwargs)

    def solve(self, msg):
        try:
            sent = self.sock.sendto(msg, self.server_address)
            data, server = self.sock.recvfrom(self.bytesize)
            return dict(zip(self.obj.__match_args__, CxmData(data).decompress()))
        except Exception as err:
            #print(err)


import cxmdata


def dgram(msg, sock, server_address, bufsize=65507):
    sock.sendto(cxmdata.CxmData(msg), server_address)
    aaa, bbb = sock.recvfrom(bufsize)
    return cxmdata.CxmData(aaa).decompress(), bbb


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


import yaml
from rpyc.cli.rpyc_classic import ClassicServer

import os

import pprint


class RpycService(ClassicServer):
    def __init_subclass__(cls, configs=None, **kwargs):
        if configs is not None:
            os.environ["RPYC_CONFIGS"] = configs
        if os.getenv("RPYC_CONFIGS").startswith("http:") or os.getenv("RPYC_CONFIGS").startswith("https:"):
            import requests

            data = yaml.unsafe_load(requests.get(os.getenv("RPYC_CONFIGS")).text)
        else:
            with open(os.getenv("RPYC_CONFIGS")) as f:
                data = yaml.unsafe_load(f)
        print(data)
        if list(data.keys())[0] == "service":

            configs = data["service"].get("configs")
            attrs = data["service"].get("attributes")
            real_attrs = {}

            pattrs = "attributes: {}\n\t\n"
            pconfigs = "configs: {}\n\t\n"
            cls.host = attrs.get("host") if attrs.get("host") is not None else '0.0.0.0'
            cls.port = attrs.get("port") if attrs.get("port") is not None else 7777

            if attrs:

                for k, v in attrs.items():
                    if hasattr(cls, k):
                        setattr(cls, k, v)
                        real_attrs[k] = v
                    else:
                        #print(f"miss {k}")
                pattrs.format(pprint.pformat(real_attrs, indent=4))
            if configs:
                pconfigs.format(pprint.pformat(configs, indent=4))


# RhService.ssl_certfile = f"{os.getenv('HOME')}/ssl/ca-certificates/certificate_full_chain.pem"
# RhService.ssl_keyfile = f"{os.getenv('HOME')}/ssl/ca-certificates/private_key.pem"
# RhService.logfile = f"{os.getenv('HOME')}/rhpyc.log"


class RpycBinder:
    """
    Поставьте этот декоратор над методом если хотите, чтобы он был вызван в определенном контексте
    conn -- объект соединения
    module -- модуль контекста
    decoder -- функция обратного вызова для возвращаемых данных
    Example:

    >>> @Binder(conn=rhconn, module="models.axis", decoder=decode_axis)
    ... def solve(curves, t0=0.0, t1=0.0, h=600.0):
    ...     crv0, crv1 = generate_polyline(curves[0]).ToNurbsCurve(), generate_polyline(curves[1]).ToNurbsCurve()
    ...     cells = CellingGenerator((crv0, crv1), (t0, t1), h)
    ...     return list(cells)

    В этом случае порядок действий будет следуюй:
    module
    solve
    decoder
    """

    def __init__(self, conn, module="", decoder=lambda x: x):
        self.target_module = importlib.import_module(module)
        self.conn = conn
        self.decoder = decoder
        self.conn.root.execute(dill.source.getsource(self.target_module))

    def __call__(self, obj):
        self.conn.root.execute("\n".join(dill.source.getsource(obj).split('\n')[1:]))

        @functools.wraps(obj)
        def wrap(**params):
            self.conn.root.execute(f"params = {params};result = {obj.__name__}(**params)")
            return self.decoder(self.conn.root.namespace["result"])

        return wrap
