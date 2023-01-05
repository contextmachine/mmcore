import os
import sys
import socket
from typing import Protocol

import dill
import httpx

from cxmdata import CxmData
from mmcore.services.service import Service
from mmcore.addons import compute


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
    """

    Overhead restrictions
    """

    def __init__(self, address=('localhost', 10004), bytesize=4096 * 2, **kwargs):
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
