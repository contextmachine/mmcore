from typing import Callable, Protocol

from mmcore.utils.sockets.examples.upd_client import CxmData


class Service(Protocol):
    _server_address = None

    server_address: tuple[str, int] | str
    bytesize: int | None
    extra_kwargs: dict

    def solve(self, msg) -> dict: ...

    def __call__(self, obj) -> Callable[..., 'Service']:
        self.outputs = obj.__match_args__
        print(self.outputs)
        self.obj = obj

        print(self.outputs)

        def wrp(**parameters):
            print(parameters)
            compressed = self.request(parameters=parameters, injection=obj.__doc__)
            print(compressed)
            cxm_out = self.solve(compressed)
            print(cxm_out)
            self.req = cxm_out
            self.inst = self.obj.__new__(obj, *cxm_out)

            return self

        return wrp

    def request(self, parameters=None, injection=None) -> CxmData:
        return CxmData.compress(
            dict(
                input=parameters,
                py=injection,
                output=self.outputs
                )
            )


from mmcore.collection.multi_description import ElementSequence

ImportStatement = namedtuple("ImportStatement", ["name", "asname"])


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

