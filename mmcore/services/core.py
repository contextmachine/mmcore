from typing import Protocol


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


