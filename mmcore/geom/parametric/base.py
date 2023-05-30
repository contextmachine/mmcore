import abc

import typing

T = typing.TypeVar("T")


@typing.runtime_checkable
class ParametricObject(typing.Protocol[T]):
    @abc.abstractmethod
    def evaluate(self, t):
        ...
