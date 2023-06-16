import typing

from mmcore.base.registry import AGraph, T


class SceneGraph(AGraph):

    def set_relay(self, node: T, name: str, v: typing.Any):
        pass

    ...


sgraph = SceneGraph()
