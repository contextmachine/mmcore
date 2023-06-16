import dataclasses

from mmcore.base.params import ParamGraphNode, pgraph

idict = pgraph.relay_table
adict = pgraph.item_table


@dataclasses.dataclass
class SceneNode(ParamGraphNode):
    graph = pgraph
