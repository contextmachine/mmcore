import dataclasses
import typing
import uuid as _uuid

from steputils import p21
from steputils.tools import guid
from string import ascii_uppercase

from mmcore.base import AGroup
from mmcore.collections import OrderedSet

refs = {}

stepdct = dict()

typenodes = dict()


def step_to_nodes(data):
    for k, l in data.instances.items():
        if hasattr(l, "entity"):
            nested = STEPNestedEntity(type=l.entity.name, params=list(l.entity.params))
            if not (l.entity.name in typenodes.keys()):
                typenodes[l.entity.name] = []
            typenodes[l.entity.name].append(nested.params)

            stepdct[k] = STEPNode(ref=k, children=[nested])

        else:
            nitm = []
            for i in l.entities:
                nitm.append(STEPNestedEntity(i.name, i.params))
                if not (i.name in typenodes.keys()):
                    typenodes[i.name] = []
                typenodes[i.name].append(i.params)
            stepdct[k] = STEPNode(ref=k, children=nitm)


def step_refcheck(params, callback=lambda x: x.todict()):
    l = []
    for p in params:
        if isinstance(p, str):
            if p.startswith("#"):

                obj = stepdct[p]
                l.append(callback(obj))
            elif p == "NONE":
                l.append(None)
            elif p == ".T.":
                l.append(True)
            elif p == ".F.":
                l.append(False)
            else:
                l.append(p)
        elif isinstance(p, (tuple, list, p21.ParameterList)):
            if len(p) == 0:
                l.append(p)
            else:
                l.append(step_refcheck(p, callback=callback))

        else:
            l.append(p)
    return l


@dataclasses.dataclass
class STEPNestedEntity:
    type: str
    params: list

    def todict(self):
        return {
            "type": self.type,
            "params": step_refcheck(self.params)
        }

    def typemap(self):
        return {
            "type": self.type,
            "params": step_refcheck(self.params, callback=lambda x: x.typemap())
        }

    def convert(self, uuid_prefix=""):
        return {
            "type": self.type,
            "params": step_refcheck(self.params, callback=lambda x: x.convert(uuid_prefix=uuid_prefix))
        }


@dataclasses.dataclass
class STEPNode:
    ref: str
    children: list[STEPNestedEntity]
    uuid_prefix = ""

    def todict(self):
        return {
            "ref": self.ref,
            "children": [child.todict() for child in self.children]

        }

    def typemap(self):
        return [child.typemap() for child in self.children]

    def __getitem__(self, item):
        return stepdct[item]

    def convert(self, uuid_prefix=""):
        grp = AGroup(uuid=uuid_prefix+self.ref)
        for child in self.children:
            grp.add(child.convert(uuid_prefix=uuid_prefix))
        return grp


