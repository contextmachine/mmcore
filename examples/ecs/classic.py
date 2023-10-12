import dataclasses

import time

from mmcore.base.ecs.entity import EcsModel, Entity, component


@component
@dataclasses.dataclass(slots=True, unsafe_hash=True)
class PointComponent:
    xyz: tuple


@component
@dataclasses.dataclass(slots=True, unsafe_hash=True)
class MountComponent:
    mount: bool
    mount_date: str = None


_tag_values = ["A0", "B1"]


@component
@dataclasses.dataclass(slots=True, unsafe_hash=True)
class EnumTagComponent:
    name: str = 'tag'
    _value: int = 0

    @property
    def value(self):
        return _tag_values[self._value]

    @property
    def values(self):
        return _tag_values

    @value.setter
    def value(self, v):
        if v not in _tag_values:
            _tag_values.append(v)
        self._value = _tag_values.index(v)

    def __iter__(self):
        return iter((self.name, self.value))


ecs = EcsModel()

e1 = Entity(model=ecs)
e2 = Entity(model=ecs)
e3 = Entity(model=ecs)

pc = PointComponent((0, 2, 3))
pc2 = PointComponent((5, 4, 2))

ecs.mount_components(e1, pc)
ecs.mount_components(e2, pc2)

mnt1 = MountComponent(mount=False)
mnt2 = MountComponent(mount=False)

ecs.mount_components(e1, mnt2)
ecs.mount_components(e3, mnt1)

mnt2.mount = True

tc1 = EnumTagComponent()
tc2 = EnumTagComponent()

ecs.mount_components(e1, tc2)
ecs.mount_components(e2, tc2)
ecs.mount_components(e3, tc1)

tc2.value = "C3"
print(tc2.values)

*mountable, = ecs.find_has_component_types(MountComponent)
print(mountable)


def stats_system(model):
    def stats_query(component_type, constrain):
        objs = model.find_has_component_types(component_type)

        for o in objs:
            if constrain(model.entities[o][component_type.__qualname__]):
                yield o

    return stats_query


my_stats_system = stats_system(ecs)

*mounted, = my_stats_system(MountComponent, lambda x: x.mount == True)

*not_mounted, = my_stats_system(MountComponent, lambda x: not x.mount)

print(f'mountable: {len(mountable)}, mount: {len(mounted)}, not mount: {len(not_mounted)}')
import numpy as np

for i in range(100_000):
    ent = Entity(model=ecs)
    mnt = MountComponent(ecs)
    tc = EnumTagComponent(ecs)
    ecs.mount_components(ent, mnt, tc2)
    mnt.mount = bool(np.random.randint(2))
    tc._value = np.random.randint(3)

s = time.time()
*mountable, = ecs.find_has_component_types(MountComponent)
*mounted, = my_stats_system(MountComponent, lambda x: x.mount == True)
# *not_mounted, = my_stats_system(MountComponent, lambda x: not x.mount)
rt = time.time() - s
print(
    f'mountable: {len(mountable)}, mount: {len(mounted)}, not mount: {len(not_mounted)} calculated at: {divmod(rt, 60)}')
