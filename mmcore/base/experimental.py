import uuid as _uuid

object_table = dict()
component_table = dict()
geom_table = dict()
mat_table = dict()
tables = dict(component=component_table, geometry=geom_table, material=mat_table)


class Object3D:
    def __new__(cls, uuid=None):
        if uuid is None:
            uuid = _uuid.uuid4().hex
        obj = super().__new__(cls)
        obj.uuid = uuid
        component_table[uuid] = dict()
        object_table[uuid] = obj
        return obj


class Group(Object3D):
    def __new__(cls, children=None, *args, **kwargs):
        obj = super().__new__(cls, *args, **kwargs)
        if children is not None:
            children = set(child.uuid for child in children)
        else:
            children = set()
        component_table[obj.uuid]["__children__"] = children
        return obj

    def add(self, obj):
        component_table[self.uuid]["__children__"].add(obj.uuid)

    def remove(self, obj):
        component_table[self.uuid]["__children__"].remove(obj.uuid)

    @property
    def children_uuids(self):
        return component_table[self.uuid]["__children__"]

    @property
    def children(self):
        return [object_table[uuid] for uuid in self.children_uuids]


class GeometryObject(Object3D):
    def __new__(cls, geometry=None, material=None, *args, **kwargs):
        obj = super().__new__(cls, *args, **kwargs)
        component_table[obj.uuid]["__material__"] = material
        component_table[obj.uuid]["__geometry__"] = geometry
        return obj

    @property
    def geometry_uuid(self):
        return component_table[self.uuid]['__geometry__']

    @property
    def geometry(self):
        return geom_table[self.geometry_uuid]


def get_all_geometry_test(grp: Group):
    geoms = []
    for uuid in grp.children_uuids:
        if "__children__" in component_table[uuid].keys():
            geoms.extend(get_all_geometry_test(object_table[uuid]))
        if "__geometry__" in component_table[uuid].keys():
            geoms.append(object_table[uuid])
    return geoms
