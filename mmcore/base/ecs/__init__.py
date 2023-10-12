from uuid import uuid4


class ECS:
    def __init__(self):
        self.entities_stack = dict()
        self.components_stack = dict()
        self.components_type_stack = dict()
        self.entities = dict()
        self.components = dict()

        self.systems = dict()
        self.relay = dict()

    def register_entity(self, entity, *components):
        self.entities[entity.uuid] = dict()
        # self.entities_stack[entity.uuid] = entity
        for component in components:
            self.mount_component(entity.uuid, component)

    def mount_component(self, entity_uuid, component):
        self.components[component.__component_type_name__].add(entity_uuid)
        self.entities[entity_uuid][component.__component_type_name__] = component

    def unmount_component(self, entity_uuid, component):
        self.entities[entity_uuid].__delitem__(component.__component_type_name__)
        self.components[component.__component_type_name__].remove(entity_uuid)

    def get_components(self, entity):
        return self.entities[entity.uuid].values()

    def has_components(self, entity_uuid, *component_types):
        return all([component_type.__component_type_name__ in self.entities[entity_uuid] for component_type in
                    component_types])


class NodeGetItemAdaptor:
    def __init__(self, node):
        self.node = node

    def __getitem__(self, k):
        return self.node[k].value

    def __setitem__(self, k, v):
        self.node[k].update(v)

    def update(self, kwsa):
        self.node.update(kwsa)


class Component:
    __slots__ = ('ecs',)

    def __init__(self, ecs: ECS, **kwargs):
        super().__init__()


class Entity:
    __slots__ = ('uuid', 'ecs')

    def __new__(cls, uuid=None, /, *components, ecs: ECS = None):

        if uuid in ecs.entities:
            return ecs.entities_stack[uuid]
        if uuid is None:
            uuid = uuid4().hex
        obj = super().__new__(cls)
        obj.uuid = uuid
        obj.ecs = ecs
        ecs.register_entity(obj, components)
        return obj
