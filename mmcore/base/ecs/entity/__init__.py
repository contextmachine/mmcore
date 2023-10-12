import functools
import typing
from uuid import uuid4

cmps = dict()
T = typing.TypeVar("T")


def component_type_name(cls) -> str:
    return cls._component_type_name


components = dict()
components_ent = dict()


def component(cls: typing.Type[T]) -> typing.Type[T]:
    name = cls.__qualname__
    components[name] = set()
    components_ent[name] = dict()
    cls._component_type_name = name
    cls.entities = property(fget=lambda self: components[name])

    return cls


@component
class EcsComponent:
    _component_type_name: str = ''


def get_entity(uuid: str):
    return entities[uuid]


EcsComponentType = typing.Type[EcsComponent]
entities = dict()


class Entity(object):
    __slots__ = ['uuid', 'model']

    def __new__(cls, uuid=None, model: 'EcsModel' = None):
        if uuid is None:
            uuid = uuid4().hex
        entity = super().__new__(cls)
        entity.uuid = uuid
        entity.model = model
        model.register_entity(entity)
        return entity

    def __hash__(self):
        return hash(self.uuid)

    def __str__(self):
        return f'{self.uuid}'

    def __repr__(self):
        return f'Entity object at {self.uuid}'


class EcsModel:

    def __init__(self):
        super().__init__()
        self.entities = dict()
        self.entiti_stack = dict()
        self.components_to_entities = dict()
        self.components = components

    def get_entity(self, entity_id: str):

        return get_entity(entity_id)

    def register_entity(self, entity: 'Entity', *components):
        self.entities[str(entity)] = dict()

        for component in components:
            self.mount_component(str(entity), component)

    def mount_component(self, entity_uuid, component):
        c = component_type_name(component)
        self.components[c].add(entity_uuid)
        components_ent[c][component] = entity_uuid
        self.entities[entity_uuid][c] = component

    def unmount_component(self, entity_uuid, component):
        self.entities[entity_uuid].__delitem__(component_type_name(component))
        self.components[component_type_name(component)].remove(entity_uuid)

    def get_components(self, entity):
        return self.entities[str(entity)].values()

    def has_components(self, entity_uuid, *components):
        return all([component_type_name(component) in self.entities[entity_uuid] for component in
                    components])

    def has_component_types(self, entity_uuid, *component_types):

        return all([component_type_name(component_type) in self.entities[entity_uuid] for component_type in
                    component_types])

    def find_has_component_types(self, *component_types):
        return functools.reduce(set.intersection,
                                [self.components[component_type_name(component_type)] for component_type in
                                 component_types])

    def find_has_component(self, component):
        tn = component_type_name(component)
        return filter(lambda x: component in self.get_components(x), self.components[tn])

    def mount_components(self, entity: 'Entity', *components):
        for cmp in components:
            self.mount_component(str(entity), cmp)

    def unmount_components(self, entity: 'Entity', *components):
        for component in components:
            self.entities[str(entity)].__delitem__(component_type_name(component))
            self.components[component_type_name(component)].remove(entity)
            component.entities.remove(str(entity))

    def register_component_type(self, cmp: EcsComponentType, name=None):
        if name is None:
            name = cmp.__qualname__
        cmp._component_type_name = name
        self.components[name] = set()

    def __contains__(self, item: typing.Union[Entity, EcsComponentType]):
        if isinstance(item, str):
            return self.entities.keys().__contains__(item)
        elif isinstance(item, Entity):
            return self.entities.keys().__contains__(item.uuid)
        else:
            return self.components.__contains__(component_type_name(item))

    def get_components_by_type(self, entity_id: str, component_type: EcsComponentType):

        return self.entities[entity_id][component_type_name(component_type)]

    def get_components_types(self, entity_id: str):

        return self.entities[entity_id].keys()
