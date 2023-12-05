'''

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

    def __contains__(self, item: 'typing.Union[Entity, EcsComponentType]'):
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













'''
import uuid as _uuid

from mmcore.base.ecs.components import apply

entities = dict(
    entities=dict(),
    attrs=dict(),
    links=dict(),
    comp=dict()

)


def create_entity():
    entity = _uuid.uuid4().hex
    entities['entities'][entity] = dict()
    return entity


def create_entity_unsafe(uuid: str):
    entity = uuid
    entities['entities'][entity] = dict()
    return entity


def create_entity_attr(value=None):
    entity_attr = _uuid.uuid4().hex
    entities['attrs'][entity_attr] = dict()
    entities['links'][entity_attr] = value
    # entities['comp'][entity_attr] = 'any_component'

    return entity_attr


def create_entity_attr_unsafe(uuid: str, value=None):
    entity_attr = uuid
    entities['attrs'][entity_attr] = dict()
    entities['links'][entity_attr] = value

    # entities['comp'][entity_attr] = 'any_component'

    return entity_attr


def link_entity_attr(entity, name, attr):
    entities['entities'][entity][name] = attr
    entities['attrs'][attr][entity] = name


def get_entity_attr_value(attr):
    return entities['links'][attr]


def set_entity_attr_value(attr, value):
    entities['links'][attr] = value


attrs_types = dict()


def update_entity_attr_value(attr, data: dict):
    apply(entities['links'][attr], data)


def entity_components(entity):
    for attr in entities['entities'][entity].values():
        tp = entities['comp'].get(attr, None)
        if tp is not None:
            yield tp


def remove_entity_attr_value(attr):
    entities['comp'][attr] = None
    entities['links'][attr] = None


def entity_getattr_uuid(entity, field):
    return entities['entities'][entity][field]


def entity_getattr(entity, field):
    return get_entity_attr_value(entity_getattr_uuid(entity, field))


def link_entities_fields(entity1, entity2, field_name1, field_name2):
    link_entity_attr(entity2, field_name2, entity_getattr_uuid(entity1, field_name1))


def remove_link_entities_fields(entity, field_name):
    """
    entity.field attribute to new entity attribute with equal value
    Parameters
    ----------
    entity : str

    field_name : str


    Returns
    -------

    """

    entity_setattr_uuid(entity, field_name, create_entity_attr(entity_getattr(entity, field_name)))


# entity_getattr(e1, 'b')


def entity_setattr_value(entity, field, value):
    set_entity_attr_value(entity_getattr_uuid(entity, field), value)


def entity_setattr_uuid(entity, field, attr_uuid):
    """
    link_entity_attr alias

    Parameters
    ----------
    entity :
    field :
    attr_uuid :

    Returns
    -------

    """
    link_entity_attr(entity, field, attr_uuid)


def entity_hasattr(entity, field):
    return field in entities['entities'][entity].keys()


def entity_setattr(entity, field, value):
    if entity_hasattr(entity, field):
        entity_setattr_value(entity, field, value)
    else:
        link_entity_attr(entity, field, create_entity_attr(value))


def get_linked_entities(attr):
    return entities['attrs'][attr]


def remove_entity_attr(entity, name):
    attr = entities['entities'][entity][name]
    del entities['attrs'][attr][entity]
    del entities['entities'][entity][name]


def dispose_entity(entity):
    for k in list(entities['entities'][entity].keys()):
        remove_entity_attr(entity, k)
    del entities['entities'][entity]


def entity_dir(entity):
    return entities['entities'][entity].keys()


EntityId = str


