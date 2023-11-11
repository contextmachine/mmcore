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
import inspect
import typing
import uuid as _uuid
from dataclasses import dataclass
from typing import Callable, Type

from mmcore.base.ecs.components import Component, apply

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


@dataclass
class StoredSystem:
    variables: dict[str, typing.Any]
    components: dict[str, Type[Component]]  # key is argument name
    has_entity_id_argument: bool
    has_ecs_argument: bool


class EntityComponentSystem:
    def __init__(self, on_create: Callable[[EntityId, list[Component]], None] = None,
                 on_remove: Callable[[EntityId], None] = None):
        """
        :param on_create:
        Хук, отрабатывающий при создании сущности,
        например может пригодиться, если сервер сообщает клиентам о появлении новых сущностей

        :param on_remove:
        Хук, отрабатывающий перед удалением сущности
        """

        # Здесь хранятся все системы вместе с полученными от них сигнатурами
        self._systems: dict[Callable, StoredSystem] = {}
        # По типу компонента хранятся словари, содержащие сами компоненты по ключам entity_id
        self._components: dict[Type[Component], dict[EntityId, Component]] = {}
        self._entities: list[EntityId] = []
        self._vars = {}
        self.on_create = on_create
        self.on_remove = on_remove

    def _unsafe_get_component(self, entity_id: EntityId, component_class: Type[Component]) -> Component:
        """
        Возвращает компонент сущности с типом переданного класса component_class
        Кидает KeyError если сущность не существует или не имеет такого компонента
        """

        return self._components[component_class][entity_id]

    def init_component(self, component_class: Type[Component]) -> None:
        """
        Инициализация класса компонента. Следует вызвать до создания сущностей
        """

        self._components[component_class] = {}

    def add_variable(self, variable_name: str, variable_value: typing.Any) -> None:
        """
        Инициализация переменной. Далее может быть запрошена любой системой.
        """

        self._vars[variable_name] = variable_value

    def init_system(self, system: Callable):
        """
        Инициализация системы. Если система зависит от внешней переменной - передайте её в add_variable до инициализации.

        Внешние переменные и специальные аргументы (ecs: EntityComponentSystem и entity_id: EntityId) запрашиваются
        через указание имени аргумента в функции системы.

        Запрашиваемые компоненты указываются через указание типа аргумента (например dummy_health: HealthComponent).
        Название аргумента в таком случае может быть названо как угодно.
        Запрашиваемый компонент должен быть инициализирован до инициализации системы
        """

        stored_system = StoredSystem(
            components={},
            variables={},
            has_entity_id_argument=False,
            has_ecs_argument=False
        )

        # Через сигнатуру функции системы узнаем какие данные и компоненты она запрашивает.
        # Сохраним в StoredSystem чтобы не перепроверять сигнатуру каждый кадр.
        system_params = inspect.signature(system).parameters
        for param_name, param in system_params.items():
            if param_name == 'entity_id':  # Система может требовать конкретный entity_id для переданных компонентов
                stored_system.has_entity_id_argument = True

            elif param_name == 'ecs':  # Системе может потребоваться ссылка на ecs. Например, для удаления сущностей
                stored_system.has_ecs_argument = True

            elif param.annotation in self._components:
                stored_system.components[param_name] = param.annotation

            elif param_name in self._vars:
                stored_system.variables[param_name] = self._vars[param_name]

            else:
                raise Exception(f'Wrong argument: {param_name}')

        self._systems[system] = stored_system

    def create_entity(self, components: list[Component]) -> EntityId:
        """
        Создание сущности на основе списка его компонентов
        Можно задавать свой entity_id но он обязан быть уникальным
        """

        entity_id = create_entity()
        for component in components:
            self._components[component.__class__][entity_id] = component
        self._entities.append(entity_id)

        if self.on_create:
            self.on_create(entity_id, components)

        return entity_id

    def get_entity_ids_with_components(self, *component_classes) -> set[EntityId]:
        """
        Получить все entity_id у которых есть каждый из компонентов, указанных в component_classes
        """

        if not component_classes:
            return set(self._entities)

        # Если запрошено несколько компонентов - то следует вернуть сущности, обладающие каждым из них
        # Это достигается пересечением множеств entity_id по классу компонента
        entities = set.intersection(*[set(self._components[component_class].keys())
                                      for component_class in component_classes])
        return entities

    def get_entities_with_components(self, *component_classes):
        """
        Получить все entity_id вместе с указанными компонентами
        """

        for entity_id in self.get_entity_ids_with_components(*component_classes):
            components = tuple(self._unsafe_get_component(entity_id, component_class)
                               for component_class in component_classes)
            yield entity_id, components

    def update(self) -> None:
        """
        Вызывает все системы.
        Следует вызывать в игровом цикле.
        """

        for system_function, system in self._systems.items():
            for entity_id in self.get_entity_ids_with_components(*system.components.values()):
                special_args = {}
                if system.has_ecs_argument:
                    special_args['ecs'] = self
                if system.has_entity_id_argument:
                    special_args['entity_id'] = entity_id

                # Сделано для того чтобы в системе можно было указывать любые имена для запрашиваемых компонентов
                required_components_arguments = {param: self._unsafe_get_component(entity_id, component_name) for
                                                 param, component_name in
                                                 system.components.items()}

                system_function(**(required_components_arguments | system.variables | special_args))

    def remove_entity(self, entity_id: EntityId):
        """
        Удаляет сущность
        """

        if self.on_remove is not None:
            self.on_remove(entity_id)
        for components in self._components.values():
            components.pop(entity_id, None)
        self._entities.remove(entity_id)

    def get_component(self, entity_id: EntityId, component_class: Type[Component]):
        """
        :return
        Возвращает компонент сущности с типом переданного класса component_class
        Возвращает None если сущность не существует или не имеет такого компонента
        """

        return self._components[component_class].get(entity_id, None)

    def get_components(self, entity_id: EntityId, component_classes):
        """
        :return
        Возвращает требуемые компоненты сущности.
        Возвращает None если сущность не существует или не имеет всех этих компонентов
        """

        try:
            return tuple(self._unsafe_get_component(entity_id, component_class)
                         for component_class in component_classes)
        except KeyError:
            return None
