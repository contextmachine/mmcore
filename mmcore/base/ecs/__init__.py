import inspect
import typing
from dataclasses import dataclass
from typing import Callable, Type

from mmcore.base.ecs.components import Component
from mmcore.base.ecs.entity import EntityId, create_entity


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

    def register_create_hook(self, hook: Callable[[EntityId, list[Component]], None]):
        self.on_create = hook
        return hook

    def register_remove_hook(self, hook: Callable[[EntityId], None]):
        self.on_remove = hook
        return hook

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


@dataclass
class StoredSystem:
    variables: dict[str, typing.Any]
    components: dict[str, Type[Component]]  # key is argument name
    has_entity_id_argument: bool
    has_ecs_argument: bool


ECS = EntityComponentSystem()
