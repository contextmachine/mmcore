import abc
import typing

import strawberry.scalars

from strawberry.extensions import DisableValidation
from typing import TYPE_CHECKING, Annotated, TypeVar, Generic

import mmcore.base.models.gql
from mmcore.base.basic import *

TYPE_CHECKING = False

ee = ElementSequence(list(adict.values()))

T = TypeVar("T")



@strawberry.type
class Root(Generic[T]):
    object: T
    metadata: gql_models.Metadata
    materials: list[gql_models.AnyMaterial]
    geometries: list[typing.Union[gql_models.BufferGeometry, None]]


class ObjectRootFix(A):

    @property
    def root(self):
        target = self

        return Root[target.bind_class]


class GroupRootFix(AGroup):

    @property
    def root(self):
        target = self

        return Root[target.bind_class]


class MutableObject3D(A):
    def mutate(self, **inputs):
        ...


def mutation(slf):
    # bind_class = self.bind_class
    target = slf

    class Mutation:
        @strawberry.field
        def matrix(self, matrix: list[float]) -> target.bind_class:
            target.matrix = matrix
            return target.get_child_three().object

        @strawberry.field
        def name(self, name: str) -> target.bind_class:
            slf.name = name

            return target.get_child_three().object

        @strawberry.field
        def properties(self, key: str, value: str) -> target.bind_class:
            target.__setattr__("_" + key, value)
            return target.get_child_three().object

        @strawberry.field
        def material_by_uuid(self, uuid: str, material: gql_models.MaterialInput) -> gql_models.Material:
            mat = material.material
            mat.uuid = uuid
            amatdict[uuid] = mat
            return amatdict[uuid]

        @strawberry.field
        def material(self, material: mmcore.base.models.gql.MaterialInput) -> gql_models.Material:
            mat = material.material

            mat.uuid = target.material.uuid
            amatdict[mat.uuid] = mat

            return target.material

        """
        @strawberry.field
        def geometry(self, uuid: str) -> target.bind_class:
            target._geometry = uuid
            return target.get_child_three()["object"]



        @strawberry.field
        def geometry_by_uuid(self, geometry: GqlGeometry) -> target.bind_class:
            target.geometry = geometry
            return target.get_child_three()["object"]

        @strawberry.field
        def material_by_uuid(self, material: models.MeshPhongMaterial) -> target.bind_class:
            target.material = material

            return target.get_child_three()["object"]

        @strawberry.field
        def material(self, uuid: str) -> target.bind_class:
            target._material = uuid
            return target.get_child_three()["object"]"""

    """
    for name in self.properties.keys():
        def set_prop_resolver(_name):
            def wrap(value: type(getattr(target, _name))) -> target.bind_class:
                tp=type(getattr(target, _name))
                target.__setattr__(name, value)
                ##print(tp)
                return target.get_child_three()["object"]

            return wrap

        setattr(Mutation, "_" + name, strawberry.field(resolver=set_prop_resolver(name), name=name))"""

    return strawberry.type(Mutation)
strawberry.union("WhereInput", [] )
from mmcore.func.datatools.rules import Rule
class AbstractWhere:
    @abc.abstractmethod
    def interpret(self):
        ...
Lambda=type(lambda x:x)

class RuleCond(AbstractWhere):
    def __init__(self, ruleA, val):
        self.ruleA=ruleA
        self.val=val
    def interpret(self):
        return lambda ctx: self.ruleA(ctx, self.val.interpret())
class TermKeyCond(AbstractWhere):
    def __init__(self, key):
        self.key=key

    def interpret(self)->Lambda:
        return lambda ctx: ctx[self.key.interpret()]
class TermCond(AbstractWhere):
    def __init__(self, val):
        self.val=val
    def interpret(self)->typing.Any:
        return self.val


class BiRuleCond(AbstractWhere):
    def __init__(self, rule, a, b):
        self.a, self.b = a, b
        self.rule = rule

    def interpret(self):
        return lambda ctx: self.rule(self.a.interpret(ctx), self.b.interpret(ctx))


class KeyRuleCond(AbstractWhere):
    def __init__(self, key,rule):
        self.key=key
        self.rule=rule
    def interpret(self)->typing.Any:
        return lambda ctx : self.rule.interpret()(self.key.interpret()(ctx))



