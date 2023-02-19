import json
from typing import Any, Union

import pydantic
import pydantic.generics as pg

import mmcore.collections.generics.simple as gens
from ...baseitems import Item


class MModel(pydantic.BaseModel):
    """
    Base class from pydantic-mmodel
    """

    def __init__(self, **data: Any):
        super().__init__(**data)


class MmodelDict(gens.D3[Item, dict, str, MModel]):
    ...


class GenericMModel(pg.GenericModel):
    """
    Base class from pydantic-mmodel
    """

    def __init__(self, **data: Any):
        super().__init__(**data)


from typing import List
from pydantic import BaseModel


class pprop(property):

    def getter(self, fget):
        def w(slf):
            slf.__dict__[fget.__name__] = fget(slf)
            return slf.__dict__['mass']

        super().setter(w)


class LazyModel(BaseModel):
    symbols: List[int]

    @pprop
    def prp(self):
        ...


class PropertyBaseModel(pydantic.BaseModel):
    """
    Workaround for serializing properties with pydantic until
    https://github.com/samuelcolvin/pydantic/issues/935
    is solved
    """

    @classmethod
    def get_properties(cls):
        return [prop for prop in dir(cls) if
                isinstance(getattr(cls, prop), property) and prop not in ("__values__", "fields")]

    def dict(
            self,
            *,
            include: Any = None,
            exclude: Union['AbstractSetIntStr', 'MappingIntStrAny'] = None,
            by_alias: bool = False,
            skip_defaults: bool = None,
            exclude_unset: bool = False,
            exclude_defaults: bool = False,
            exclude_none: bool = False,
    ) -> 'DictStrAny':
        attribs = super().dict(
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            skip_defaults=skip_defaults,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none
        )
        props = self.get_properties()
        # Include and exclude properties
        if include:
            props = [prop for prop in props if prop in include]
        if exclude:
            props = [prop for prop in props if prop not in exclude]

        # Update the attribute dict with the properties
        if props:
            attribs.update({prop: getattr(self, prop) for prop in props})

        return attribs

    def json(self, *args, **kwargs):

        return json.dumps(self.dict(), *args, **kwargs)
