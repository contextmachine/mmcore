import dataclasses
import typing


@dataclasses.dataclass
class GQLAPIQuery:
    query: str
    variables: dict[str, typing.Any] | dict[None, None]
