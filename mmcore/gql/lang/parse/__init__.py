import typing
from typing import Optional
import dataclasses

import graphql.language.parser

from mmcore.gql.lang.ast import convert, ExtendedSelectionSetNode
from graphql.language.parser import Parser, SourceType
from graphql.language.ast import DocumentNode


def parse_line(line, target=" "):
    word = ""
    words = []
    for char in list(line + target):
        if char == target:
            if not (word == ""):
                words.append(word)
            word = ""
            continue
        else:
            word += char

    return words


class LineParser:
    replace_map = [("\t", ""),
                   ("\n", ""),
                   ("{", " { "),
                   ("}", " } "),
                   ("(", " ( "),
                   (")", " ) "),
                   (':', " : "),
                   ('"', "")]

    def __init__(self):
        super().__init__()

    def __call__(self, query: str):
        for rpl in self.replace_map:
            query = query.replace(*rpl)

        return parse_line(query)


line_parser = LineParser()


@dataclasses.dataclass
class ParsedQuery:
    query: str
    no_location: bool = False
    max_tokens: Optional[int] = None
    allow_legacy_fragment_variables: bool = False

    def __post_init__(self):
        self.query = " ".join(line_parser(self.query))
        self.parser = Parser(
            self.query,
            no_location=self.no_location,
            max_tokens=self.max_tokens,
            allow_legacy_fragment_variables=self.allow_legacy_fragment_variables

        )
        self.doc = self.parser.parse_document()
        self._getter = convert(self.doc.definitions[0].selection_set, ExtendedSelectionSetNode)

    def resolve(self, x: typing.Union[dict, typing.Mapping]) -> dict[str, typing.Any]:
        return {"data": self._getter.generic_getter()(x)}


def parse_simple_query(source: SourceType,
                       no_location: bool = False,
                       max_tokens: Optional[int] = None,
                       allow_legacy_fragment_variables: bool = False) -> ParsedQuery:
    """
    
    @return: ParsedQuery
    
    >>> # language=GraphQL
    ... query = '''
    ... {
    ...  grid {
    ...    object3d {
    ...      name
    ...     }
    ...   }
    ... } '''
    >>> dataset={
    ...     "grid": {
    ...         "object3d": {
    ...             "name":"A",
    ...             "uuid":"uu"
    ...         }
    ...     }
    ... }
    >>> parsed=parse_simple_query(query)
    >>> parsed
    ParsedQuery(query='{ grid { object3d { name } } }', no_location=False, max_tokens=None, allow_legacy_fragment_variables=False)
    >>> parsed.resolve(dataset)
    {'data': {'grid': {'object3d': {'name': 'A'}}}}
    

    """
    return ParsedQuery(
        source,
        no_location=no_location,
        max_tokens=max_tokens,
        allow_legacy_fragment_variables=allow_legacy_fragment_variables,
    )

parse_simple_query.__doc__+=graphql.language.parser.parse.__doc__