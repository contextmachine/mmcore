# language=Jinja2
import typing

import abc

import jinja2

_query_temp = """
query {
  {{ root }} {
    {% for attr in attrs %}
    {{ attr }}{% endfor %}
  }
}
"""
# language=Jinja2
_mutation_insert_one = """
mutation {{ directive_name }} ({{ x }}) {
     insert_{{ schema }}_{{ table }}_one(object: {{ y }}) {
        {% for attr in attrs %}
        {{ attr }}{% endfor %}
  }
}
"""
# language=Jinja2
selection_set = """{
  {% for attr in attrs %}
  {{ attr.render() }}{% endfor %}
}
"""

jenv = jinja2.Environment()


class AbstractGqlItem(typing.Protocol):
    @property
    def temp_string(self) -> str:
        ...

    @property
    def template(self) -> jinja2.Template: return jinja2.Template(self.temp_string)

    def render(self) -> str:
        return self.template.render({"item": self})

    def __repr__(self):
        return self.render()


class DQStr(str):
    def __repr__(self):
        return f'"{self}"'


class GqlVar(str):

    def __repr__(self) -> str:
        return f"${self}"


class GqlAttr(AbstractGqlItem):
    def __init__(self, key, value=None):
        self.key = key
        self.value = DQStr(value) if isinstance(value, str) else value if value is not None else GqlVar(self.key)

    @property
    def temp_string(self) -> str:
        # language=Jinja2
        return "{{ item.key }}: {{ item.value.__repr__() }}"

    def render(self) -> str:
        return self.template.render({"item": self})


from graphql import type as gtype

type_dict = {
    int: gtype.GraphQLInt,
    float: gtype.GraphQLFloat,
    str: gtype.GraphQLString,
    DQStr: gtype.GraphQLString,
    bool: gtype.GraphQLBoolean

}


class GqlTypedAttr(GqlAttr):
    def __init__(self, key, value=None, gql_type=None):
        super().__init__(key, value)
        self._gql_type = gql_type

    @property
    def gql_type(self):
        if self._gql_type is None:
            if type(self.value) in type_dict.keys():
                return type_dict[type(self.value)].name
            else:
                return "JSON"
        else:
            return self._gql_type

    @property
    def temp_string(self) -> str:

        return f"${self.key}: {self.gql_type} = {self.value.__repr__()}"


class GqlItemSpec(AbstractGqlItem):
    def __init__(self, attrs=None):

        self._spec = attrs

    @property
    def spec(self):
        if self._spec is not None:
            return self._spec
        else:
            return ""

    @property
    def temp_string(self) -> str:
        # language=Jinja2
        if self._spec is not None:
            return "(" + ", ".join(attr.render() for attr in self.spec) + ")"
        return ""


class SelectionSet(AbstractGqlItem):
    """
    >>> query = SelectionSet(
    ...     SelectionSetItem(
    ...         name="first",
    ...     ),
    ...     SelectionSetItem(
    ...         name="second",
    ...         spec=GqlItemSpec(attrs=[
    ...             GqlAttr("one", 3),
    ...             GqlAttr("two", "foo"),
    ...             GqlAttr("bar",None)
    ...         ]),
    ...         value=SelectionSet(
    ...
    ...                 SelectionSetItem(
    ...                     name="foo"
    ...                 ),
    ...                 SelectionSetItem(
    ...                     name="bar"
    ...                 ),
    ...                 SelectionSetItem(
    ...                     name="baz"
    ...                 )
    ...
    ...         )
    ...     )
    ...
    ... )
    >>> query
    { first second(one: 3, two: "foo", bar: $bar) { foo bar baz } }
    """

    def __init__(self, *items: 'SelectionSetItem'):
        self.items = items

    @property
    def temp_string(self) -> str:
        if len(self.items) > 0:
            # language=Jinja2
            return "{ " + " ".join(attr.render() for attr in self.items) + " }"
        return ""


class GqlName(AbstractGqlItem):
    def __init__(self, name):
        self.name = name

    def temp_string(self) -> str:
        # language=Jinja2
        return f"{self.name}"


class SelectionSetItem(AbstractGqlItem):
    def __init__(self, name, spec: GqlItemSpec = None, value: SelectionSet = None):
        self.name = name
        self.spec = spec
        self.value = value

    @property
    def temp_string(self):
        # language=Jinja2
        if isinstance(self.name, str):
            temp = self.name
        else:
            temp = self.name.render()
        if self.spec is not None:
            temp += self.spec.render()
        if self.value is not None:
            temp += (" " + self.value.render())
        return temp
