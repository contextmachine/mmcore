from dataclasses import dataclass
from functools import wraps
from typing import Any

import requests
from jinja2.nativetypes import NativeEnvironment

from mmcore.baseitems.descriptors import DataView
from mmcore.collection.multi_description import ElementSequence
from mmcore.collection.traversal import traverse

GQL_PLATFORM_URL = "http://84.201.140.137/v1/graphql"
from pg import String, Int, name


def pg_type_matcher(f):
    @wraps(f)
    def wrp(obj):
        match obj[1].__class__.__name__, obj[0]:

            case 'str', 'name':
                return f(obj[0], name(obj[1]))
            case 'str', _:
                return f(obj[0], String(obj[1]))
            case 'int', _:
                return f(obj[0], Int(obj[1]))

    return wrp


@pg_type_matcher
def match_pg_type(kv):
    k, v = kv
    return k, v


@pg_type_matcher
def formatter(x, y):
    return f'${x}: {y.__class__.__name__} = {y.pg_default}'


def get(self, route, *args, **kwargs):
    request = requests.get(url="/".join((self.url, route)), headers=self.headers)
    assert request.ok, f"Failed with code {request.status_code}"
    data = []
    res = request.json()
    trv = traverse(callback=lambda x: data.extend(x) if isinstance(x, list) else None, traverse_seq=False)
    trv(res)
    return data


class GQLRequest:
    def __init__(self):
        super().__init__()

    @property
    def query(self) -> str:
        ...

    @property
    def variables(self) -> dict:
        ...


@dataclass
class GQLClient:
    url: str
    headers: dict

    def __call__(self, dcls):
        self._cls = dcls
        return self


class GQLPaginateClient(GQLClient):
    def post_processing(self, request: requests.Response) -> dict[str, Any] | list[dict[str, Any]]:
        data = []
        paginate = traverse(callback=lambda x: data.extend(x) if isinstance(x, list) else None, traverse_seq=False)
        paginate(super().post_processing(request))
        return data


client = GQLClient(url="http://84.201.140.137/v1/graphql",
                   headers={
                       "content-type": "application/json",
                       "user-agent": "JS GraphQL",
                       "X-Hasura-Role": "admin",
                       "X-Hasura-Admin-Secret": "mysecretkey"
                   })

mutat = """
mutation MaskMutation($matrix: jsonb = "", $uuid: uuid! = "") {
  update_buffgeom_objects(where: {uuid: {_eq: $uuid}}, _set: {matrix: $matrix}) {
    returning {
      uuid
      geometry
      matrix
      children
      name
      userData
    }
  }
}
"""
get_uuids = """
mutation {classname}($matrix: jsonb = "", $uuid: uuid! = "") {
  update_buffgeom_objects(where: {uuid: {_eq: $uuid}}, _set: {matrix: $matrix}) {
    returning {
      {return_keys}
      geometry
      matrix
      children
      name
      userData
    }
  }
}
"""

"""
mutation InsertColor(${{  }}: Int = 10, $decimal: Int = 10, $name: name = "", $palette: Int = 10, $hex: String = "", $r: Int = 10, $g: Int = 10) {
  insert_presets_colors_one(object: {b: $b, decimal: $decimal, g: $g, hex: $hex, name: $name, palette: $palette, r: $r})
"""


class BufferGeometryDictionary(dict):
    def __init__(self, req):
        dct = {
            "data": {
                "attributes": dict((a, req[a]) for a in ["position", "normal", "uv"]),
                "index": req["index"]
            },
            "type": req["type"],
            "uuid": req["uuid"]
        }

        dict.__init__(self, **dct)


_query_temp = """
query {
  {{ root }} {
    {% for attr in attrs %}
    {{ attr }}{% endfor %}
  }
}
"""
_mutation_insert_one = """
mutation {{ directive_name }} ({{ x }}) {
     insert_{{ schema }}_{{ table }}_one(object: {{ y }}) {
        {% for attr in attrs %}
        {{ attr }}{% endfor %}
  }
}
"""


class GQLVarsDescriptor(DataView):
    def item_model(self, name, value):
        return name, value

    def data_model(self, instance, value: list[None | tuple[str, Any]]):
        return dict(value)


def format_mutation(obj: dict, tmpl, back=("id", "name"), directive_name="InsertAny",
                    schema="presets", table="colors"):
    a, b = ", ".join([formatter(kv) for kv in obj.items()]), "{" + ", ".join([f'{k}: ${k}' for k in obj.keys()]) + "}"

    return tmpl.render(directive_name=directive_name,
                       schema=schema,
                       table=table,
                       x=a,
                       y=b,
                       attrs=back
                       )


class AbsGQLTemplate:
    schema: str = "test"
    table: str = "test"
    variables = {}
    client = GQLClient(url=GQL_PLATFORM_URL,
                       headers={
                           "content-type": "application/json",
                           "user-agent": "JS GraphQL",
                           "X-Hasura-Role": "admin",
                           "X-Hasura-Admin-Secret": "mysecretkey"
                       })
    temp: str = _query_temp

    @property
    def root(self):
        return f'{self.schema}_{self.table}'

    def __init__(self, temp_args=None, variables=None, **kwargs):
        super().__init__()
        self.directive_name = self.__class__.__name__
        if temp_args is None:
            temp_args = dict()
        self.temp_args = temp_args
        self.__dict__ |= kwargs
        self._jinja_env = NativeEnvironment()
        self.template = self._jinja_env.from_string(self.temp)
        if variables is not None:
            self.variables |= variables

    def __set_name__(self, owner, name):
        self.name = name

    def do(self, instance=None, variables=None, **kwargs):
        return self.template.render(**self.temp_args)

    def __get__(self, inst, own):
        return self.run_query(inst, variables=self.variables)

    def run_query(self, inst=None, variables: dict = None, **kwargs):
        """
        Возвращает на самом деле `self.post_processing(request)` ,
        где `request` -- необработанный результат запроса к апи.
        @param variables:
        @param inst:
        @param own:
        @return:

        """

        if variables is None:
            variables = {}
        mtt = self.do(inst, variables=variables, **kwargs)
        print(inst, variables, mtt)
        # print(self._body(inst, own))
        request = requests.post(self.client.url,
                                headers=self.client.headers,
                                json={
                                    "query": mtt,
                                    "variables": variables

                                }
                                )
        return self.post_processing(request)

    def post_processing(self, request: requests.Response) -> dict[str, Any]:
        return request.json()

    def dumps(self, obj=None, variables=None, **kwargs):
        if variables is None:
            variables = {}
            variables |= self.variables

        return self.do(instance=obj, variables=variables, **kwargs)

    def dump(self, file, **kwargs):
        if isinstance(file, str):
            with open("".join(file.split(".")[:-1]) + ".graphql", "w") as fl:
                fl.write(self.do(**kwargs))
        else:
            return file.write(self.do(**kwargs))


class GQLQuery(AbsGQLTemplate):
    """
    It is identical this:
    ```graphql
    query {
      buffgeom_attributes {

         uuid
        }
      }
    ```
    >>> query = GQLQuery(temp_args=dict(root="buffgeom_attributes",
    ...                         attrs=("uuid",)))
    >>> query.run_query()
    [{'uuid': '05c70b0e-3c8a-4459-a0e5-a777b0263cf2'},
     {'uuid': '313ad897-3b58-497f-80fd-cf1344c4a827'}]
    >>> class GQLQuerySet:
    ...     uuids = GQLQuery(temp_args=dict(
    ...         attrs=("uuid",)),
    ...         schema="buffgeom",
    ...         table="attributes")
    ...
    ...     all = GQLQuery(temp_args=dict(
    ...
    ...         attrs=("position", "normal", "uv", "type", "uuid", "index")),
    ...         schema="buffgeom",
    ...         table="attributes")
    ...
    ...     attributes = GQLQuery(temp_args=dict(
    ...         attrs=("position", "normal", "uv")),
    ...         schema="buffgeom",
    ...         table="attributes")
    ...     first_level = GQLQuery(temp_args=dict(
    ...
    ...         attrs=("type", "uuid")),
    ...     index = GQLQuery(temp_args=dict(
    ...
    ...         attrs=("index",)),
    ...         schema="buffgeom",
    ...         table="attributes")
    ...     """

    def __init__(self, temp_args=None, variables=None, **kwargs):
        super().__init__(temp_args=temp_args, variables=variables, **kwargs)
        if variables is not None:
            self.variables |= variables

    def __get__(self, inst, own):
        return self.run_query(inst)

    def post_processing(self, request: requests.Response) -> dict[str, Any]:
        """
        Переопределите этот метод для кастомной обработки ответа
        @param request:
        @return:
        """
        return super().post_processing(request)["data"][self.root]


class GQLError(BaseException):
    ...


class GQLMutation(AbsGQLTemplate):
    """
    """
    temp = _mutation_insert_one

    @property
    def root(self):
        return f'insert_{self.schema}_{self.table}_one'

    def do(self, instance=None, variables=None, **kwargs):
        return format_mutation(variables, self.template, back=self.temp_args['attrs'],
                               directive_name=self.directive_name,
                               schema=self.schema, table=self.table)

    def __set__(self, instance, value: dict):
        self.run_query(instance=instance, variables=value)

    def post_processing(self, request: requests.Response) -> Any:
        """
        Переопределите этот метод для кастомной обработки ответа
        @param request:
        @return:
        """
        try:
            return super().post_processing(request)["data"][self.root]
        except KeyError as err:
            print(request.request.body.decode())
            raise GQLError(f'\n{super().post_processing(request)}')
        # return request.json()


class _mutation(str):
    def __new__(cls, client=client, returned=("uuid", "name"), variables: dict = dict({"matrix": ("jsonb", "")}), ):
        name, (typ, val) = list(variables.items())[0]
        varss = []
        for n in list(variables.items()):
            name, (typ, val) = n
            varss.append((name, val))
        format_seq = [cls.__name__, name, typ, name, name, "\n\t".join(returned)]

        mutat = """
        mutation %($% : % = "", $uuid: uuid! = "") {
          update_buffgeom_objects(where: {uuid: {_eq: $uuid}}, _set: {%: $%}) {
            returning {
              %
            }
          }
        }
        """

        uuu = list(mutat)
        i = 0
        while True:
            try:
                uuu[uuu.index("%")] = format_seq[i]
                i += 1
            except:
                break
        indt = super().__new__(cls, "".join(uuu))

        indt.client = client
        indt.variables = varss
        return indt

    def run_query(self):
        request = requests.post(
            self.client.url,
            headers=self.client.headers,
            json={"query": self, "variables": self.variables},
        )
        assert request.ok, f"Failed with code {request.status_code}"
        return request.json()


matrix, uuid = [
    0.721,
    0,
    0,
    0,
    0,
    0.22767221,
    0,
    0,
    0,
    0,
    - 0.3444,
    0,
    0,
    0,
    0,
    1
], "9de4c938-c011-4b05-a958-2fbd455e5c30"


class GQLPaginateQuery(GQLQuery):
    """
    @note
    Единственнвя существенная разница -- это то что Paginate возвращвет ElementSequence вместо оригинального массива.
    А также принимает необязательным аргументом класс в котлорый нужно преобразовать каждый dict из массива.
    ### Simple example:
    >>> pf=GQLPaginateQuery(temp_args=dict(
    ...      root="buffgeom_attributes",
    ...      attrs=("position", "normal", "uv", "type", "uuid", "index")))
    >>> pf.run_query()
    ElementSequence[list(['position', 'normal', 'uv', 'type', 'uuid', 'index'])]


    @note: Example with a custom class:
    [1] Class definition. Не трудно заметить что класс стал несравнимо меньше.
        Если мы используем graphql, все что нам нужно -- это переодически перетасовывать аргументы,
        для полцучения разных структур данных.
    >>> class BufferGeometryDict(dict):
    ... def __init__(self, req):
    ...    dct = {
    ...        "data": {
    ...            "attributes": dict((kwarg, req[kwarg]) for kwarg in ["position", "normal", "uv"]),
    ...            "index": req["index"]
    ...        },
    ...        "type": req["type"],
    ...        "uuid": req["uuid"]
    ...    }
    ...
    ...    dict.__init__(self, **dct)


    [2] Пременение остается столь-же простым:

    >>> pf=GQLPaginateQuery(target_class=BufferGeometryDict, temp_args=dict(
    ...      root="buffgeom_attributes",
    ...      attrs=("position", "normal", "uv", "type", "uuid", "index")))
    >>> pf.run_query()
    ElementSequence[list(['data', 'type', 'uuid'])]

    @note: * В отображении мы получим имена аргументов первого уровня,тк мы унаследовались от дикта.
    """

    def __init__(self, *args, target_class=None, **kwargs, ):
        super().__init__(*args, **kwargs)
        self.target_class = target_class

    def post_processing(self, request: requests.Response) -> ElementSequence:
        resp = super().post_processing(request)
        if self.target_class is not None:
            resp = list(map(lambda x: self.target_class(x), resp))
        return ElementSequence(resp)
