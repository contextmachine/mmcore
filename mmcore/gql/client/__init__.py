from collections import UserDict, namedtuple
from dataclasses import dataclass
from typing import Any, NamedTuple

import requests
from jinja2.nativetypes import NativeEnvironment

from mmcore.collections.multi_description import ElementSequence
from mmcore.gql.templates import _query_temp, _mutation_insert_one
from ..pg import format_mutation


class GQLException(BaseException):
    def __init__(self, *args, error_dict: dict = None):
        super().__init__(*args)
        self.error_dict = error_dict
        # self.message = error_dict["message"]
        # self.extensions = error_dict["extensions"]
        print(self.error_dict)


GQL_PLATFORM_URL = "http://62.84.116.219:8081/v1/graphql"

__all__ = [
    "GQLClient",
    "GQLQuery",
    "GQLPaginateQuery",
    "GQLMutation"
]


@dataclass
class GQLClient:
    url: str
    headers: dict

    def __call__(self, dcls):
        self._cls = dcls
        return self


class GqlString(str):
    def __new__(cls, a: str, b: str, condition: str = None):
        s = "_".join([a, b])
        if condition is not None:
            s += condition

        inst = str.__new__(cls, s)
        inst.a, inst.b, inst.condition = a, b, condition
        inst._no_condition = "_".join([a, b])
        return inst


class query:
    """
    Minimal syntax:
    >>> my_query = query("geometry_attributes",fields= {
    ...         "index",
    ...         "normal",
    ...         "position",
    ...         "type",
    ...         "uuid",
    ...         "uv"
    ...       }
    ...     )
    """

    temp: str = """
    query {
        {{ root }} {
        {% for attr in attrs %}
        {{ attr }}{% endfor %}
        }
    }"""

    headers = {
        "content-type": "application/json",
        "user-agent": "JS GraphQL",
        "X-Hasura-Role": "admin",
        "X-Hasura-Admin-Secret": "mysecretkey"
    }

    def __init__(self, parent: str | GqlString, fields: set = {}, variables={}):
        super().__init__()

        self.parent = parent
        self.children = set(fields)
        self.fields = fields
        self._jinja_env = NativeEnvironment()
        self.template = self._jinja_env.from_string(self.temp)
        self.variables = variables

    def render(self, fields=None):
        if fields is not None:

            return self.template.render(root=self.parent, attrs=fields)
        else:
            return self.template.render(root=self.parent, attrs=self.fields)

    @property
    def client(self):
        return GQLClient(url=GQL_PLATFORM_URL, headers=self.headers)

    def __call__(self, fields=None, variables=None, full_root=True, return_json=True):
        if fields is None:
            fields = self.fields
        if variables is None:
            variables = self.variables
        ren = self.render(fields=fields)
        # print(self._body(inst, own))
        request = requests.post(self.client.url,

                                headers=self.client.headers,
                                json={
                                    "query": ren,
                                    "variables": variables

                                }
                                )
        if request.status_code == 200:
            if return_json:

                try:
                    data = request.json()
                    if "errors" in data.keys():
                        raise GQLException("GraphQl response error", error_dict=data)
                    elif "data" in data.keys():
                        if not full_root:
                            return request.json()["data"][self.parent]
                        else:
                            return request.json()

                    else:
                        return data

                except Exception as err:
                    raise err
            else:
                return request
        else:
            print("bad response")
            return request


class mutate(query):
    temp = """mutation mutate($objects: [{{ roota }}_{{ rootb }}_insert_input!] = {}) { 
    insert_{{ roota }}_{{ rootb }}(on_conflict: {constraint: {{ rootb }}_pkey}, objects: $objects) { 
    returning {
        {% for attr in attrs %}
        {{ attr }}{% endfor %}
        }
      }
    }
    """

    def __init__(self, parent: GqlString, fields, variables, **kwargs):
        super().__init__(parent, fields, variables, **kwargs)

    def render(self, fields=None):
        if fields is not None:

            return self.template.render(roota=self.parent.a, rootb=self.parent.b, attrs=fields)
        else:
            return self.template.render(roota=self.parent.a, rootb=self.parent.b, attrs=self.fields)


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
        return self.template.render(root=self.root, **self.temp_args)

    def __get__(self, inst, own):
        return self.post_processing(self.run_query(inst, variables=self.variables))

    def run_query(self, inst=None, variables: dict = None, **kwargs):
        """
        ???????????????????? ???? ?????????? ???????? `self.post_processing(request)` ,
        ?????? `request` -- ???????????????????????????? ?????????????????? ?????????????? ?? ??????.
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
        return request

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
    >>> query = GQLQuery(temp_args=dict(attrs=("uuid",)), schema="buffgeom", table="attributes")
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
        ???????????????????????????? ???????? ?????????? ?????? ?????????????????? ?????????????????? ????????????
        @param request:
        @return:
        """
        try:
            return super().post_processing(request)["data"][self.root]
        except KeyError as err:
            print(request.request.body.decode())
            raise GQLError(f'\n{super().post_processing(request)}')


class GQLError(BaseException):
    ...


namedtuple("GQLrequest", [])


class GQl(UserDict):
    def __getitem__(self, variables, inst, kwargs):
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
        ???????????????????????????? ???????? ?????????? ?????? ?????????????????? ?????????????????? ????????????
        @param request:
        @return:
        """
        try:
            return super().post_processing(request)["data"][self.root]
        except KeyError as err:
            print(request.request.body.decode())
            raise GQLError(f'\n{super().post_processing(request)}')
        # return request.json()


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
    ???????????????????????? ???????????????????????? ?????????????? -- ?????? ???? ?????? Paginate ???????????????????? ElementSequence ???????????? ?????????????????????????? ??????????????.
    ?? ?????????? ?????????????????? ???????????????????????????? ???????????????????? ?????????? ?? ???????????????? ?????????? ?????????????????????????? ???????????? dict ???? ??????????????.
    ### Simple example:
    >>> pf=GQLPaginateQuery(temp_args=dict( attrs=("position", "normal", "uv", "type", "uuid", "index")),schema="buffgeom",table="attributes")

    >>> pf.run_query()
    ElementSequence[list(['position', 'normal', 'uv', 'type', 'uuid', 'index'])]


    @note: Example with a custom class:
    [1] Class definition. ???? ???????????? ???????????????? ?????? ?????????? ???????? ???????????????????? ????????????.
        ???????? ???? ???????????????????? graphql, ?????? ?????? ?????? ?????????? -- ?????? ???????????????????????? ???????????????????????????? ??????????????????,
        ?????? ???????????????????? ???????????? ???????????????? ????????????.
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


    [2] ???????????????????? ???????????????? ??????????-???? ??????????????:

    >>> pf=GQLPaginateQuery(target_class=BufferGeometryDict, temp_args=dict(
    ...    attrs=("position", "normal", "uv", "type", "uuid", "index")),schema="buffgeom",table="attributes")
    >>> pf.run_query()
    ElementSequence[list(['data', 'type', 'uuid'])]

    @note: * ?? ?????????????????????? ???? ?????????????? ?????????? ???????????????????? ?????????????? ????????????,???? ???? ???????????????????????????? ???? ??????????.
    """

    def __init__(self, *args, target_class=None, **kwargs, ):
        super().__init__(*args, **kwargs)
        self.target_class = target_class

    def post_processing(self, request: requests.Response) -> ElementSequence:
        resp = super().post_processing(request)
        if self.target_class is not None:
            resp = list(map(lambda x: self.target_class(x), resp))
        return ElementSequence(resp)


def geometry_query(schema="geometries", table="attributes"):
    def wrap(*attrs):
        return GQLQuery(temp_args=dict(attrs=attrs), schema=schema, table=table)

    return wrap


class GQLSimpleQuery:
    client = GQLClient(url=GQL_PLATFORM_URL,
                       headers={
                           "content-type": "application/json",
                           "X-Hasura-Role": "admin",
                           "X-Hasura-Admin-Secret": "mysecretkey"
                       })

    def __init__(self, template, fields=()):
        super().__init__()
        self._jinja_env = NativeEnvironment()
        self.template = self._jinja_env.from_string(template)
        self.fields = fields



    def render(self, fields=None):
        if fields is not None:

            return self.template.render(attrs=fields)
        else:
            return self.template.render(attrs=self.fields)

    def __call__(self, variables=None, full_root=False, return_json=True, fields=None):

        if variables is None:
            variables = {}

        # print(self._body(inst, own))
        request = requests.post(self.client.url,

                                headers=self.client.headers,
                                json={
                                    "query": self.render(fields=fields if fields is not None else None),
                                    "variables": variables

                                }
                                )
        if request.status_code == 200:
            if return_json:

                try:
                    data = request.json()
                    if "errors" in data.keys():
                        print(data)
                        raise GQLException("GraphQl response error", error_dict=data)
                    elif "data" in data.keys():
                        if not full_root:
                            return request.json()["data"]
                        else:
                            return request.json()

                    else:
                        return data

                except Exception as err:
                    raise err
            else:
                return request
        else:
            print("bad response")
            return request


class GQLFileBasedQuery(GQLSimpleQuery):
    def __init__(self, path):
        with open(path) as f:
            super().__init__(f.read())
