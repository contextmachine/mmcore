from dataclasses import dataclass

import requests

from mmcore.collection.traversal import traverse


@dataclass
class Client:
    url: str
    headers: dict

    def get(self, route, *args, **kwargs):
        request = requests.get(url="/".join((self.url, route)), headers=self.headers)
        assert request.ok, f"Failed with code {request.status_code}"
        data = []
        res = request.json()
        trv = traverse(callback=lambda x: data.extend(x) if isinstance(x, list) else None, traverse_seq=False)
        trv(res)
        return data

    def post(self, **kwargs):
        request = requests.post(self.url,
                                headers=self.headers,
                                json={
                                    "query": mutat,
                                    "variables": kwargs,
                                    "operationName": "MaskMutation"
                                }
                                )
        data = []
        res = request.json()
        trv = traverse(callback=lambda x: data.extend(x) if isinstance(x, list) else None, traverse_seq=False)
        trv(res)
        return data


client = Client(url="http://84.201.140.137/v1/graphql",
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


class mutation(str):
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
res = client.post(matrix=matrix, uuid=uuid)
