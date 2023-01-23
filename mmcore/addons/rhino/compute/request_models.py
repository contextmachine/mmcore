import json

with open("data/mmconfig.json") as j:
    mmconfig = json.load(j)
MMCONFIG = mmconfig["bundles"]
GHDEPLOYPATH = "c:/users/administrator/compute-deploy/"


class ComputeRequest:

    def __init__(self, endpoints):
        object.__init__(self)
        self.endpoint_get, self.endpoint_post = self.endpoints = endpoints

    @property
    def compute_url(self):
        return Util.url

    @property
    def url(self):
        return f"{self.compute_url}"

    @property
    def headers(self):
        return mmconfig[""]

    def get(self):
        return requests.get(self.url + self.endpoint_get)

    def post(self, data=None):
        return requests.post(self.url + self.endpoint_post, json=data, headers=self.headers)


class ComputeGeometryRequest(ComputeRequest):
    ...


from functools import wraps
from typing import Any

import requests
from compute_rhino3d import Util
from cxmdata import CxmData


class GHRequest(ComputeRequest):
    def __init__(self, target="example.gh", endpoints=("io?pointer=", "grasshopper")):
        self.target = GHDEPLOYPATH + target
        get, post = endpoints
        super().__init__(endpoints=(get + self.target, post))

    def solve(self, *args, **kwargs):
        response = self.post(*args, **kwargs)
        return CxmData(
            list(response.json()["values"][0]["InnerTree"].values())[0][0]["data"].replace('"', "")).decompress()


class SlimGHRequest(GHRequest):
    """
    Предполагается что определение имеет один вход и один выход "input/output" с типами данных cxmdata/cxmdata.
    Этого должно быть достаточно в 90% случаев.

    Если в метод post не поступает аргумент data, определение будет вычисляться с input по умолчанию (если такое
    значение существует)
    """

    @property
    def defaults(self) -> Any:
        return CxmData(self.get().json()["Inputs"][0]["Default"]).decompress()

    def post(self, data=None):
        if data is None:
            data = self.defaults

        return requests.post(self.url + self.endpoint_post,
                             json={
                                 "pointer": GHDEPLOYPATH + self.target,

                                 "values": [{
                                     "ParamName": "input",
                                     "InnerThree": {
                                         "0": [
                                             {
                                                 "type": "System.String",
                                                 "data": CxmData.compress(data).decode()
                                                 }
                                             ]
                                         }
                                     }]},
                             headers=self.headers

                             )

    def solve(self, *args, **kwargs):
        response = self.post(*args, **kwargs)
        return CxmData(
            list(response.json()["values"][0]["InnerTree"].values())[0][0]["data"].replace('"', "")).decompress()


class AdvanceGHRequest(GHRequest):
    """
    Предполагается что определение имеет несколько входов и один выход "input/output" с типами данных cxmdata/cxmdata.
    Этого должно быть достаточно в 90% случаев.

    Если в метод post не поступает аргумент data, определение будет вычисляться с input по умолчанию (если такое
    значение существует)
    """

    def __init__(self, target="example.gh", endpoints=("io?pointer=", "grasshopper")):
        super().__init__(target=target, endpoints=endpoints)

    @property
    def defaults(self) -> Any:
        fields = []
        for field in self.get().json()["Inputs"]:
            fields.append((field["Name"], CxmData(field["Default"]).decompress()))
        return dict(fields)

    def post(self, data=None):
        if data is None:
            data = self.defaults
        body = []
        for k, v in data.items():
            body.append({
                "ParamName": k,
                "InnerThree": {
                    "0": [
                        {
                            "type": "System.String",
                            "data": CxmData.compress(v).decode()
                            }
                        ]
                    }
                }
                )
        return requests.post(self.url + self.endpoint_post, json={
            "pointer": GHDEPLOYPATH + self.target, "values": body},
                             headers=self.headers, timeout=1200)

    def solve(self, data=None):
        response = self.post(data=data)
        return CxmData(
            list(response.json()["values"][0]["InnerTree"].values())[0][0]["data"].replace('"', "")).decompress()


class GhDecore(SlimGHRequest):

    def __call__(self, f):
        self.name = f.__name__

        @wraps(f)
        def wrapped_call(slf, **data):
            if data is None:
                data = self.defaults

            result = self.solve(data=data)
            return f(slf, result
                     )

        return wrapped_call
