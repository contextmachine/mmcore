from functools import wraps
from typing import Any

import requests
from compute_rhino3d import Util
from cxmdata import CxmData

from mmcore.addons.compute import ComputeRequest, secrets


class GHRequest(ComputeRequest):
    def __init__(self, definition_name="example.gh"):
        super().__init__()
        self.definition_name = definition_name

    _address = secrets['RHINO_COMPUTE_GH_DEPLOY_PATH']

    @property
    def address(self):
        return self._address

    @address.setter
    def address(self, value):
        self._address = value

    def get(self, address=None):
        return super().get(address=f"io?pointer={self.address if address is None else address}/{self.definition_name}")

    def post(self, data, address=None):
        return requests.post(Util.url + "grasshopper", json={
            "pointer": f"{self.address if address is None else address}/{self.definition_name}", "values": data},
                             timeout=1200)

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
        return CxmData(self.get(address=self.address).json()["Inputs"][0]["Default"]).decompress()

    def post(self, data=None, address=None):
        if data is None:
            data = self.defaults

        return super().post(
            [{
                "ParamName": "input",
                "InnerThree": {
                    "0": [
                        {
                            "type": "System.String",
                            "data": CxmData.compress(data).decode()
                            }
                        ]
                    }
                }],
            address=address
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

    def __init__(self, definition_name="example.gh"):
        super().__init__(definition_name=definition_name)

    @property
    def defaults(self) -> Any:
        fields = []
        for field in self.get(address=self.address).json()["Inputs"]:
            fields.append((field["Name"], CxmData(field["Default"]).decompress()))
        return dict(fields)

    def post(self, data=None, address=None):
        if data is None:
            data = self.defaults
        dt = []
        for k, v in data.items():
            prm = {
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
            dt.append(prm)
        return super().post(dt, address=address)

    def solve(self, *args, **kwargs):
        response = self.post(*args, **kwargs)
        return CxmData(
            list(response.json()["values"][0]["InnerTree"].values())[0][0]["data"].replace('"', "")).decompress()


class GhDecore(AdvanceGHRequest):

    def __call__(self, f):

        @wraps(f)
        def wrapped_call(slf, *args, **kwargs):
            dd = {}
            res = f(slf, *args, **kwargs)

            for k, v in self.defaults.items():

                if k in res.keys():
                    dd[k] = res[k]
                else:
                    dd[k] = v

            return self.solve(dd, address=self.address)

        return wrapped_call
