from typing import Any

import requests
from compute_rhino3d import Util
import setupsecrets
from cxmdata import CxmData

secrets = setupsecrets.setup_secrets(update=True)
Util.url = "http://" + secrets['RHINO_COMPUTE_URL'] + f":{secrets['RHINO_COMPUTE_PORT']}/"

import compute_rhino3d
from compute_rhino3d import \
    Grasshopper, \
    GeometryBase, \
    Curve, \
    Brep, \
    VolumeMassProperties, \
    BrepFace, \
    BezierCurve, \
    NurbsCurve, \
    Extrusion, \
    Intersection, \
    AreaMassProperties, \
    NurbsSurface, \
    Surface, \
    SubD, \
    Mesh
from mmcore.utils.pydantic_mm.models import InnerTreeItem


class ComputeRequest:
    def get(self, address):
        return requests.get(Util.url + address)

    def post(self, data, address):
        return requests.get(Util.url + address, json=data)


class GrashopperRequest(ComputeRequest):
    def __init__(self, definition_name="example.gh"):
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
            "pointer": f"{self.address if address is None else address}/{self.definition_name}", "values": [data]})


class StandardGrasshopperRequest(GrashopperRequest):
    """
    Предполагается что определение имеет один вход и один выход "input/output" с типами данных cxmdata/cxmdata.
    Этого должно быть достаточно в 90% случаев .

    Если в метод post не поступает аргумент data, определение будет вычисляться с input по умолчанию (если такое
    значение существует)
    """

    @property
    def defaults(self) -> Any:
        return CxmData(self.get(address=self.address).json()["Inputs"][0]["Default"]).decompress()[0]

    def post(self, data=None, address=None):
        if data is None:
            data = self.defaults

        response = super().post(
            {
                "ParamName": "input",
                "InnerThree": {
                    "0": [
                        {
                            "type": "System.String",
                            "data": CxmData.compress(data).decode()
                            }
                        ]
                    }
                },
            address=address
            )

        return CxmData(
            list(response.json()["values"][0]["InnerTree"].values())[0][0]["data"].replace('"', "")).decompress()


