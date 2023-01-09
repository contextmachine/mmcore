from types import ModuleType

import requests

from mmcore.addons.addons import AddonBaseType


class _ComputeRequest(metaclass=AddonBaseType, source="mmcore.addons.compute", from_source=("Util",)):
    Util: ModuleType

    def __init__(self):
        object.__init__(self)

    @property
    def compute_url(self):
        return self.Util.url

    def get(self, address):
        return requests.get(self.compute_url + address)

    def post(self, address, data):
        return requests.get(self.compute_url + address, json=data)


class ComputeRequest(_ComputeRequest):
    def __init__(self):
        super().__init__()

    def get(self, address):
        return super().get(address)

    def post(self, address, data):
        return super().post(address, data)


class ComputeGeometryRequest(ComputeRequest):
    ...
