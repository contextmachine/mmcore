from enum import Enum
from typing import Any

from mmcore.baseitems.descriptors import DataView
from .chart import Chart, ChartTypes, create_chart


class GuiColors(str, Enum):
    default = "default"


GuiControls = {"type": "controls",
               "data": {

               },
               "post": {
                   "endpoint": "https://api.contextmachine.online/api/any",
                   "mutation": {
                       "scene": {
                           "where": {
                               "userData": {
                                   "properties": {
                                       "": {
                                           "_eq": "original"
                                       }
                                   }
                               }
                           }
                       }
                   }
               }
               }

GuiModel = list[GuiControls, GuiColors, dict]



class UserCharts(DataView):
    targets = []

    def __init__(self, *targets):
        super().__init__(*[k for k, v in targets])
        self.dt = dict(targets)

    def item_model(self, name: str, value: Any):

        v = self.dt[name]
        return v(key=name).dict()

    def data_model(self, instance, value: GuiModel | dict | None = None):
        if (value is None) or (value == []):
            pass
        else:
            return value
