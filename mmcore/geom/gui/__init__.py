from enum import Enum
from typing import Any

import pydantic

from mmcore.baseitems.descriptors import ChartTemplate, DataView, Template
from ._gui import classes

dt = {}
for i, k in enumerate(classes):
    # eval(f"{k.__name__} = classes[{i}]")
    dt[k.__name__] = classes[i]


class GuiChart(str, Enum):
    linechart = dt['LineChart']
    piechart = dt['PieChart']
    doublechart = dt['DoubleChart']


class GuiTemplates(Template, Enum):
    line_chart = ChartTemplate("linechart")
    pie_chart = ChartTemplate("piechart")


class GuiStatProperty(pydantic.BaseModel):
    field: str
    chart: GuiChart


class GuiColors(str, Enum):
    default = "default"


GuiControls = dt["GuiControls"]


class UserDataGui(DataView):
    targets = []

    def __init__(self, *targets):
        super().__init__(*targets)
        self.targets = list(self.targets)

    def item_model(self, name: str, value: GuiChart | GuiControls):
        return value

    def data_model(self, instance, value: list[GuiChart | GuiControls] | None = None):
        if (value is None) or (value == []):
            pass
        else:
            return value


class UserCharts(DataView):
    targets = []

    def __init__(self, targets):
        super().__init__(*[k for k, v in targets])
        self.dt = dict(targets)

    def item_model(self, name: str, value: Any):
        print(name, GuiChart)
        v = self.dt[name]
        return dt[v](key=name).dict()

    def data_model(self, instance, value: list[GuiChart | GuiControls] | dict | None = None):
        if (value is None) or (value == []):
            pass
        else:
            return value
