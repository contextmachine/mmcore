from enum import Enum

from jinja2.nativetypes import NativeEnvironment

__all__ = ["ChartTypes", "Chart"]

chart_temp = """
{
    "id": "{{ id }}",
    "name": "{{ name }}",
    "type": "chart",
    "key": "{{ key }}",
    "colors": "{{ default }}",
    "require": ["{{ types }}"]
}"""


class ChartTypes(str, Enum):
    piechart = "piechart"
    linechart = "linechart"


def create_chart(key, colors="default", types=(ChartTypes.piechart, ChartTypes.linechart)):
    env = NativeEnvironment()
    ss = env.from_string(chart_temp)
    return ss.render(key=key,
                     colors=colors,
                     require=",".join(types),
                     id="-".join(("chart", key) + types),
                     name=key.capitalize() + " Chart")


class Chart(dict):
    def __init__(self, key, colors="default", types=(ChartTypes.piechart, ChartTypes.linechart)):
        dict.__init__(self, create_chart(key, colors=colors, types=types))
