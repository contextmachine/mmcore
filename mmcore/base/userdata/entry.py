"""
Model Entry Protocol
----
:doc:`
Спецификация механики поддерживаемой моделью. Хранится в `userData` в массиве `entries`.
Общее JSON представление (начиная с корня объекта)`
:code:`
{
  "name": "root",
  "userData": {
    "entries":[
    {
      "name": "update_props",
      "endpoint": {
        "protocol": "REST",
        "url": "..."
      }
    },
    {
      "name": "control_points",
      "endpoint": {
        "protocol": "WS",
        "url": "..."
       }
     }
   ]
  }
}  `
"""
import dataclasses
from enum import Enum


class EntryProtocol(str, Enum):
    """
    Перечисление веб-протоколов которые мы планируем использовать
    """
    REST = "REST"
    WS = "WS"
    GRAPHQL = "GRAPHQL"


@dataclasses.dataclass
class EntryEndpoint:
    """
    Спецификация определяющая эндпоинт бекенда для данной механики. url и протокол, не более.
    """
    protocol: EntryProtocol
    url: str


@dataclasses.dataclass
class Entry:
    """
    Entry
    -----

    Спецификация механики поддерживаемой моделью. Хранится в `userData` в массиве `entries`.

    Общее JSON представление (начиная с корня объекта):

    ```json
       {
          ...
          "userData": {
            "entries":[
              {
                "name": "update_props",
                "endpoint": {
                  "protocol": "REST",
                  "url": "..."
                }
              },
              {
                "name": "control_points",
                "endpoint": {
                  "protocol": "WS",
                  "url": "..."
                }
              }
            ]
          }
       }```

    """
    name: str
    endpoint: EntryEndpoint

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return hash(self.name)


def add_entry_safe(entries: list[Entry], entry: Entry) -> None:
    """

    :param entries : Массив `entries` из `userData`
    :type entries: list[Entry]

    :param entry :  Объект Entry который требующий добавления
    :type entry: Entry


    """
    if entry not in entries:
        entries.append(entry)


def add_entry(viewer_object, entry: Entry):
    add_entry_safe(viewer_object._user_data_extras['entries'], entry)


def add_props_update_support(viewer_object):
    if not viewer_object.has_entries:
        viewer_object.add_entries_support()

    viewer_object.add_entry(Entry(name="update_props",
                                  endpoint=EntryEndpoint(protocol=EntryProtocol.REST,
                                                         url=viewer_object.object_url + f"props-update/{viewer_object.uuid}")))
