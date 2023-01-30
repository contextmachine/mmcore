# Это экспериментальная фича предполагающая возможность глобальной конфигурации модулей в разных проектах
# В этом случае mmcore может управляться единым файлом спецификации
# Это может быть полезно для кастомный настройки переменных окружения,
# но в первую очередь для массивной генерации классов реализующие связывающие апи
# перенос всего генеративного кода в core делает его фабрикой модулей.

# тем не менее у меня нет уверенности что проектный сетап должен происходить на стороне mmcore
import json
from typing import ContextManager

from mmcore.baseitems import descriptors
from mmcore.collection import multi_description


class VarString(str):
    def __new__(cls, *args, **kwargs):
        # target form "{$"

        s = super().__new__(*args, **kwargs)
        s = s.replace("${", "{os.getenv('", -1).replace("}", "')}", -1)

        return s


class _target_getitem(descriptors.NoDataDescriptor):

    def __init__(self, data):
        super().__init__()
        self._data = data

        self._bundles = multi_description.ElementSequence(self._data["bundles"])

    def __get__(self, instance, owner):
        if self.path is None:
            self.path = self.name

        return self._bundles.search_from_key_value(self.path, instance.target)

    _path = None

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        self._path = value


class BundleConfigs(descriptors.NoDataDescriptor):
    def __init__(self, data):
        self.data = data
        self._bundles = multi_description.ElementSequence(self.data["bundles"])

    def __get__(self, instance, owner): ...


class ConfigJsonDriver(ContextManager):
    def __init__(self, target, path="mmconfig.json"):
        self.target = target
        self.path = path

    def __enter__(self):
        with open(self.path, "r") as file:
            data = json.load(file)
            self.target.bundle = BundleConfigs(data)

        return data

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class MmConfigType(type):
    @classmethod
    def __prepare__(mcs, name, bases, driver=ConfigJsonDriver, driver_kwargs=None, **kws):
        if driver_kwargs is None:
            driver_kwargs = {}
        ns = dict(super().__prepare__(name, bases))
        with driver(name, **driver_kwargs) as mmconfig:
            ns["mmconfig"] = mmconfig

        return ns

    def __new__(mcs, name, bases, attrs, **kws):
        return super().__new__(name, bases, attrs, **kws)
