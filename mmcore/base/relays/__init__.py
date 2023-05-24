import uuid

relaydict=dict()
import os
#print(os.environ)

from collections import namedtuple
namedtuple("RelayTuple",["name","resolver"])
from mmcore.services.redis import connect
rconn=connect.get_cloud_connection()
class BaseD:
    prefix: str
    def __init__(self, handler,**kwargs):
        self._handler=handler
        self.__dict__|=kwargs
    def key(self, obj):
        return f"{self.prefix}:{obj.uuid}"

    def __get__(self, obj, typ):
        if obj is None:
            return self._handler[rconn.hget(self.key(typ), self._name)]

        return self._handler[rconn.hget(self.key(obj), self._name)]
    def __set_name__(self, owner, name):
        self._name = "_"+name
    def __set__(self, obj, val):

        if isinstance(val, dict):
            v=val["uuid"]
        elif hasattr(val, "uuid"):
            v=val.uuid
        else:
            v=val
            self.validate(v)

        rconn.hset(self.key(obj), self._name, v)

    def validate(self, val):
        if isinstance( val, str):
            assert len(uuid.uuid4().__str__()), len(val)
        else:
            raise TypeError(f"{self.__class__.__name__}:{self._name} "
                            f"Ожидалось значение с типом str (uuid)\n\t{val}")

class DataD:


    #key=staticmethod(lambda  obj:f"{prefix}:{obj.uuid}")

    @classmethod
    def key(cls, obj):
        return f"{cls.prefix}:{obj.uuid}"
    def __init__(self, *, type_map, relays):
        self._type_map = type_map
        self._relays=relays

    def __get__(self, obj, typ):
        if obj is None:
            return self._defaul
        rconn.hget(obj.uuid, self._name)

        val=getattr(obj, self._name,self._default)

        resolver = rconn.hget(self.key(obj),[self._name[1:]])
        resolver(obj)
        return getattr(obj, self._name, self._default)


    def __set__(self, obj, value):
        setattr(obj, self._name, int(value))

    def __set_name__(self, owner, name):
        self._name = "_" + name
