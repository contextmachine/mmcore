import json
import uuid

from mmcore.base.sharedstate import serve

pr_redirects = dict()


class DbType(type):
    ...


class Db(dict):
    _redirects = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._redirects = dict()

    @property
    def redirects(self):
        return self._redirects

    @redirects.setter
    def redirects(self, v):
        self._redirects = v

    def get_object_tree(self, key):
        dct = dict()
        for k, v in self[key].items():
            if k in self._redirects[key]:
                dct[k] = self.get_object_tree(self._redirects[key][k])
            else:
                dct[k] = v
        return dct


objects = Db()

props_db = Db()

params_db = Db()


def is_link(k, token="$"):
    if isinstance(k, str):
        if k.startswith(token):
            return True
    return False


def clear_token(val, token="$"):
    return val.replace(token, "")


class ItemProps:
    __slots__ = "key", "db"

    def __init__(self, key, db=None, **kwargs):

        if db is None:
            db = props_db
        self.db = db
        self.key = key
        if self.key not in self.db:
            self.db[self.key] = dict()
            self.db.redirects[self.key] = dict()
        for k, v in kwargs.items():
            self[k] = v

    def setlink(self, field, key):

        self.db[self.key][field] = key
        self.db.redirects[self.key][field] = clear_token(key)

    def __getitem__(self, k):
        if k in self.db.redirects[self.key]:
            return ItemProps(self.db.redirects[self.key][k])[k]
        else:

            return self.db[self.key][k]

    def __setitem__(self, k, v):
        if is_link(v):
            self.setlink(k, v)
        elif k in self.db.redirects[self.key]:
            ItemProps(self.db.redirects[self.key][k])[k] = v
        else:
            self.db[self.key][k] = v

    def __delitem__(self, k):
        if k in self.db.redirects[self.key]:
            self.db.redirects[self.key].__delitem__(k)
        self.db[self.key].__delitem__(k)

    def update(self, kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def __repr__(self):
        return f'{self.__class__.__qualname__}({dict(self)})'

    def __iter__(self):
        return ((k, self.__getitem__(k)) for k in self.db[self.key].keys())

    def __setattr__(self, key, value):

        if not (key.startswith("_") or key in self.__slots__):
            self.__setitem__(key, value)

        else:
            object.__setattr__(self, key, value)

    def __getattr__(self, key):

        if key.startswith("_"):

            return object.__getattribute__(self, key)
        elif key in self.__slots__:
            return object.__getattribute__(self, key)
        elif key in self.db.redirects[self.key].keys():
            return self.__getitem__(key)
        else:
            return object.__getattribute__(self, key)


class ParamsProps(ItemProps):
    def __init__(self, key, db=None, **kwargs):
        if db is None:
            db = params_db
        super().__init__(key, db=db, **kwargs)

    def __getitem__(self, k):
        if k in self.db.redirects[self.key]:
            return ParamsProps(self.db.redirects[self.key][k])
        else:

            return self.db[self.key][k]

    def __setitem__(self, key, value):
        if is_link(value):
            self.setlink(key, value)
        elif key in self.db.redirects[self.key]:
            ParamsProps(self.db.redirects[self.key][key]).update(value)
        else:
            self.db[self.key][key] = value


class MyBaseEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ItemProps):
            return dict(o)
        elif isinstance(o, MyBaseObject):
            return o.todict()
        else:
            return super().default(o)


class MyBaseObject:
    def __init__(self, key=None, properties=None, params=None, **kwargs):
        super().__init__()

        if key is None:
            key = uuid.uuid4().hex
        if properties is None:
            properties = dict()
        if params is None:
            params = dict()
        self.key = key
        objects[key] = self
        objects.redirects[self.key] = dict()

        self.params = ParamsProps(self.key, **params)
        self.properties = ItemProps(self.key, **properties)
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    def __getattr__(self, item):
        if item.startswith("_"):
            return object.__getattribute__(self, item)
        elif item in objects.redirects[object.__getattribute__(self, 'key')]:
            return objects[objects.redirects[self.key][item]]
        else:
            return object.__getattribute__(self, item)

    def __setattr__(self, key, value):
        self.__dict__[key] = value
        if is_link(value):
            objects.redirects[self.key][key] = clear_token(value)

    def recursive_update(self, data: dict):
        def upd_obj(obj, dt):
            for k, v in dt.items():

                if isinstance(obj, dict):
                    if isinstance(v, dict):
                        upd_obj(obj[k], v)
                    else:
                        obj[k] = v
                else:
                    if isinstance(v, dict):

                        upd_obj(getattr(obj, k), v)


                    else:
                        setattr(obj, k, v)

        upd_obj(self, data)

    def todict(self):
        return dict((k, self.__getattr__(k)) for k in filter(lambda x: not x.startswith('_'), self.__dict__.keys()))

    def tojson(self):

        return json.dumps(self.todict(), cls=MyBaseEncoder, indent=2)


example_api = serve.create_child("/example_api")


@example_api.get("/items/{key}")
def get_item(key: str):
    return objects[key].todict()


@example_api.post("/items/{key}")
def post_item(key: str, data: dict):
    if key in objects:

        obj = objects[key]
        obj.recursive_update(data)
    else:
        obj = MyBaseObject(key, **data)
    return obj.todict()


@example_api.get("/keys")
def get_keys():
    return list(objects.keys())


@example_api.get("/items")
def get_items():
    return [obj.todict() for obj in objects.values()]


@example_api.get("/properties")
def get_props():
    return [ItemProps(k) for k in props_db.keys()]


@example_api.get("/properties/{key}")
def get_props_by_key(key: str):
    return objects[key].properties


@example_api.post("/properties/{key}")
def post_props_by_key(key: str, data: dict):
    ItemProps(key).update(data)
    return objects[key].todict()


@example_api.post("/properties/{key}/{field}")
def post_prop_by_key(key: str, field: str, data: dict):
    ItemProps(key)[field] = data["data"]

    return objects[key].todict()


@example_api.put("/properties/{key}")
def put_props_by_key(key: str, data: dict):
    props_db[key] = data
    return objects[key].todict()


@example_api.delete("/properties/{key}")
def del_props_by_key(key: str):
    props_db[key] = dict()
    return objects[key].todict()


@example_api.get("/redirects/{name}")
def get_props_redirects(name: str):
    return {"props": props_db, "objects": objects}[name].redirects


b = ItemProps("object1", tag="C", mount=False)

o = MyBaseObject("object2",
                 name="Object Two",
                 baz=MyBaseObject("object1", params=dict(x=4, y=8)),

                 properties={'tag': "B", "mount": "$object4"}
                 )
o4 = MyBaseObject("object4",
                  name="Object Four",
                  child="$object3",

                  properties={'tag': "A", 'mount': False}
                  )
o2 = MyBaseObject("object3",
                  name="Object Three",
                  baz="$object2",
                  params=dict(start="$object1", end=5),
                  properties={'tag': '$object1', 'mount': True}
                  )
serve.start()
