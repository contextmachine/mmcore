from collections import defaultdict

from uuid import uuid4


def dict_to_schema(dct: dict):
    return {k: v.__class__ for k, v in dct.items()}


TABLES = dict()


class TableProxy:
    __slots__ = ['key', 'table']

    def __new__(cls, key, table):
        if key in table.proxies:
            return table.proxies[key]
        self = super().__new__(cls)
        self.table = table
        self.key = key
        table.proxies[key] = self
        return self

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        if isinstance(other, TableProxy):
            return all([other.key == other.key, other.table == self.table])

        else:
            return self.key == other

    def __getitem__(self, item):
        return self.table.rows[self.key][item]

    def __setitem__(self, item, value):
        self.table.rows[self.key][item] = value

    def update(self, kws=None, /, **kwargs):
        dict.update(self.table[self.key], kws, **kwargs)

    def __ror__(self, other):
        dict.__ror__(self.table.rows[self.key], other)

    def __iter__(self):
        return iter(self.table.rows[self.key].items())

    def items(self):
        return self.table.rows[self.key].items()

    def values(self):
        return self.table.rows[self.key].values()

    def keys(self):
        return self.table.rows[self.key].keys()

    def get(self, key, __default=None):
        return dict.get(self.table[self.key], key, __default)

    def __repr__(self):
        return f'{self.__class__.__qualname__}(key={self.key}, table={self.table})'

    def todict(self):
        dct = dict(self)
        dct2 = dict()

        for k, v in dct.items():
            if isinstance(v, TableProxy):
                dct2[k] = v.todict()
            else:
                dct2[k] = v
        return dct2


class Table:
    def __repr__(self):

        rpr = "{" + ", ".join(f"{k}: {v}" for k, v in self._reprschema.items()) + "}"
        if isinstance(self.pk, str):
            pkrp = f'"{self.pk}"'
        else:
            pkrp = f"{self.pk}"
        return f'{self.__class__.__qualname__}(pk={pkrp}, schema={rpr})'

    def __init_subclass__(cls, schema=None):
        cls.schema = dict(schema)

    def __new__(cls, pk=None, rows=None, schema=None, name=None, defaults=None, default_factory=lambda: None):

        self = super().__new__(cls)
        self.schema = dict(schema)
        self._reprschema = {_k: _v.__qualname__ for _k, _v in self.schema.items()}

        self.columns = list(self.schema.keys())
        self.pk = pk
        self.rows = dict()
        self.indexes = dict()
        self.proxies = dict()
        self.defaults = defaults if defaults else defaultdict(default_factory)
        for row in rows:
            if isinstance(pk, tuple):
                k = tuple(row[_k] for _k in pk)
            else:
                k = row[self.pk]
            self.rows[k] = row
        if name is None:
            name = uuid4().hex
        self.name = name
        TABLES[name] = self
        return self

    def __getitem__(self, item):

        return TableProxy(item, self)

    def __setitem__(self, item, value):

        self.rows[item] = value

    @classmethod
    def from_lists(cls, *args, rows=None, schema=None, **kwargs):
        return cls(*args, rows=[dict(zip(schema.keys(), row)) for row in rows], schema=schema, **kwargs)

    def update(self, kws=None, /, **kwargs):
        dict.update(self.rows, kws, **kwargs)

    def __ror__(self, other):
        dict.__ror__(self.rows, other)

    def __iter__(self):
        return iter(self.rows.items())

    def items(self):
        return self.rows.items()

    def values(self):
        return self.rows.values()

    def keys(self):
        return self.rows.keys()

    def get(self, key, __default=None):
        return self.rows.get(key, __default)
