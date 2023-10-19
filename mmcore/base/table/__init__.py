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


import copy

import numpy as np


class Index:

    def __init__(self, keys1, keys2):

        self.keys1, self.keys2 = dict(zip(keys1, range(len(keys1)))), dict(zip(keys2, range(len(keys2))))
        self.tb = np.zeros((len(self.keys1), len(self.keys2)), dtype=int)

    def first_keys(self):
        return self.keys1.keys()

    def second_keys(self):
        return self.keys2.keys()

    def remove_link(self, k1, k2):
        self.tb[self.keys1[k1], self.keys2[k2]] = 0

    def add_link(self, k1, k2):
        self.tb[self.keys1[k1], self.keys2[k2]] = 1

    def __getitem__(self, k):
        if k in self.keys1:
            return list(filter(lambda x: self.tb[self.keys1[k], self.keys2[x]], self.keys2.keys()))
        else:
            return list(filter(lambda x: self.tb[self.keys1[x], self.keys2[k]], self.keys1.keys()))

    def __setitem__(self, k, v):
        if all([k in self.keys1, v in self.keys2]):
            self.tb[self.keys1[k], self.keys2[v]] = 1
        elif k not in self.keys1:

            self.add_key1(k)

        elif v not in self.keys2:

            self.add_key1(v)
        else:
            self.add_key1(k)
            self.add_key2(v)
            self[k] = v

    def add_key1(self, k1):
        n = len(self.keys1)
        self.keys1[k1] = n

    def add_key2(self, k2):
        n = len(self.keys2)
        self.keys2[k2] = n

    def __delitem__(self, k):

        if k in self.keys1:

            self.tb[self.keys1[k], ...] = 0

        else:

            self.tb[..., self.keys2[k]] = 0

    def __iter__(self):
        return ((k, self[k]) for k in self.keys1)

    @classmethod
    def fromdict(cls, data=None, /, **kwargs):
        if data is None:
            data = dict()
        data |= kwargs
        keys1 = data.keys()
        keys2 = set(data.values())
        self = cls(keys1, keys2)
        for key, val in data.items():
            self[key] = val
        return self

    def find_link(self, *pair):
        k1, k2 = pair
        return self.tb[self.keys1[k1], self.keys2[k2]]

    def transpose(self):
        new = copy.copy(self)
        new.keys1 = self.keys2
        new.keys2 = self.keys1
        new.tb = self.tb.T
        return new

    def __contains__(self, item):
        return bool(self.find_link(*item))


class IndexCounter(Index):
    """
    filter(lambda x: any((i.items[x].get('mount'), not i.items[x].get('stub'))), i.items)
    filter(lambda x: all((i.items[x].get('mount'), not i.items[x].get('stub'))), i.items)
    """

    def __init__(self, keys1, keys2):
        super().__init__(keys1, keys2)
        self.items = dict()

    def add_link(self, k1, k2):
        super().add_link(k1, k2)

    def remove_link(self, k1, k2):
        super().remove_link(k1, k2)

    def query(self, include=(), exclude=(), condition='all'):
        conds = {'all': all, 'any': any}
        return filter(lambda x: conds[condition]([self.items[x].get(inc[0]) == inc[1] for inc in include] + [
            type(inc[1])(self.items[x].get(inc[0])) != inc[1] for inc in exclude]), self.items)

    def add_link_item(self, k1, k2, item):
        pair = k1, k2
        if item not in self.items:
            self.items[item] = dict()
        if k1 in self.items[item]:
            self.remove_link_item(k1, self.items[item][k1], item)

        self.items[item][k1] = k2

        if self.tb[self.keys1[pair[0]], self.keys2[pair[1]]] > 0:

            self.tb[self.keys1[pair[0]], self.keys2[pair[1]]] += 1
        else:
            self.add_link(*pair)

    def remove_link_item(self, k1, k2, item):
        pair = k1, k2
        del self.items[item][k1]
        if self.tb[self.keys1[pair[0]], self.keys2[pair[1]]] > 0:

            self.tb[self.keys1[pair[0]], self.keys2[pair[1]]] -= 1
        else:
            self.remove_link(*pair)
