#  Copyright (c) 2022. Computational Geometry, Digital Engineering and Optimizing your construction processe"
import numpy as np

from ..baseitems import Item


def args_flatten(arg, *args):
    arr = np.asarray((arg,)).flatten()
    for barr in args:
        arr = np.concatenate([arr, np.asarray((barr,)).flatten()])
    return arr


class ReplaceMapping:
    replacements = {
        "-": "_",
        " ": "__",
        ".": "___"
    }

    def __init__(self):
        super(ReplaceMapping, self).__init__()
        self.replacements = self.__class__.replacements

    @classmethod
    def replace(cls, string: str):
        for k, v in cls.replacements.items():
            if k in string:
                string = string.replace(k, v)
        return string

    @classmethod
    def replace_back(cls, string: str):
        for k, v in cls.replacements:
            if v in string:
                string = string.replace(v, k)
        return string


class rm(ReplaceMapping):
    def __set_name__(self, owner, name):
        self.name = self.replace(self.name)

    def __get__(self, instance, owner):
        return instance.__dict__[self.replace(self.name)]

    def __set__(self, instance, v):
        instance.__dict__[self.replace_back(self.name)] = v


class DotView(Item, ReplaceMapping):
    """
    >>> class R2(ReplaceMapping):
    ...     replacements = {
    ...             "a": "AAAA",
    ...             "b": "NNN",
    ...             "c": "CCC"
    ...         }


    >>> class BV(DotView, R2):
    ...     ...


    >>> d = BV(**dict(a=6, b=dict(t=0, d=6)))

    >>> d.AAAA
    6
    """

    def __init__(self, dct=None, **kwargs):
        super().__init__(**kwargs)
        if dct is None:
            dct = dict()
        kwargs |= dct

        self.do_traverse(self, kwargs)

    @classmethod
    def replace(cls, string: str):
        return super(DotView, cls).replace(string)


    @classmethod
    def replace_back(cls, string: str):
        return super(DotView, cls).replace(string)

    @classmethod
    def do_traverse(cls, obj, dct):
        for k, v in dct.items():

            if isinstance(v, dict):
                new_item = type(k, (DotView,), v)

                obj.__call__(**{new_item.replace(k): new_item})

                cls.do_traverse(new_item, v)
            else:
                obj.update(**{cls.replace(k): v})


class TraverseDict(dict):
    dict_attr = "__dict__"

    def __init__(self):

        super().__init__()

    def __getitem__(self, item):
        return dict.__getitem__(self, item)

    def __setitem__(self, item, v):
        dict.__setitem__(self, item, v)

    @classmethod
    def replace_back(cls, string: str):
        return super(DotView, cls).replace(string)

    @classmethod
    def do_traverse(cls, obj, dct):
        for k, v in dct.items():

            if isinstance(v, dict):
                new_item = type(k, (DotView,), v)

                obj.__call__(**{new_item.replace(k): new_item})

                cls.do_traverse(new_item, v)
            else:
                obj.update(**{cls.replace(k): v})
