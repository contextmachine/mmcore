#  Copyright (c) 2022. Computational Geometry, Digital Engineering and Optimizing your construction processe"

from __future__ import annotations

import json

import pandas as pd

from ...connectors.gzjson import gzip_encoder
from ...mm.meta import ItemEncoder, MetaItem


class _Item(metaclass=MetaItem, encoder=ItemEncoder):
    __default_keys__ = dict()

    def __init__(self, *args, **kwargs):
        self.version = 0

        super().__init__()
        self.encoder = self.__class__.encoder
        self.__call__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.version += 1
        return self

    @property
    def series(self):
        data = dict()
        data |= self.default_fields
        data |= self.dct["metadata"]
        return pd.Series(data)

    @property
    def uid(self):
        return hex(id(self))

    def dumps(self, use_gzip=True, **kwargs):
        if use_gzip:
            return gzip_encoder(json.dumps(self, cls=self.encoder, **kwargs))
        else:
            return json.dumps(self, cls=self.encoder, indent=3, **kwargs)


class BaseItem(_Item, metaclass=MetaItem):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)
