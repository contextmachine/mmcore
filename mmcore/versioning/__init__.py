from datetime import datetime


class Now(str):
    def __new__(cls, *a, **kw):
        d = datetime.now()
        instance = str.__new__(cls, d.isoformat())
        instance.__datetime__ = d
        return instance