import datetime


dt = datetime.datetime.now()
import time

timezone = datetime.timezone(datetime.timedelta(hours=-(time.timezone // 60 // 60)), name=time.tzname[0])
timezone.utcoffset(dt)


def get_datatime():
    return datetime.datetime.now(tz=timezone)

class Now(str):
    __timezone__ = timezone

    def __new__(cls, *args, cb=lambda dt: dt.isoformat()):
        if len(args) == 1:
            return cls.from_isoformat(args[0])
        else:
            dtm = get_datatime()
            instance = str.__new__(cls, cb(dtm))
            instance.__datetime__ = dtm

            return instance

    @classmethod
    def from_isoformat(cls, s):

        instance = str.__new__(cls, datetime.datetime.fromisoformat(s))
        instance.__datetime__ = datetime.datetime.fromisoformat(s)
        return instance

    def __hash__(self):
        return dt.__hash__()

    def __eq__(self, o):
        return dt.__eq__(o.__datetime__)

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()})"
