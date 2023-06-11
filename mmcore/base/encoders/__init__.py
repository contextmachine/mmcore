import dataclasses
from json import JSONEncoder


class MMCOREEncoder(JSONEncoder):
    def default(self, o):
        try:
            iterable = iter(o)
        except TypeError:
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
            elif hasattr(o, "todict"):
                return o.todict()
        else:
            return list(iterable)
        # Let the base class default method raise the TypeError
        return JSONEncoder.default(self, o)
