import functools
import json
import numpy as np
from dataclasses import asdict, is_dataclass


class CompatEncoder(json.JSONEncoder):
    def default(self, o):

        if isinstance(o, set):
            return list(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, type):
            return o.__name__


        elif hasattr(o, 'to_dict'):
            return o.to_dict()
        elif is_dataclass(o):
            return asdict(o)
        else:
            return json.JSONEncoder.default(self, o)


# Crazy but Sugary Monkey Patch

_wrapped_dumps = json.dumps


@functools.wraps(_wrapped_dumps)
def dumps(obj, *, skipkeys=False, ensure_ascii=True, check_circular=True, allow_nan=True, cls=CompatEncoder,
          indent=None, separators=None, default=None, sort_keys=False, **kw):

    return _wrapped_dumps(obj, skipkeys=skipkeys, ensure_ascii=ensure_ascii, check_circular=check_circular,
                          allow_nan=allow_nan, cls=cls, indent=indent, separators=separators, default=default,
                          sort_keys=sort_keys, **kw
                          )


json.dumps = dumps
