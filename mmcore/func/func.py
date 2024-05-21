import numpy as np
import functools

def vectorize(signature=None, excluded=None, **kws):
    def decorate(fun):
        if "doc" not in kws:
            kws["doc"] = fun.__doc__
        if excluded is not None:
            kws["excluded"] = excluded
        if signature is not None:
            kws["signature"] = signature
        vecfun = np.vectorize(fun, **kws)

        @functools.wraps(fun)
        def wrap(*args, **kwargs):
            return vecfun(*args, **kwargs)

        return wrap

    return decorate
