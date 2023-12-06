import functools
import inspect
from collections.abc import Callable

import itertools
import time

systems = dict()
debug_log = dict()


class EcsSystem(Callable):
    def __init__(self, fun, debug=False):
        self.fun = fun
        self.signature = inspect.signature(fun)
        self.tmm = 0.0
        self.calls = 0
        self.debug = debug
        self.counter = itertools.count()
        debug_log[self.fun.__name__] = {}
        self._call = {
            1: self.debug_call,
            0: self.fun
        }

    @property
    def ecs_params(self):
        return self.signature.parameters

    def __call__(self, *args, **kwargs):
        return self._call[self.debug](*args, **kwargs)

    def debug_call(self, *args, **kwargs):
        self.calls = next(self.counter)

        s = time.time()
        res = self.fun(*args, **kwargs)
        end = time.time() - s
        self.tmm += end

        self.log = {'calls': self.calls, 'time': self.tmm}

        return res

    @property
    def log(self):
        return debug_log[self.fun.__name__]

    @log.setter
    def log(self, v):
        debug_log[self.fun.__name__] = v


def system(debug=True):
    def decorate(fun):
        return functools.wraps(fun)(EcsSystem(fun, debug=debug))

    return decorate
