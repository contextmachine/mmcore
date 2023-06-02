import time

import os

from functools import wraps


def curry(func):
    """
    >>> @curry
    ... def foo(a, b, c):
    ...     return a + b + c
    >>> foo(1)
    <function __main__.foo>
    """

    @wraps(func)
    def curried(*args, **kwargs):
        if len(args) + len(kwargs) >= func.__code__.co_argcount:
            return func(*args, **kwargs)

        @wraps(func)
        def new_curried(*args2, **kwargs2):
            return curried(*(args + args2), **dict(kwargs, **kwargs2))

        return new_curried

    return curried




