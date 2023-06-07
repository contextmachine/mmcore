import types
import typing
from collections import abc


class Rule:
    """
    >>> r1 = Rule(lambda ctx, x, y: y==ctx[x])
    >>> constrains = ('b', 2),('a', 1)
    >>> r2 = Rule(rule=lambda ctx, x, y: y==ctx[x])
    >>> r3 = r2 + r1
    >>> col=[{"a":1,"b":2}, {"a":6,"b":3}, {"a":22,"b":2}, {"a":1,"b":3}]
    >>> *res,= filter(lambda i:r3(i, *constrains), col)
    >>> res
    [{'a': 1, 'b': 2}]
    """
    rule: typing.Union[types.FunctionType, types.LambdaType, types.MethodType]
    pair: tuple[...]


    def __init__(self, rule):

        self.rule = rule

    def __call__(self, ctx, *pair):
        return self.rule(ctx, *pair)

    def __and__(self, other):
        return Rule(lambda ctx, x, y: self(ctx, *x) and other(ctx, *y))

    def __or__(self, other):
        return Rule( lambda ctx, x, y: self.rule(ctx, *x) or other.rule(ctx, *y))

    def __add__(self, other):
        return Rule(lambda ctx, x, y: self.rule(ctx, *x) and other.rule(ctx, *y))

    def __sub__(self, other):
        return Rule(lambda ctx, x, y: self.rule(ctx, *x) and not other.rule(ctx, *y))


