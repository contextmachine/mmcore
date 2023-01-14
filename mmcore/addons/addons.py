from abc import ABCMeta
from typing import ContextManager


class FromSource:
    def __init__(self, name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            source = owner.source

        else:
            source = instance.__class__.source
        if source is None:
            exec(f"from . import * as source")
            source = "source"
        else:
            exec(f"import {source}")

        return eval(source).__dict__[self.name]


class SourceManager(ContextManager):

    def __init__(self, target):
        self.target = target

    def __call__(self, defaults, **kwargs):
        self.defaults = defaults
        self.kwargs = kwargs
        return self

    def __enter__(self):
        return lambda dep: self.target(dep, self.defaults, **self.kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(exc_type, exc_val, exc_tb)
        return False


class AddonBaseType(ABCMeta):
    @classmethod
    def __prepare__(mcs, name, bases, source_manager=None, deps=(), defaults=(), **kwargs):
        ns = dict(super().__prepare__(name, bases))
        with source_manager(defaults, **kwargs) as source:
            for name in deps:
                ns[name] = source(name)

            return ns

    def __new__(mcs, name, bases, dct, **kwargs):
        cls = super().__new__(mcs, name, bases, dct, **kwargs)
        return cls
