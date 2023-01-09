from abc import ABCMeta


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


class AddonBaseType(ABCMeta):
    @classmethod
    def __prepare__(mcs, name, bases, source: str = None, from_source: tuple[str] = (), **kwargs):
        ns = dict(super().__prepare__(__name=name, __bases=bases))
        if source is None:
            for base in bases:
                if hasattr(base, "source"):
                    if base.source is not None:
                        source = base.source
                        break
                else:
                    continue
        ns["source"] = source
        for name in from_source:
            ns[name] = FromSource(name)
        ns |= kwargs
        return ns

    def __new__(mcs, name, bases, dct, **kwargs):
        cls = super().__new__(mcs, name, bases, dct)
        return cls
