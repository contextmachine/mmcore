#  Copyright (c) 2022. Computational Geometry, Digital Engineering and Optimizing your construction processe"

# some version control
from __future__ import annotations

import functools
import inspect
from abc import ABCMeta
from functools import partial, wraps
from typing import Any, Callable, Optional

function_type = type(lambda: None)

from ...mm.descriptors import HookDescriptor

from json import JSONEncoder


def curry(func):
    @functools.wraps(func)
    def curried(*args, **kwargs):
        if len(args) + len(kwargs) >= func.__code__.co_argcount: return func(*args, **kwargs)
        return (lambda *args2, **kwargs2: curried(*(args + args2), **dict(kwargs, **kwargs2)))

    return curried


def metalogger(method):
    def log_wrapper(slf, *args, **kwargs):
        print(
            f"\n\n{slf.__class__.__name__}.{method.__name__}(self, *args, **kwargs) --> ...\n{'-' * 140}\n\tself: {slf}\n\targs: {args} \n\tkwargs: {kwargs}")
        return method(slf, *args, **kwargs)

    return log_wrapper


class CallBind(Callable):
    def __init__(self, func):
        self._func = func

    @metalogger
    def __set_name__(self, owner, name="__call__"):
        owner.__call__ = self
        self.name = name
        self.owner = owner

    def __call__(self, instance, *args, **kwargs):
        kwargs |= dict(zip(list(instance.__fields__)[:len(args)], args))
        return self._func(instance, **kwargs)

    @metalogger
    def __get__(self, instance, owner):
        return functools.wraps(self._func)(partial(self, instance))


class DefaultFied:
    @metalogger
    def __set_name__(self, owner, name):
        self.name = name
        self.default = owner.__default_fields__[self.name]
        self.private_name = "_" + self.name
        if self.name in owner.__annotations__.keys():
            print(owner.mcs.types)
            self.hint = owner.__annotations__[self.name]
        else:
            self.hint = Any

    @metalogger
    def __get__(self, obj, objtype=None):

        return obj.default_fields[self.name]

    @metalogger
    def __set__(self, obj, value):
        if self.validate(value):

            obj.default_fields[self.name] = value

        else:
            try:
                obj.default_fields[self.name] = self.hint(value)
            except:
                raise ValueError

    def validate(self, value):
        if self.hint is Any:
            return True
        else:
            if isinstance(value, self.hint):
                return True
            else:
                return False


class Dct(dict):
    exclude = ["default_fields", "encoder"]
    metadata = ["uid"]

    @metalogger
    def __get__(self, obj, objtype=None):
        dct = dict()

        dct |= obj.default_fields
        dct_ = dict()

        self.trav(obj.default_fields, dct_, obj=obj)

        dct_['metadata'] = dict()
        for k in self.metadata:
            dct_['metadata'][k] = getattr(obj, k)
        self.trav(obj.__dict__, dct_['metadata'], obj=obj)
        return dct_

    @metalogger
    def trav(self, dct, dct_, obj):
        for k, v in dct.items():
            if k in self.exclude:
                continue
            else:
                if hasattr(v, "dct"):
                    newdict = dict()
                    dct_[k] = newdict

                    self.trav(v.dct, newdict, obj=obj)


                else:
                    dct_[k] = v


class CallBinder:
    @metalogger
    def __get__(self, instance, owner):
        def callwrapper(*args, **kwargs):
            d = dict(zip(owner.__default__.names[:len(args)], args))
            d |= kwargs
            return owner.__call(instance, **d)

        return callwrapper


class DF:
    @metalogger
    def __init__(self, k, v=None):
        self.name = k
        self.default_name = k
        self.default_value = v
        self.value = v


class DFF:
    def __set_name__(self, owner, name):
        self.name = name
        self.owner = owner

    def __get__(self, instance, owner):
        try:
            return getattr(owner, self.name)(instance)
        finally:
            maby_method = getattr(owner, self.name)
            print("maby method", maby_method)
            return wraps(maby_method)(partial(maby_method, instance))


class DFDescriptor:
    _table = None

    @metalogger
    def __set_name__(self, owner, name):

        self.name = name

    @metalogger
    def __get__(self, instance, owner):

        # Если дескриптор вызывается от owner.
        if (instance is None) & (owner is not None):
            return owner.__dict__[self.name]

        # Если дескриптор вызывается от instance.
        elif instance is not None:
            target = owner.__dict__[self.name]
            if isinstance(target, property):
                # Если под дескриптором property.
                return target.getter(instance)
            elif inspect.ismethod(target):
                # Если под дескриптором метод.
                # - Возвращается просто `functools.wraps(<method>)(functools.partial(<method>, instance)`
                #   где `functools.wraps` маскирует имя метода на привычное
                # - В этом случае методы, не требующие передачи переменных,
                #   кроме self, должны обрабатываться иначе.?
                return wraps(target)(partial(target, instance))
            else:
                # Если под дескриптором аттрибут(не метод).
                return instance.__dict__[self.name]
        else:
            raise KeyError

    @metalogger
    def __set__(self, instance, value):

        instance.__dict__[self.name] = value

    @property
    def table(self):
        return self._table

    @table.setter
    def table(self, v):
        self._table = v


class DescTable(dict):
    """
    Обертка <class>.__dict__
    """
    exclude = ["exclude", "mmap"]

    def __set_name__(self, owner, name):
        self.name = name

        try:
            iter(owner.__fields__)
        except:

            owner.__fields__ = []
        self.owner = owner  # оборачиваемый класс
        # Проверяем наличие аттрибута у оборачиваемого класса
        # и то что он итерируемый.

    def __get__(self, instance, owner):
        return getattr(instance, self.name)

    def __set__(self, instance, value):
        dct = getattr(instance, self.name)
        for k, v in value.items():
            try:
                dct.__setitem__(k, v)
            except:
                pass

    @metalogger
    def __init__(self):
        super().__init__()
        self.fields = []

    @metalogger
    def __setitem__(self, key: str, value: Optional[Any | None] = None):
        if key not in self.exclude:

            if key not in self.fields:
                self.fields.append(key)
                d = DFDescriptor()
                d.__set_name__(self.owner, key)

            dict.__setitem__(self, key, value)
        else:
            return KeyError

    @metalogger
    def __getitem__(self, item):
        if item in self.owner.__fields__:
            return dict.__getitem__(self, item)
        else:
            return KeyError("DescTable key error")

    @metalogger
    def __ior__(self, other):
        self.owner.__defaultdict__ |= other

    def as_dict(self):
        return dict.copy(self)


class ItemCall(CallBind):

    def __init__(self):
        super().__init__(self.func)

    @metalogger
    def func(self, instance, **kwargs):
        for k, v in kwargs.items():
            instance.__dict__[k] = v

        return instance


class MetaItem(ABCMeta):
    types = dict()
    tables = dict()
    exclude = ["dtype", "mcs", "exclude"]

    def __new__(mcs, name, base, attrs, **kws):
        print(attrs, "\n", kws)
        attrs |= dict(mmtype=name)
        try:
            attrs["__call__"] = CallBind(attrs["__call__"])
        except:
            attrs["__call__"] = ItemCall()
        attrs['mcs'] = mcs
        c = super().__new__(mcs, name, base, attrs, **kws)

        # Defaultdict
        d = DescTable()
        d.__set_name__(c, "__defaultdict__")
        c.__defaultdict__ = d

        for k, v in attrs.items():
            if inspect.ismethod(v) and not (k in c.exclude):
                c.__defaultdict__[k] = v

            else:
                if (k[0] != "_") and not (k in c.exclude):
                    try:
                        if (k in c.__annotations__.keys()) and (v is None):

                            c.__defaultdict__[k] = c.__annotations__[k]
                        else:
                            c.__defaultdict__[k] = v
                    except:
                        pass

                else:
                    pass

        mcs.types[name] = c
        return c


class ItemEncoder(JSONEncoder):

    def __init__(self, *, skipkeys=False, ensure_ascii=True, check_circular=True, allow_nan=True, sort_keys=False,
                 indent=None, separators=None, default=None):
        """Constructor for JSONEncoder, with sensible defaults.

            If skipkeys is false, then it is a TypeError to attempt
            encoding of keys that are not str, int, float or None.  If
            skipkeys is True, such items are simply skipped.

            If ensure_ascii is true, the output is guaranteed to be str
            objects with all incoming non-ASCII characters escaped.  If
            ensure_ascii is false, the output can contain non-ASCII characters.

            If check_circular is true, then lists, dicts, and custom encoded
            objects will be checked for circular references during encoding to
            prevent an infinite recursion (which would cause an RecursionError).
            Otherwise, no such check takes place.

            If allow_nan is true, then NaN, Infinity, and -Infinity will be
            encoded as such.  This behavior is not JSON specification compliant,
            but is consistent with most JavaScript based encoders and decoders.
            Otherwise, it will be a ValueError to encode such floats.

            If sort_keys is true, then the output of dictionaries will be
            sorted by key; this is useful for regression tests to ensure
            that JSON serializations can be compared on a day-to-day basis.

            If indent is a non-negative integer, then JSON array
            elements and object members will be pretty-printed with that
            indent level.  An indent level of 0 will only insert newlines.
            None is the most compact representation.

            If specified, separators should be an (item_separator, key_separator)
            tuple.  The default is (', ', ': ') if *indent* is ``None`` and
            (',', ': ') otherwise.  To get the most compact JSON representation,
            you should specify (',', ':') to eliminate whitespace.

            If specified, default is a function that gets called for objects
            that can't otherwise be serialized.  It should return a JSON encodable
            version of the object or raise a ``TypeError``.

            """

        super().__init__(skipkeys=skipkeys, ensure_ascii=ensure_ascii, check_circular=check_circular,
                         allow_nan=allow_nan, sort_keys=sort_keys, indent=indent, separators=separators,
                         default=default)

    def default(self, o):
        """Implement this method in a subclass such that it returns
            a serializable object for ``o``, or calls the base implementation
            (to raise a ``TypeError``).

            For example, to support arbitrary iterators, you could
            implement default like this::

                def default(self, o):
                    try:
                        iterable = iter(o)
                    except TypeError:
                        pass
                    else:
                        return list(iterable)
                    # Let the base class default method raise the TypeError
                    return JSONEncoder.default(self, o)

            """
        if hasattr(o, "dct"):
            return o.dct
        else:
            return JSONEncoder.default(self, o)


class BufferDescriptor(dict):
    exclude = ["default_fields", "encoder"]
    metadata = ["uid", "dtype"]

    def __get__(self, obj, objtype=None):
        dct = dict()

        dct |= obj.__class__.table[obj.uid]
        dct['metadata'] = dict()
        for k in self.metadata:
            dct['metadata'][k] = getattr(obj, k)

        return dct

    def __set__(self, instance, value):
        instance.__class__.table[instance.uid] = value


class RemoteType(type):

    @classmethod
    def __prepare__(mcs, name, bases, prefix=None, client=S3Client, default_descriptor=HookDescriptor, **kws):
        dct = super(RemoteType, mcs).__prepare__(name, bases)
        print(dct)

        _client = client(**kws)
        kws["prefix"] = prefix

        table = _client.table(Prefix=prefix)
        kws["__client__"] = _client
        kws.update(dct)
        print(table.Key)
        for k in table.Key:
            ky = k.replace(prefix, '')
            if not ky == '':
                kws[ky] = default_descriptor()
            else:
                pass
        return kws

    def __new__(mcs, classname, bases, dct, **kwds):

        print(classname, bases, dct)
        return type(classname, bases, dct)


from json import JSONEncoder
