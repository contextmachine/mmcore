import typing
from abc import abstractmethod
from typing import Callable

class HashedSeq(list):
    """ This class guarantees that hash() will be called no more than once
        per element.  This is important because the lru_cache() will hash
        the key multiple times on a cache miss.

    """

    __slots__ = 'hashvalue'

    def __init__(self, tup, hash=hash):
        self[:] = tup
        self.hashvalue = hash(tup)

    def __hash__(self):
        return self.hashvalue

class _wrapper:
    def __init__(self, mask, instance, owner, **kwargs):
        # print(mask, instance, owner)

        self.mask, self.instance, self.owner = mask, instance, owner
        self.__dict__ |= kwargs

    def __call__(self, constrains, *args, **kwargs):
        return self.mask.wrapper(self.mask, self.instance, self.owner, tuple(constrains), **kwargs)


class MaskType(type):
    # noinspection PyCallingNonCallable
    """
        >>> class FooMask(metaclass=MaskType):
        ...     @staticmethod
        ...     def wrapper(mask, instance, owner, constrains, *args, **kwargs):
        ...         return list(filter(constrains, instance[mask.name]))
        >>> class A(dict):
        ...     foo=FooMask()
        >>> bar=A({"foo":[2,1,5,6,1,2,3]})
        >>> bar.foo(lambda x: x<3)
        [2, 1, 1, 2]
        """


    class setter_dict(dict):
        def setter(self, key):
            return lambda value: dict.__setitem__(self, key, value)


    @classmethod
    def __prepare__(mcs, name, bases, **kws):
        namespace = mcs.setter_dict(super().__prepare__(name, bases, **kws))


        # template class
        class wrap(_wrapper):
            __name__ = name + "Wrapper"

            def __repr__(self): return f"<{name}Wrapper object at {id(self)}>"

            def __str__(self): return f"{name}Wrapper object at {id(self)}"


        @namespace.setter("__get__")
        def _(self, instance, owner) -> wrap | typing.Any:
            wrp = wrap(self, instance, owner)
            if self._constrains is not None:
                return wrp(self._constrains, **self._kwargs)
            else:
                return wrp

        @namespace.setter("__set__")
        def _(self, instance, value) -> None:
            if isinstance(value, tuple):
                self._constrains, kw = value
                self._kwargs |= kw
            elif isinstance(value, dict):
                try:
                    self._constrains = value.pop("constrains")
                except KeyError as err:
                    pass
                except Exception as err:
                    raise err
                self._kwargs |= value
            else:
                self._constrains = value

        # template __set_name__
        @namespace.setter("__set_name__")
        def _(self, owner, nm):
            self.name = nm

        namespace |= kws
        return namespace


class PartialMask(metaclass=MaskType):
    @staticmethod
    @abstractmethod
    def wrapper(mask, instance, constrains, *args, **kwargs): ...

    # stub method
    def __get__(self, instance, owner) -> Callable: ...


class _Mask(metaclass=MaskType):

    def __init__(self, constrains=None, **kwargs):
        self._constrains = constrains
        self._kwargs = kwargs

    def __set_name__(self, self1, name): ...

    def __get__(self, instance, owner): ...


class Mask(_Mask, metaclass=MaskType):
    """
    Common Mask class
    """

    def __init__(self, constrains=None, **kwargs):
        super().__init__(constrains=constrains, **kwargs)
