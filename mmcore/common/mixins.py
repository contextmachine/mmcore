import inspect
from textwrap import dedent
from types import new_class
from uuid import uuid4


def _generic_mixin_from_body(mxn, parent_cls):
    _n = parent_cls.__name__.replace('Mixin', '')
    nm = f'{_n}{mxn.__name__}'
    body = dedent(inspect.getsource(mxn.MixinPrototype)).replace('MixinPrototype', f'{nm}({parent_cls.__name__})')
    exec(body)
    return eval(nm)


class GenericMixin:
    """


    :param minimum_base : Minimum required parent mixin type.
    :type minimum_base : type

    A generic mixin. Can be mixed with any Mixin originating from its own minimum_base

    Example
    -------
    .. code-block:: python


    >>> class UUIDMixin:
    ...     uuid:str
    ...     ...


    >>> class InstancesMixin:
    ...     _instances = {}
    ...     key_func=lambda obj: id(obj) # Because we don't assume that user classes will be orderable.
    ...     def __new__(cls, *args, **kwargs):
    ...         obj=super().__new__(cls, *args, **kwargs)
    ...
    ...         cls._instances[cls.key_func(obj)]=obj
    ...         ...

    FlyWeight pattern required access to class instances and the ability to identify if a suitable instance exists.
    There are many ways to do this. For this example, we will focus on a simple uuid search .
    >>> class FlyWeightMixin(InstancesMixin, UUIDMixin):
    ...     key_func=lambda obj: obj.uuid
    ...     def __new__(cls, *args, uuid=None, **kwargs):
    ...         if uuid in cls._instances:
    ...             return cls._instances[uuid]
    ...



    >>> class ChildrenMixin(GenericMixin):
    ...     minimum_base = InstancesMixin
    ...
    ...     class MixinPrototype:
    ...         children:list
    ...
    ...         def parents(self):
    ...
    ...             for item in self.__class__._instances.values():
    ...                 if self in item.children:
    ...                     yield item


    ChildrenMixin.MixinPrototype required _instances attribute (like InstancesMixin or FlyWeightMixin) for access to parents.
    However, there may be reasons not to use the FlyWeight pattern for this implementation.
    Now imagine that your Mixin tree is large enough.

    Result of derived mixin generation

    >>> ChildrenMixin[InstancesMixin] # Ok
    >>> ChildrenMixin[FlyWeightMixin] # Ok
    >>> ChildrenMixin[UUIDMixin] # Raise

    >>> class HashFlyWeightMixin(InstancesMixin):
    ...     key_func=lambda attrs:hash(attrs)
    ...     def __new__(cls, *args, hashable_attrs=(),**kwargs):
    ...         if hashable_attrs in cls._instances:
    ...             return cls._instances[hashable_attrs]
    ...
    ...         ...



    """
    minimum_base: type

    class MixinPrototype:
        ...

    def __class_getitem__(cls, item):
        if not issubclass(item, cls.minimum_base):
            raise TypeError(f"{item} is not subclass of {cls.minimum_base}")

        return _generic_mixin_from_body(cls, item)


class ParamsMixin:
    """
    Description
    -----------
    :en
        Mixin allows the attribute params, which is a dictionary.
        Calls to __getattr__ and __setattr__ will be redirected to params, bypassing __dict__ if it is present in the key.
        Editing params will directly affect the values of the object's attributes and vice versa.

    :ru
        Миксин позволяет аттрибут params, который представляет собой словарь.
        Вызовы __getattr__ и __setattr__ будут перенаправлены на params  в обход __dict__, если в ключ нем присутствует.
        Редактирование params будет прямо влиять на значения аттрибутов объекта и наоборот.

    Basic Implementation Example
    ----------------------------
    :en
        In this example, we only write to params those names and values,
        that were passed as keywords to the constructor when the object was created.
    :ru
        В этом примере мы записываем в params только те имена и значения,
        которые были переданы как ключивые слова в конструктор при создании объекта .

    >>> class Object(ParamsMixin):
    ...    def __init__(self, **kwargs):
    ...        self._params = kwargs
    >>> o = Object(a=3, v=10, b=11)
    >>> o.params
    {'a': 3, 'v': 10, 'b': 11}
    >>> o.a
    3
    >>> o.a = 5
    >>> o.params
    {'a': 5, 'v': 10, 'b': 11}
    >>> o.params['a'] = 40
    >>> o.a
    40
    >>> o.new_attr = 'no param'
    >>> o.params
    {'a': 40, 'v': 10, 'b': 11}
    >>> o.new_attr
    'no param'


    """
    _params: dict

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, v):
        self._params = v

    def __getattr__(self, k):
        if not k.startswith('_') and hasattr(self, '_params'):
            params = object.__getattribute__(self, '_params')
            if k in params:
                return params[k]
        return super().__getattribute__(k)

    def __setattr__(self, k, v):
        if not k.startswith('_') and hasattr(self, '_params'):
            params = object.__getattribute__(self, '_params')
            params.__setitem__(k, v) if k in params else super().__setattr__(k, v)
        else:
            super().__setattr__(k, v)


class UUIDMixin:
    def __init__(self, *args, uuid=None, **kwargs):
        if uuid is None:
            uuid = uuid4().hex

        super(UUIDMixin, self).__init__()
        self._uuid = uuid

    @property
    def uuid(self):
        return self._uuid


class InstancesMixin:
    _instances = []

    key_func = lambda obj: id(obj)

    def __new__(cls, *args, **kwargs):
        obj = super(InstancesMixin, cls).__new__(cls)
        cls._instances.append(obj)
        return obj


class FlyWeightMixin(UUIDMixin, InstancesMixin):
    _instances = {}

    key_func = lambda obj: obj.uuid

    def __new__(cls, *args, uuid=None, **kwargs):
        if uuid in cls._instances:
            return cls._instances[uuid]

        obj = super().__new__(cls, uuid=uuid)

        return obj

    @property
    def uuid(self):
        return self._uuid

    @uuid.setter
    def uuid(self, v):
        if not isinstance(v, str):
            raise TypeError('uuid must be a string')
        del self.__class__._instances[self._uuid]
        self.__class__._instances[v] = self
        self._uuid = v


class ChildrenMixin(GenericMixin):
    """
    >>> from mmcore.common.mixins import InstancesMixin, ChildrenMixin, ParamsMixin
    >>> class MyObj2(ChildrenMixin[InstancesMixin], ParamsMixin):
    ...     def __init__(self, *children, **kwargs):
    ...         self._params = kwargs
    ...         super().__init__()
    ...         self.children=list(children)
    >>> o = MyObj2(MyObj2(a=4,b=7,d='D'),
    ...             MyObj2(a=5,b=6,c=-1),
    ...             MyObj2(),
    ...             a=1,
    ...             b=-1
    ...         )
    >>> o.traverse_children(lambda ob:print(ob.params))
    {'a': 1, 'f': -1}
    {'a': 4, 'b': 7, 'd': 'D'}
    {'a': 5, 'b': 6, 'c': -1}
    {}
    >>> o.traverse_children(lambda ob:print(ob.params),forward=False)
    """
    minimum_base = InstancesMixin

    class MixinPrototype:
        children: list

        def parents(self):
            for item in self.__class__._instances.values():
                if self in item.children:
                    yield item

        def traverse_children(self, cb, forward=True):
            if forward:
                cb(self)
            for child in self.children:
                child.traverse_children(cb)
            if not forward:
                cb(self)

        def traverse_parents(self, cb, forward=True):
            if forward:
                cb(self)
            for parent in self.parents():
                parent.traverse_parents(cb)
            if not forward:
                cb(self)

        def roots(self):
            roots = []

            def callback(obj):
                nonlocal roots
                if len(obj.children) == 0:
                    roots.append(obj)

            self.traverse_parents(callback)
            return roots

        def leafs(self):
            leafs = []

            def callback(obj):
                nonlocal leafs
                if len(obj.children) == 0:
                    leafs.append(obj)

            self.traverse_children(callback)
            return leafs
