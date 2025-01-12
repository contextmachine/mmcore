from __future__ import annotations

import inspect
from abc import ABCMeta
from collections import namedtuple

from ._typing import TypeVar, Type
from types import FunctionType, MethodType

CreateAcceptArgs = namedtuple("CreateAcceptArgs", ["names", "varkw", "spec"])


class BaseType(ABCMeta):
    def __new__(mcs, name, bases, attrs, **kws):
        res: MethodType = attrs.get("create")
        res = res if res else classmethod(lambda cls, **kwargs: cls())

        spec = inspect.getfullargspec(res.__func__)

        attrs["__create_accept_args__"] = CreateAcceptArgs(
            spec.args[1:], bool(spec.varkw), spec
        )

        attrs["create_accept_args"] = property(
            fget=lambda self: self.__class__.__create_accept_args__
        )
        return super().__new__(mcs, name, bases, attrs, **kws)


class Base(metaclass=BaseType):
    """
    The base class that all other classes are derived from.
    """

    __create_accept_args__: CreateAcceptArgs

    @classmethod
    def create(cls, *args, **kwargs):
        ...

    @classmethod
    def from_data(cls, data):
        spec = cls.__create_accept_args__
        if spec.varkw:
            return cls.create(**data)
        else:
            return cls.create(**{k: data.get(k, None) for k in spec.names})

    def __init__(self) -> None:
        pass

    @classmethod
    def cast(cls, arg) -> Base:
        return cls(arg)

    @classmethod
    def class_type(cls) -> str:
        """
        Static function that all classes support that returns the type of the class as a string.
        The returned string matches the string returned by the objectType property. For example if you
        have a reference to an object and you want to check if it's a SketchLine you can use
        myObject.objectType == fusion.SketchLine.classType().
        Returns a string indicating the type of the object.
        """
        return cls.__name__

    @property
    def object_type(self) -> str:
        """
        Returns a string indicating the type of the object.
        """
        return self.class_type()

    @property
    def is_valid(self) -> bool:
        """
        Indicates if this object is still valid, i.e. hasn't been deleted
        or some other action done to invalidate the reference.
        """
        return bool()


TB = TypeVar("TB", bound=Base)

__collection_classes__ = dict()


class ObjectCollection(list, Base):
    """
    Generic collection used to handle lists of any object type.

    >>> from mmcore.api import ObjectCollection,Point3D
    >>> import numpy as np
    >>> ObjectCollection[Point3D].cast(np.array([[8., 9., 6.], [7., 4., 9.], [7., 1., 3.], [0., 6., 8.], [1., 3., 3.]]))
    ObjectCollection[Point3D]([Point3D(x=8.0, y=9.0, z=6.0), Point3D(x=7.0, y=4.0, z=9.0), Point3D(x=7.0, y=1.0, z=3.0), Point3D(x=0.0, y=6.0, z=8.0), Point3D(x=1.0, y=3.0, z=3.0)])
    """

    def __repr__(self):
        return f"{type(self).__name__}({super().__repr__()})"

    def __class_getitem__(cls, item: Type[TB]):
        name = f"{cls.__name__}[{item.__name__}]"
        if name not in __collection_classes__:
            new_cls = type(
                f"{cls.__name__}[{item.__name__}]", (cls,), {"item_type": item}
            )
            new_cls.item_type = item
            __collection_classes__[name] = new_cls
        return __collection_classes__[name]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def cast(cls, arg) -> ObjectCollection:
        return cls(cls.item_type.cast(item) for item in arg)

    @classmethod
    def create(
        cls,
    ) -> ObjectCollection:
        """
        Creates a new ObjectCollection object.
        Returns the newly created ObjectCollection.
        """
        return ObjectCollection()

    def __getitem__(self, index: int) -> Base:
        """
        Function that returns the specified object using an index into the collection.
        index : The index of the item within the collection to return. The first item in the collection has an index of 0.
        Returns the specified item or null if an invalid index was specified.
        """
        return Base()

    def add(self, item: Base) -> bool:
        """
        Adds an object to the end of the collection.
        Duplicates can be added to the collection.
        item : The item to add to the list.
        Returns false if the item was not added.
        """
        return bool()

    def remove(self, item: Base) -> bool:
        """
        Function that removes an item from the collection.
        item : The object to remove from the collection.
        Returns true if the removal was successful.
        """
        return bool()

    def __delitem__(self, index: int) -> bool:
        """
        Function that removes an item from the list.
        Will fail if the list is read only.
        index : The index of the item to remove from the collection. The first item has an index of 0.
        Returns true if the removal was successful.
        """
        return bool()

    def find(self, item: Base, startIndex: int) -> int:
        """
        Finds the specified component in the collection.
        item : The item to search for within the collection.
        startIndex : The index to begin the search.
        Returns the index of the found item. If not found, -1 is returned.
        """
        return int()

    def __contains__(self, item: Base) -> bool:
        """
        Returns whether the specified object exists within the collection.
        item : The item to look for in the collection.
        Returns true if the specified item is found in the collection.
        """
        return bool()

    def clear(self) -> bool:
        """
        Clears the entire contents of the collection.
        Returns true if successful.
        """
        return bool()

    @property
    def count(self) -> int:
        """
        Returns the number of occurrences in the collection.
        """
        return int()
