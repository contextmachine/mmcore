import copy
import inspect


class TraitType(type):

    def __new__(metacls, name, bases, attrs, **kws):
        trait = type.__new__(metacls, name, bases, attrs, **kws)

        def get_cls(self, cls):



            for k in dir(trait):
                v=getattr(trait,k)
                print(k,v)
                if not k.startswith("_") and( inspect.isfunction(v) or inspect.ismethod(v)):
                    print(k, v)
                    class TraitMethodDescriptor:

                        def __init__(self, cls):
                            super().__init__()
                            self._cls = cls
                            setattr(cls, k, self)
                            self.name = k
                            self._trait_name = name

                        def __get__(self, inst, owner):
                            return lambda *args, **kwargs: getattr(self._cls.__traits__[self._trait_name],self.name)( inst, *args ,**kwargs)

                    TraitMethodDescriptor(cls)
                    if not hasattr(cls, '__traits__'):
                        cls.__traits__ = {}
                    cls.__traits__[name]=self
            self._cls=cls

        trait.__init__ = get_cls
        trait.__call__=lambda self, *args,**kwargs:self._cls(*args,**kwargs)

        return trait


class Trait(metaclass=TraitType):
    def trait_add(self, obj, obj2):
        return self._cls(arr=[a + b for a, b in zip(obj.arr, obj2.arr)])

    def trait_sub(self, obj, obj2):
        return self._cls(arr=[a - b for a, b in zip(obj.arr, obj2.arr)])
