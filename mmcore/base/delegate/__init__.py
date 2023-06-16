class class_bind_delegate_method:
    """
    Оборачивает в класс результат функции возвращяющей делегат этого класса,
    Принимает 2 функции в качестве аргументов:
    1. функция преобразования в которую передаются все входящие аргументы + delegate (резултат следующий функции),
        должна вернуть объект целевого класса
    2. функция принимающая аргументы и возвращающая делегата
    Оборачивает в класс результат функции возвращяющей делегат этого класса
    Example:
    >>> @class_bind_delegate_method
def bind_poly_to_shape(self,  other, delegate=None ):
    return self.__class__(boundary=list(delegate.boundary.coords), holes=list(delegate.interiors), color=self.color, h=self.h)

    """

    def __init__(self, f):
        self.f = f

    def __call__(self, m):
        def wrap(*args, **kwargs):
            return self.f(*args, **kwargs, delegate=m(*args, **kwargs))

        return wrap


class delegate_method:
    def __init__(self, check):
        self._check = check
        self._m = lambda slf, dlg, oth: ...

    def __call__(self, slf, item):
        return self._m(slf, slf._ref, self._check(slf, item, self._m))

    def bind(self, m):
        self._m = m
        return self


class delegate_method_full:
    def __init__(self, check):
        self._check = check
        self._m = lambda slf, *args, **kwargs: ...

    def __call__(self, slf, *args, **kwargs):
        return self._m(slf, slf._ref, *args, **kwargs)

    def bind(self, m):
        self._m = m
        return self
