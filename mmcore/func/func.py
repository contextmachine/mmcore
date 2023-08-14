from operator import methodcaller


class MapMethodDescriptor:
    def __init__(self, fun):
        super().__init__()
        self._fun = fun

    def __set_name__(self, owner, name):
        self._name = "_" + name
        self._caller = self._fun(name)

    def __get__(self, instance, owner):
        if instance:
            return self._caller(instance)
        elif owner:
            return self


@MapMethodDescriptor
def mapper(name):
    def wrap(seq):
        def inner(*args, **kwargs):
            if args == () and len(kwargs) == 0:
                return map(methodcaller(name), seq)
            else:
                return map(methodcaller(name, *args, **kwargs), seq)

        return inner

    return wrap


def even_filter(iterable, reverse=False):
    def even_filter_num(item):
        return reverse != ((iterable.index(item) % 2) == 0)

    return filter(even_filter_num, iterable)
