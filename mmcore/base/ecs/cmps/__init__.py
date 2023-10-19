labels = dict()
components = []


def selector(*names):
    s = set()
    [s.add(components[i]) for i in labels.get(names[0], ())]
    for name in names[1:]:
        s.intersection_update(set(components[i] for i in labels.get(name, [])))

    return s


def label(*names):
    def wrap(cls):
        if not hasattr(cls, '__labels__'):
            cls.__labels__ = (cls.__qualname__,)
        cls.__labels__ = tuple(set(cls.__labels__ + (cls.__qualname__,) + names))
        return cls

    return wrap


@label('base', )
class EntityComponent:
    __labels__ = ()

    def __hash__(self):
        return hash(repr(self))

    def __new__(cls, *args, **kwargs):

        obj = super().__new__(cls)
        components.append(obj)
        ix = len(components)
        for lbl in cls.__labels__:

            if lbl not in labels:
                labels[lbl] = []
            labels[lbl].append(ix)
        return obj

    @property
    def component_index(self):
        return components.index(self)

    def __del__(self):
        ix = self.component_index

        for lbl in self.__labels__:
            labels[lbl].remove(ix)
        components[ix] = None
        del self
