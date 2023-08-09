__derived_attributes_table__ = dict()
__derived_dispatch_table__ = dict()


class DerivedMethod:
    def __init__(self, name, ref):
        super().__init__()

        self.__mmcore_ref__ = ref
        self._name = name

    def __call__(self, own):
        if own.__qualname__ not in __derived_attributes_table__:
            __derived_attributes_table__[own.__qualname__] = dict()
        __derived_attributes_table__[own.__qualname__][self._name] = self.__mmcore_ref__

        setattr(own, self._name, property(
            lambda slf: __derived_dispatch_table__[self.__mmcore_ref__](slf)))

        return own


class Derive:
    """
    >>> pts = [[0, 0, 0], [1, 0, 0], [1, 2, 0]]

    >>> @Derive
    ... def prev_attr_resolver(self):
    ...     return pts[self._pt - 1]

    >>> @Derive
    ... def next_attr_resolver(self):
    ...     try:
    ...         return pts[self._index + 1]
    ...     except IndexError:
    ...         return pts[(self._index  + 1) % len(pts)]

    >>> @Derive
    ... def prev_attr_resolver(self):
    ...    return pts[self._pt - 1]

    >>> @prev_attr_resolver.derived("prev_pt")
    ... @next_attr_resolver.derived("next_pt")
    ... class ExperimentalLinkedNode:
    ...       def __init__(self, index=1):
    ...           self._index = index







pts = [[0, 0, 0], [1, 0, 0], [1, 2, 0]]



from scipy.spatial import distance


@Derive
def all_tests_test(self):
    return str(self._test) * 2





@Derive
def dist_resolver(self):
    return distance.euclidean(prev_attr_resolver(self), next_attr_resolver(self))


@all_tests_test.derived("test")
@prev_attr_resolver.derived("prev_pt")
@next_attr_resolver.derived("next_pt")
class B:
    _test = 1
    dist = DerivedAttribute("dist_resolver")

    def __init__(self, pt=1):
        self._pt = 1


stop = False


def iii():
    global stop
    for i in np.linspace(0.1, 2 * np.pi, 200):
        time.sleep(0.01)
        pts[0] = [np.cosh(i), np.sinh(i), 1 / i]
    stop = True


th = threading.Thread(target=iii)
th.start()
j = -1
while True:
    if stop:
        break
    j += 1
    print(f'[loop: {j}] {o.dist}', flush=True, end="\r")

    """

    def __init__(self, fun):
        super().__init__()
        self.__mmcore_ref__ = fun.__name__
        __derived_dispatch_table__[self.__mmcore_ref__] = fun

    def __call__(self, *arga, **kwargs):
        return __derived_dispatch_table__[self.__mmcore_ref__](*arga, **kwargs)

    def derived(self, name):
        return DerivedMethod(name, self.__mmcore_ref__)


class DerivedAttribute:

    def __init__(self, ptr):
        self.__mmcore_ref__ = ptr

    def __set_name__(self, own, name):
        self._name = name
        if own.__qualname__ not in __derived_attributes_table__:
            __derived_attributes_table__[own.__qualname__] = dict()

        __derived_attributes_table__[own.__qualname__][self._name] = self.__mmcore_ref__

    def __get__(self, inst, own=None):

        if inst is not None:

            return __derived_dispatch_table__[__derived_attributes_table__[own.__qualname__][self._name]](inst)


        else:
            return __derived_dispatch_table__[__derived_attributes_table__[own.__qualname__][self._name]]


class DerivedProperty(DerivedAttribute):
    def __init__(self, fget):
        self.fget = fget
        self._name = fget.__name__
        __derived_dispatch_table__[fget.__qualname__] = fget

        super().__init__(fget.__qualname__)

    def __get__(self, inst, own=None):
        return super().__get__(inst, own=None)
