from functools import total_ordering
from operator import attrgetter

import itertools


@total_ordering
class FieldMap:
    """
>>> from mmcore.geom.rectangle import Rectangle
>>> r=Rectangle(10,20)
>>> target=dict()
>>> target
Out: {}
>>> mappings=(FieldMap('area', 'area', backward_support=False),
...                  FieldMap('u', 'width'),
...                  FieldMap('v', 'height'))
>>> for mapping in mappings:
...     mapping.forward(r, target)
>>> target
Out: {'width': 10, 'height': 20, 'area': 200.0}
>>> target['width']=30
>>> for mapping in sorted(mappings):
...     mapping.backward(r, target)
>>> target
Out: {'width': 30, 'height': 20, 'area': 600.0}
>>> r.u
Out: 30
    """

    def __init__(self, getter, target_field_name, backward_support=True, update_equal=False):
        self._source_field_name = getter
        self.update_equal = update_equal
        self.backward_support = backward_support
        self._back_tuple = getter.split('.')
        if len(self._back_tuple) > 1:
            self._back = attrgetter('.'.join(self._back_tuple[:-1]))
        else:
            self._back = lambda x: x

        self._last_field = self._back_tuple[-1]
        self.getter = attrgetter(getter)
        self.target_field_name = target_field_name

    def forward(self, source, target: dict):
        target[self.target_field_name] = self.getter(source)

    def backward(self, source, target: dict):

        if self.update_equal:
            self._backward_update(source, target)
        elif self.getter(source) != target[self.target_field_name]:
            self._backward_update(source, target)

    def __eq__(self, other):
        return other.backward_support.__eq__(self.backward_support)

    def __lt__(self, other):
        return other.backward_support.__lt__(self.backward_support)

    def _backward_update(self, source, target):
        if self.backward_support:

            setattr(self._back(source), self._last_field, target[self.target_field_name])
        else:
            self.forward(source, target)

    def __hash__(self):
        return hash((self._source_field_name, self.target_field_name, self.backward_support))

    def __repr__(self):
        return f'{self.__class__.__name__}({self._source_field_name} -> {self.target_field_name}, backward_support={self.backward_support}, update_equal={self.update_equal})'


class Observable:
    def __init__(self, i):
        self._observers = []
        self.i = i

    def register_observer(self, observer: 'Observer'):
        self.unsafe_register_observer(observer)
        observer.unsafe_add_observable(self)

    def unsafe_register_observer(self, observer):
        self._observers.append(observer)

    def notify_observers(self, **kwargs):
        for obs in self._observers:
            obs.notify(self, **kwargs)


class Observer:
    def __init__(self, i):
        self.observe = []

        self.i = i

    def notify(self, observable, **kwargs):
        print(observable, kwargs)

    def unsafe_add_observable(self, observable):
        insort(self.observe, observable.i)

    def add_observable(self, observable):
        self.unsafe_add_observable(observable)
        observable.unsafe_register_observer(self)

    def __repr__(self):
        return f'{self.__class__.__name__}(id={self.i}, observe_count={len(self.observe)})'


class Observation:
    def __init__(self):
        self.observers = []
        self.observers_counter = itertools.count()
        self.observable_counter = itertools.count()
        self.observable = []

    def unregister_observer(self, observer):
        for j in observer.observe:
            self.observable[j].observers.remove(observer)

    def dispose_observable(self, observable):

        for observer in observable._observers:
            i = bisect(observer.observe, observable.i)
            for j in list(range(len(observer.observe)))[i:]:
                observer[j] -= 1
            del observer[i]

        self.observable.remove(observable)

    def init_observable(self, *observers, cls=Observable):

        i = next(self.observable_counter)
        obj = cls(i)
        self.observable.append(obj)
        for obs in observers:
            obj.register_observer(obs)

        return obj

    def init_observer(self, cls=Observer):
        i = next(self.observers_counter)
        self.observers.append(cls(i))
        return self.observers[i]

    def get_observables(self, observer: 'Observer'):

        return (self.observable[i] for i in observer.observe)

    def get_observers(self, observer: 'Observable'):
        return observer._observers


from bisect import bisect, insort

observation = Observation()
