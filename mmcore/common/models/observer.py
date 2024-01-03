import abc
import copy

import itertools


class Observable:
    def __init__(self, i=0):
        self._observers = []

        self.i = i

    def register_observer(self, observer: "Observer"):
        self.unsafe_register_observer(observer)
        observer.unsafe_add_observable(self)

    def unsafe_register_observer(self, observer):
        self._observers.append(observer)

    def notify_observers(self, **kwargs):
        for obs in self._observers:
            obs.notify(self, **kwargs)

    def notify_observers_backward(self, **kwargs):
        for obs in self._observers:
            obs.notify_backward(self, **kwargs)

    def notify_observers_forward(self, **kwargs):
        for obs in self._observers:
            obs.notify_forward(self, **kwargs)


class Observer:
    def __init__(self, i=0):
        self.observe = []

        self.i = i

    @abc.abstractmethod
    def notify(self, observable, **kwargs):
        ...

    def unsafe_add_observable(self, observable):
        insort(self.observe, observable.i)

    def add_observable(self, observable):
        self.unsafe_add_observable(observable)
        observable.unsafe_register_observer(self)

    def __repr__(self):
        return (f"{self.__class__.__name__}(id={self.i}, observe_count={len(self.observe)})")


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

    def init_observable_obj(self, *observers, obj):
        obj.i = next(self.observable_counter)
        self.observable.append(obj)
        for obs in observers:
            obj.register_observer(obs)

        return obj

    def _postinit_observable(self, observable, *observers):
        i = next(self.observable_counter)
        observable.i = i

        self.observable.append(observable)
        for obs in observers:
            observable.register_observer(obs)

    def init_observer(self, cls=Observer):
        i = next(self.observers_counter)
        self.observers.append(cls(i))
        return self.observers[i]

    def _postinit_observer(self, observer):
        i = next(self.observers_counter)
        observer.i = i
        self.observers.append(observer)

    def get_observables(self, observer: "Observer"):
        return (self.observable[i] for i in observer.observe)

    def get_observers(self, observer: "Observable"):
        return observer._observers


from bisect import bisect, insort

observation = Observation()

import functools


def observe(cls):
    @functools.wraps(cls)
    def wrapper(observer, *args, **kwargs):
        return observation.init_observable(observer, cls=lambda x: cls(x, *args, **kwargs)
                )

    return wrapper


class Listener(Observer):
    def __init__(self):
        super().__init__()
        observation._postinit_observer(self)

    @abc.abstractmethod
    def notify(self, observable, **kwargs):
        ...


class Notifier(Observable):
    def __init__(self, *observers):
        super().__init__()
        observation._postinit_observable(self, *observers)


class MeshObserver(Notifier):
    def __init__(self, *observers, mesh=None):
        super().__init__(*observers)
        self.mesh = mesh

    def notify_observers(self, props=None, control_points=None, controls=None, **kwargs):
        for obs in self._observers:
            obs.notify(self.mesh, props=props, controls_points=control_points, controls=controls, **kwargs, )
