import numpy as np
from functools import total_ordering
from operator import attrgetter

import itertools


@total_ordering
class FieldMap:
    """

    FieldMap
    ========

    Class representing a mapping between a source field and a target field.


    Subclassing
    -----------
    This class is decorated with the ``@total_ordering`` decorator, which provides the default comparison methods (
    __eq__ and __lt__) based on the backward_support attribute.

    Example:
    ----------------


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



    Methods
    -----------
        __init__(self, getter, target_field_name, backward_support=True, update_equal=False)
            Constructs a new FieldMap object.

            Parameters:
                getter (str): The source field in dot notation.
                target_field_name (str): The target field name.
                backward_support (bool): If True, the backward method will update the source field if the target
                field has changed. Defaults to True.
                update_equal (bool): If True, the backward method will update the source field even it is equal to
                the target field. Defaults to False.

        forward(self, source, target: dict)
            Maps the value from the source field to the target dictionary.

            Parameters:
                source (Any): The source object.
                target (dict): The target dictionary.

        backward(self, source, target: dict)
            Maps the value from the target field back to the source object if the target field has changed.

            Parameters:
                source (Any): The source object.
                target (dict): The target dictionary.

        __eq__(self, other)
            Compares two FieldMap objects based on the backward_support attribute.

            Parameters:
                other (FieldMap): The other FieldMap object.

            Returns:
                bool: True if the backward_support attributes are equal, False otherwise.

        __lt__(self, other)
            Compares two FieldMap objects based on the backward_support attribute.

            Parameters:
                other (FieldMap): The other FieldMap object.

            Returns:
                bool: True if the backward_support attribute is less than the other FieldMap's backward_support
                attribute, False otherwise.

        __hash__(self)
            Calculates the hash value of the FieldMap object.

            Returns:
                int: The hash value.

        __repr__(self)
            Returns a string representation of the FieldMap object.

            Returns:
                str: The string representation.


    """

    def __init__(self, source_field_name, target_field_name, backward_support=True, update_equal=False,
                 callbacks=(None, None)):
        self._source_field_name = source_field_name
        self.update_equal = update_equal
        self.backward_support = backward_support
        self._back_tuple = source_field_name.split('.')
        if len(self._back_tuple) > 1:
            self._back = attrgetter('.'.join(self._back_tuple[:-1]))
        else:
            self._back = lambda x: x
        self.callbacks = callbacks
        self._last_field = self._back_tuple[-1]
        self.getter = attrgetter(source_field_name)
        self.target_field_name = target_field_name

    @property
    def source_field_name(self):
        return self._source_field_name
    @property
    def forward_callback(self):
        return self.callbacks[0]

    @property
    def backward_callback(self):
        return self.callbacks[1]
    def forward(self, source, target: dict):
        val = self.getter(source)
        if self.forward_callback:

            val = self.forward_callback(val)

        target[self.target_field_name] = val

    def backward(self, source, target: dict):
        if self.target_field_name in target:

            if self.update_equal:
                self._backward_update(source, target)


            else:
                first = self.getter(source)
                second = target[self.target_field_name]
                if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
                    self._backward_update(source, target)
                else:
                    comparsion = lambda x, y: x == y
                    if not comparsion(first, second):

                        self._backward_update(source, target)


    def __eq__(self, other):
        return other.backward_support.__eq__(self.backward_support)

    def __lt__(self, other):
        return other.backward_support.__lt__(self.backward_support)

    def _backward_update(self, source, target):
        if self.backward_support:
            val = target[self.target_field_name]
            if self.backward_callback:
                val = self.backward_callback(val)

            setattr(self._back(source), self._last_field, val)
        else:
            self.forward(source, target)

    def __hash__(self):
        return hash((self._source_field_name, self.target_field_name, self.backward_support))

    def __repr__(self):
        return f'{self.__class__.__name__}({self._source_field_name} -> {self.target_field_name}, backward_support={self.backward_support}, update_equal={self.update_equal})'


class Observable:
    """
    Observable

    A class that represents an observable object. Observers can register themselves with
    the observable to receive notifications when the state of the observable object changes.

    Attributes:
        _observers (list): A list of registered observers
        i (int): An integer representing the state of the observable object

    Methods:
        __init__(self, i)
            Initializes a new instance of the Observable class.

        register_observer(self, observer: Observer)
            Registers an observer with the observable object and adds the observable
            object to the observer's list of observables.

        unsafe_register_observer(self, observer)
            Registers an observer with the observable object. The observer's list
            of observables is not updated.

        notify_observers(self, **kwargs)
            Notifies all registered observers about the change in the observable object's state.
            Optional keyword arguments can be provided to be passed to the observers.

    Example:
        observable = Observable(10)
        observer = Observer()

        # Register the observer with the observable object
        observable.register_observer(observer)

        # Notify the observers about the change in the observable object's state
        observable.notify_observers()
    """
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
    """
    Initializes an Observer object.

    :param i: The unique identifier for the Observer.
    :type i: int
    """
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
    """Class to manage observations and observers.

    This class provides methods for registering and unregistering observers, managing observables, and retrieving
    observables and observers.

    Attributes:
        observers (list): List of registered observers.
        observers_counter (itertools.count): Counter to assign unique IDs to observers.
        observable_counter (itertools.count): Counter to assign unique IDs to observables.
        observable (list): List of available observables.

    """
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
