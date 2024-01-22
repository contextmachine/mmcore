import functools
from typing import Any, Generic, Type, Union

import numpy as np

from typing import TypeVar

T = TypeVar('T')


class Parameter(Generic[T]):
    value: T
    cast_type = Union[Type[T], np.dtype, str, Any]

    def __init__(self, value):
        super().__init__()

        self.value = self.cast(value)

    def cast(self, val, typ=None):

        if np.isscalar(val):
            if isinstance(val, Parameter):
                return self.cast(val.value, typ=typ)
            else:
                if typ is not None:
                    return self.cast(val, typ=typ)
                else:
                    return self.cast_type(val)

        else:
            if typ is None:
                typ = self.cast_type

            if isinstance(val, (list, tuple)):

                return np.array(val.__class__((self.cast(i, typ=typ) for i in val)), dtype=typ)
            else:
                return np.array(val, dtype=typ)

    def __repr__(self):
        return f'{self.__class__}(value={self.value})'

    def __float__(self):
        return self.cast(self.value, float)

    def __int__(self):
        return self.cast(self.value, int)

    def __bool__(self):
        return self.cast(self.value, bool)

    def __str__(self):
        return str(self.value)

    @property
    def is_scalar(self):
        return np.isscalar(self.value)

    def __array__(self, dtype=None):
        return np.array(self.value, dtype=dtype if dtype else self.cast_type)

    def __iter__(self):
        if self.is_scalar:
            raise TypeError('Scalar Parameter object is not iterable')
        else:
            return iter(self.value)

    def set(self, val: 'P|T'):
        if isinstance(val, Parameter):
            self.value = self.cast(val.value)
        else:
            self.value = self.cast(val)

    def get(self):
        return self.value


P = TypeVar('P', bound=Parameter)


@functools.total_ordering
class NumericParameter(Parameter):

    value: float
    cast_type = float

    def __init__(self, value):
        super().__init__(value)

    def __add__(self, other):
        return self.__class__(self.value + self.cast(other))

    def __iadd__(self, other):
        self.value += self.cast(other)
        return self

    def __radd__(self, other):
        return other + self.value

    def __sub__(self, other):
        return self.__class__(self.value - self.cast(other))

    def __isub__(self, other):
        self.value -= self.cast(other)
        return self

    def __rsub__(self, other):
        return other.__sub__(self.value)

    def __mul__(self, other):
        return self.__class__(self.value * self.cast(other))

    def __imul__(self, other):
        self.value *= self.cast(other)
        return self

    def __round__(self, n=None):
        self.__class__(self.value.__round__(n))

    def __divmod__(self, other):
        return self.value, self.cast(other)

    def __truediv__(self, other):
        return self.__class__(self.value / self.cast(other))

    def __pow__(self, other):
        return self.value ** self.cast(other)

    def __mod__(self, other):
        return self.value.__mod__(other)

    def __eq__(self, other):
        return self.value == getattr(other, 'value', other)

    def __le__(self, other):
        return self.value.__le__(getattr(other, 'value', other))

    def __lt__(self, other):
        return self.value.__lt__(getattr(other, 'value', other))

    def __ge__(self, other):
        return self.value.__ge__(getattr(other, 'value', other))

    def __gt__(self, other):
        return self.value.__gt__(getattr(other, 'value', other))

    def scalar_iterator(self):
        return ScalarParameterIterator(self)


class ScalarParameterIterator:
    def __init__(self, val: Parameter):
        self._cls = val.__class__
        self._val = iter((val,)) if val.is_scalar else np.nditer(val)

    def __iter__(self):
        return self

    def __next__(self):
        val = self._val.__next__()
        if isinstance(val, Parameter):
            return val
        else:
            return self._cls(val)


class Length(NumericParameter):
    value: float
    cast_type = float


def cast_to_parameter(val: 'Any | P', parameter_type: Type[P]) -> P:
    if isinstance(val, Parameter):
        return val
    else:
        return parameter_type(val)
