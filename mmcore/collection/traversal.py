import collections
from typing import Any, Callable, Sequence

import numpy
import numpy as np
import typing
from typing import TypeVar


"""
    class traverse(Callable):
        def __call__(self, seq: Sequence | Any) -> Any:
            ...
            
            for typ in self.type_extras
                if isinstance(seq, typ.cls):
                    return typ.resolver(seq)
                else:
                    continue
                
            
"""
TraverseTypeExtra = collections.namedtuple("TraverseTypeExtra", ["cls", "resolver"])

class traverse(Callable):
    """
    Полностью проходит секвенцию любой степени вложенности.
    В момент обладания каждым из объектов может применить `placeholder`.
    По умолчанию не делает ничего (вернет генератор той-же самой секвенции, что и получил).
    Генератор всегда предсказуем. самый простой способ распаковки -- один раз передать его в `next`.
    -----------------------------------------------------------------------------------------------------
    Example:
    >>> import numpy as np
    >>> a = np.random.random((4,2,3,7))
    >>> b = next(traverse()(a))
    >>> b.shape, a.shape
    ((4, 2, 3, 7), (4, 2, 3, 7))
    >>> np.allclose(a,b)
    True
    """
    __slots__ = ("callback", "traverse_dict", "type_extras")

    def __init__(self, callback: Callable | None = None, traverse_dict = True, type_extras=()):
        super().__init__()
        self.traverse_dict = traverse_dict

        self.type_extras=type_extras
        if callback is None:
            self.callback = lambda x: x
        else:
            self.callback = callback
    @property
    def extra_classes(self):
        return [typ.cls for typ in self.type_extras]
    def __call__(self, seq: Sequence | typing.Mapping | Any) -> Sequence | typing.Mapping | Any:
        if seq.__class__ in self.extra_classes:
            for typ in self.type_extras:
                if isinstance(seq, typ.cls):
                    return typ.resolver(seq)
                else:
                    continue

        elif isinstance(seq, str):
            # Это надо обработать в самом начале
            return self.callback(seq)
        elif isinstance(seq, Sequence | np.ndarray):

            return seq.__class__([self(l) for l in seq])

        elif isinstance(seq, dict):
            if self.traverse_dict:
                dt={}
                for k,v in seq.items():
                    dt[k]=self(v)
                return dt
            else:
                return seq
        else:
            return self.callback(seq)


type_extractor = traverse(lambda x: x.__class__, traverse_dict=False)
