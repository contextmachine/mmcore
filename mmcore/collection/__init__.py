#  Copyright (c) 2022. Computational Geometry, Digital Engineering and Optimizing your construction processe"
from __future__ import absolute_import, annotations

import inspect
from collections.abc import Iterator
from functools import wraps

from ..baseitems import Item


def t(glb):
    for i, kv in enumerate(glb.items()):
        k, v = kv

        yield i, k


def clsmap(seq, item):
    cls = seq[0].__class__
    cls_attr = getattr(cls, item)
    # print(f"Target <{item}> is <{cls.__name__}>'s method or base attribute ...")
    if inspect.ismethod(cls_attr):
        # print("It is method!")

        @wraps(cls_attr)
        def wrp(args, **kwargs):
            for i, slf in enumerate(cls.seq):
                arg = args[i]
                f = getattr(slf, item)
                # print(f"Start yielding ...\nwith seq[{i}] yield {slf.__repr__()}.{item}(args={arg}, kwargs={kwargs})")

                try:
                    kw = dict(arg)
                    kw |= kwargs
                    yield f(**kw)
                finally:
                    yield f(*arg, **kwargs)
            return wrp
    else:
        raise "It is not method"


# To accomplish this, subclass list in such a manner that each list element is
# really a two tuple, where the first tuple value is the actual value and the
# second tuple value is a reference to the itemlist node for that value. Users
# only interact with the first tuple values, the actual values, but behind the
# scenes when an element is modified, deleted, inserted, etc, the according
# itemlist nodes are modified, deleted, inserted, etc accordingly. In this
# manner, users can manipulate omdict objects directly through direct list
# manipulation.
#
# Once accomplished, some methods become redundant and should be removed in
# favor of the more intuitive direct value list manipulation. Such redundant
# methods include getlist() (removed in favor of values()?), addlist(), and
# setlist().
#
# With the removal of many of the 'list' methods, think about renaming all
# remaining 'list' methods to 'values' methods, like poplist() -> popvalues(),
# poplistitem() -> popvaluesitem(), etc. This would be an easy switch for most
# methods, but wouldn't fit others so well. For example, iterlists() would
# become itervalues(), a name extremely similar to iterallvalues() but quite
# different in function.
#

class BaseCollection(Item, Iterator):
    def __init__(self, seq, *args, **kwargs):
        try:
            iter(seq)
            self.seq = seq
        except:
            self.seq = [seq]
        self.item_dtype = seq[0].__class__
        super().__init__(*args, seq=seq, **kwargs)
        self._i = -1

    def reload(self):
        self._i = -1

    @property
    def state(self):
        return self._i

    def __iter__(self):
        return self

    def __call__(self, *args, **kwargs):
        super(BaseCollection, self).__call__(*args, **kwargs)
        self.item_dtype = self.seq[0].__class__

    def get(self, key):
        return self.seq[key]

    def __getattr__(self, item):
        try:
            cls_attr = getattr(self.item_dtype, item)
            # print(f"Target <{item}> is <{self.item_dtype.__name__}>'s method or base attribute ...")
            if inspect.ismethod(cls_attr):
                # print("It is method!")

                @wraps(cls_attr)
                def wrp(args, **kwargs):
                    for i, slf in enumerate(self.seq):
                        arg = args[i]
                        f = getattr(slf, item)
                        # print(f"Start yielding ...\nwith {self.uid}[{i}] yield {slf.__repr__()}.{item}(args={arg}, kwargs={kwargs})")

                        try:
                            kw = dict(arg)
                            kw |= kwargs
                            yield f(**kw)
                        finally:
                            yield f(*arg, **kwargs)
                    return wrp


            else:
                # print(f"it is base attribute")
                for itm in self.seq:
                    yield getattr(itm, item)
        finally:
            print(
                f"WARN!  Вызываемый аттрибут <{item}> не является обязательным для типа коллекции <{self.item_dtype.__name__}> ."
                "У некоторых элементов он может отсутствовать. В этом случае будет сгенерирован None .")
            for itm in self.seq:
                try:
                    g = getattr(itm, item)
                    yield g
                except:

                    yield None

    def __setattr__(self, k, v):
        for i, itm in enumerate(self):
            setattr(itm, k, v[i])

    def call_item(self, args, **kwargs):
        """Safe call the __call__ method piecewise.
            Pass a list of private arguments to the args, common arguments can be passed as **kwargs."""
        return (item.__call__(args[i], **kwargs) for i, item in enumerate(self.seq))

    def __delitem__(self, key):
        del self.seq[key]

    def __getitem__(self, key):
        return self.seq[key]

    def __setitem__(self, key, value):
        self.seq[key] = value

    def __len__(self):
        return len(self.seq)

    def __next__(self):
        self._i += 1
        if len(self) > self._i:
            return self.seq[self._i]
        else:
            raise StopIteration

    def _fmt(self, prf, pfx, form="({})"):
        bs = "".join([f"{x[0]}={x[1]}" for x in self.__dict__.items()])
        return prf + form.format(bs) + form.format(pfx)

    def __str__(self):
        self._fmt(self.dtype, self.state, form=self.seq)
        return f"{self.dtype}({self.state} in {self.seq})"

    def __repr__(self):
        return f"<{self.dtype}({self.state} in {self.seq}) at {self.uid}>"


class MultiDict(dict):
    """"""

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

    def __setitem__(self, k, v):

        try:
            item = dict.__getitem__(self, k)
            item.append(v)
        except KeyError:
            item = [v]
            dict.__setitem__(self, k, item)
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def __getitem__(self, __k):
        return dict.__getitem__(self, __k)
