import collections.abc
import dataclasses
from collections import deque, namedtuple

__all__ = ["ElementSequence", "ParamContainer", "DoublyLinkedList", "DCLL", "FuncMultiDesc", "CircularLinkedList",
           "CallbackList", "LinkedList", "ParamAccessible", "OrderedSet", "UnlimitedAscii", "ListNode", "CallsHistory",
           "DNode","DCNode","DLLNode","DLLIterator","DCLLIterator"
           ]

from itertools import count


def chain_split_list(iterable):
    ss = deque(iterable)
    l = []
    for i in range(len(ss)):
        print(i)
        a, b, _ = ss[0], ss[1], ss.rotate()
        l.append((a, b))
    return l


class IndexOrderedSet(set):
    """An OrderedFrozenSet-like object
       Allows constant time 'index'ing
       But doesn't allow you to remove elements"""

    def __init__(self, iterable=()):
        self.num = count()
        self.dict = dict(zip(iterable, self.num))

    def add(self, elem):
        if elem not in self:
            self.dict[elem] = next(self.num)

    def extend(self, iterable):

        for i in iterable:
            self.add(i)

    def index(self, elem):
        return self.dict[elem]

    def __contains__(self, elem):
        return elem in self.dict

    def __len__(self):
        return len(self.dict)

    def __iter__(self):
        return iter(self.dict)

    def __repr__(self):
        return 'IndexOrderedSet({})'.format(self.dict.keys())
class OrderedSet(collections.abc.MutableSet):
    def __init__(self, iterable=None):
        self.end = end = []
        end += [None, end, end]  # sentinel node for doubly linked list
        self.map = dict()  # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def extend(self, keys):
        [self.add(key) for key in keys]

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)


import string


class UnlimitedAscii:
    def __init__(self, upper=True):
        if upper:
            self._ascii = string.ascii_uppercase
        else:
            self._ascii = string.ascii_lowercase
        self._ptr = -1

    def __getstate__(self):
        return {"ptr": self._ptr, "sequence": self._ascii}

    def __setstate__(self, state):
        self._ptr = state["ptr"]
        self._ascii = state["sequence"]

    def __iter__(self):
        return self

    def __next__(self):
        self._ptr += 1
        n, c = divmod(self._ptr, len(self._ascii))
        return (n + 1) * self._ascii[c]


import uuid


class Node:
    __match_args__ = "name", "links"

    def __init__(self, name, links=(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.uuid = uuid.uuid4()

        self.name = name
        self.links = OrderedSet(list(links))

    def add_link(self, link):
        self.links.add(link)

    def remove_link(self, link):
        self.links.remove(link)

    def __repr__(self):
        return self.name


class ListNode:
    def __init__(self, name="n", data=None):
        self.uuid = uuid.uuid4()
        self.data = data
        self.next = None
        self.name = name
        self.previous = None

    def __repr__(self):
        return self.name


class LinkedList:
    def __init__(self, nodes=None):
        self.head = None
        if nodes is not None:
            node = ListNode(data=nodes.pop(0))
            self.head = node
            for elem in nodes:
                node.next = ListNode(data=elem)
                node = node.next

    def __repr__(self):
        node = self.head
        nodes = []
        while node is not None:
            nodes.append(node.data)
            node = node.next
        nodes.append("None")
        return " -> ".join([i.__repr__() for i in nodes])

    def __iter__(self):
        node = self.head
        while node is not None:
            yield node
            node = node.next

    def add_first(self, node):
        node.next = self.head
        self.head = node

    def add_last(self, node):
        if self.head is None:
            self.head = node
            return
        for current_node in self:
            pass
        current_node.next = node

    def add_after(self, target_node_data, new_node):
        if self.head is None:
            raise Exception("List is empty")

        for node in self:
            if node.data == target_node_data:
                new_node.next = node.next
                node.next = new_node
                return

        raise Exception("Node with data '%s' not found" % target_node_data)

    def add_before(self, target_node_data, new_node):
        if self.head is None:
            raise Exception("List is empty")
        if self.head.data == target_node_data:
            return self.add_first(new_node)
        prev_node = self.head

        for node in self:

            if node.data == target_node_data:
                prev_node.next = new_node

            new_node.next = node

            return

            prev_node = node
        raise Exception("Node with data '%s' not found" % target_node_data)

    def remove_node(self, target_node_data):
        if self.head is None:
            raise Exception("List is empty")

        if self.head.data == target_node_data:
            self.head = self.head.next
            return

        previous_node = self.head
        for node in self:
            if node.data == target_node_data:
                previous_node.next = node.next
                return
            previous_node = node

        raise Exception("Node with data '%s' not found" % target_node_data)


class CircularLinkedList:
    def __init__(self):
        self.head = None

    def traverse(self, starting_point=None):
        if starting_point is None:
            starting_point = self.head
        node = starting_point
        while node is not None and (node.next != starting_point):
            yield node
            node = node.next
        yield node

    def print_list(self, starting_point=None):
        nodes = []
        for node in self.traverse(starting_point):
            nodes.append(str(node))
        print(" -> ".join(nodes))


from collections.abc import Iterator

from typing import TypeVar
from typing_extensions import TypeVarTuple

Ts = TypeVarTuple('Ts')

T = TypeVar("T")


class ParamContainer:
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._args = list(args)
        self._kwargs = kwargs

    def __call__(self, *args, **kwargs):
        self._args.extend(args)
        self._kwargs.update(kwargs)
        return self

    @property
    def args(self):
        return tuple(self._args)

    @property
    def kwargs(self):
        return dict(self._kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}(args={self.args}, kwargs={self.kwargs})"


from mmcore.collections.multi_description import CallbackList, ElementSequence
import typing
from dataclasses import make_dataclass

ParamTuple = namedtuple("ParamTuple", ["name", "type", "default"])
import strawberry


@dataclasses.dataclass
class Param:
    name: str
    type: typing.Optional[typing.Any] = None
    default: typing.Optional[typing.Any] = None

    def __post_init__(self):
        if self.type is None:
            self.type = typing.Any

    def todict(self):
        return self.to_namedtuple()._asdict()

    def to_namedtuple(self):
        return ParamTuple(self.name, self.type, self.default)

    def to_strawberry(self):
        return strawberry.field(**self.to_namedtuple()._asdict())


def to_tuple(self):
    if dataclasses.is_dataclass(self):
        return dataclasses.astuple(self)
    elif isinstance(self, dict):
        return tuple(self.values())
    else:
        return tuple(self)


def params_eq(self, other):
    return to_tuple(self) == to_tuple(other)


class ParamAccessible:
    params: list[typing.Union[Param, None]]
    schema: typing.Any
    defaults: dict[str, typing.Any]
    params_sequence: ElementSequence[Param]

    def __init__(self, fun):
        super().__init__()
        self.f = fun
        self.name = self.f.__name__
        self.__name__ = self.f.__name__
        self.varnames = OrderedSet(list(self.f.__code__.co_varnames))
        self.__defaults__ = self.f.__defaults__
        self.solve_defaults()

        self.solve_params()
        self.solve_schema()

    def solve_defaults(self):
        self.defaults = dict(zip(list(self.f.__code__.co_varnames)[-len(self.f.__defaults__):], self.f.__defaults__))

    def solve_params(self):
        self.params = []
        for name in self.varnames:
            tp = self.f.__annotations__.get(name)

            df = self.defaults.get(name)
            tp = type(df) if tp is None else tp
            tp = typing.Any if tp is type(None) else tp
            self.params.append(Param(name, type=tp, default=df))
        self.params_sequence = ElementSequence(self.params)

    def solve_schema(self):
        self.schema = make_dataclass(self.name.capitalize() + "ParamSchema",
                                     zip(self.params_sequence["name"],
                                         self.params_sequence["type"],
                                         self.params_sequence["default"]))
        self.schema.to_tuple = property(fget=to_tuple)

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

class CallsHistory(ParamAccessible):
    CallEvent = namedtuple("CallEvent", ["name", "params", "result", "func"])

    def __init__(self, f):

        super().__init__(f)
        self._history = CallbackList(orig=self)

    @property
    def history(self):
        return self._sequence

    def __call__(self, *args, **kwargs):

        kwargs.update(dict(zip(list(self.varnames)[:len(args)], args)))
        s = self.schema(**kwargs)

        if s in self:
            return self.get_result(**kwargs)
        else:
            val = self.f(**kwargs)
            self._history.append(
                self.CallEvent(self.name, params=self.schema(**kwargs), result=val,
                               func=self.f.__code__.co_code.hex())._asdict())

            return val

    def __contains__(self, item):
        try:
            return to_tuple(item) in (to_tuple(i) for i in self.history['params'])
        except AttributeError:
            return False

    def get_result(self, **params):
        return self.history.where(params=self.schema(**params))[-1]["result"]


class FuncMultiDesc:

    def __set_name__(self, owner, name):
        self.name = name
        self.reg_name = "_" + self.name + "_registry"
        setattr(owner, self.reg_name, dict())

    def get_id(self, instance):
        return instance["uuid"]

    def __get__(self, instance, owner):

        def wrap(*args, **kwargs):

            func = getattr(instance.element_type, self.name)

            z = zip(*(instance._seq, args, kwargs))

            try:
                data = []
                for slf, arg, kw in z:
                    print(slf, arg, kw)
                    if not self.get_id(slf) in instance[self.reg_name].keys():
                        instance[self.reg_name][self.get_id(slf)] = ParamContainer(*arg, **kw)
                    a = instance[self.reg_name][self.get_id(slf)]

                    data.append(func(slf, *a.args, **a.kwargs))
                return data
            except:
                for slf, arg, kw in zip(instance._seq, args, kwargs):
                    if not self.get_id(slf) in instance[self.reg_name].keys():
                        instance[self.reg_name][self.get_id(slf)] = ParamContainer(*arg, **kw)
                    a = instance[self.reg_name][self.get_id(slf)]

                    instance[self.reg_name][self.get_id(slf)](slf, *a.args, **a.kwargs)
                return wrap

        return wrap


class Orig:
    @property
    def names(self):
        return self._sequence

    def append_name(self, name):
        self._names.append(name)


class DLLNode:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None


# Function to insert at the end


class DNode:
    def __init__(self, data=None):
        self.data = data
        self.prev = None
        self.next = None

    def __rpr__(self):
        return f'{self.__class__.__name__}({self.data})'

    def __repr__(self):
        rpr = f'[{self.__rpr__()}]'
        if self.next is not None:
            rpr = f'{rpr} -> {self.next.__rpr__()}'

        if self.prev is not None:
            rpr = f'{self.prev.__rpr__()} -> {rpr}'
        return rpr


class DLLIterator(Iterator):
    def __init__(self, dcll: 'DoublyLinkedList'):
        self._dcll = dcll
        self._temp = None


    def __next__(self):

        if self._temp is None:
            self._temp = self._dcll.head
            return self._temp.data


        elif self._temp.next is not None:


            self._temp = self._temp.next
            return self._temp.data


        else:

            raise StopIteration


class DoublyLinkedList:
    __node_type__ = DNode

    def __class_getitem__(cls, item):

        return type(cls.__name__ + f"[{item}]", (cls,), {"__node_type__": item})

    def __init__(self, seq=()):
        super().__init__()

        self.start_node = None
        self.count = 0

        if len(seq) > 0:
            for item in seq:
                self.append(item)
            self.count = len(seq)

        self._current_node = self.start_node

    @property
    def head(self):
        return self.start_node

    def __iter__(self):
        return DLLIterator(self)

    @head.setter
    def head(self, v):
        self.start_node = v

    # Insert Element to Empty list
    def append(self, data):
        self.insert(data)

    def insert(self, data):

        if self.start_node is None:
            new_node = self.__class__.__node_type__(data)

            self.start_node = new_node
        else:
            self.insert_end(data)

    # Insert element at the end
    def insert_end(self, data):
        # Check if the list is empty
        if self.start_node is None:
            new_node = self.__class__.__node_type__(data)
            self.start_node = new_node
            return
        n = self.start_node
        # Iterate till the next reaches NULL
        while True:
            if n.next is None:
                break
            n = n.next
        new_node = self.__class__.__node_type__(data)
        n.next = new_node
        new_node.prev = n
        self.count += 1

    # Delete the elements from the start
    def delete_at_start(self):
        if self.start_node is None:
            #print("The Linked list is empty, no element to delete")
            return
        if self.start_node.next is None:
            self.start_node = None
            return
        self.start_node = self.start_node.next
        self.start_prev = None;
        self.count -= 1

    # Delete the elements from the end
    def delete_at_end(self):
        # Check if the List is empty
        if self.start_node is None:
            #print("The Linked list is empty, no element to delete")
            return
        if self.start_node.next is None:
            self.start_node = None
            return
        n = self.start_node
        while n.next is not None:
            n = n.next
        n.prev.next = None

        self.count -= 1

    def _find_val(self, v):
        temp = self.start_node
        self._i = 0
        while True:
            if temp.data == v:

                break
            elif temp == self.start_node.prev:
                raise ValueError(f"Value not in list: {v}")

            temp = temp.next
            self._i += 1
        return self._i, temp

    def _find_idx(self, i):
        temp = self.head
        self._i = 0
        while True:
            if self._i == i:

                break
            elif temp.next is None:
                temp = None
                break

            temp = temp.next
            self._i += 1

        return temp

    def __getitem__(self, item):
        try:
            return self._find_idx(item).data
        except AttributeError:
            raise IndexError

    def __setitem__(self, key, value):
        node = self.get(key)
        node.prev.next = value
        node.next.prev = value

    def __delitem__(self, key):
        node = self.get(key)
        node.prev.next = node.next
        node.next.prev = node.prev
        del node

    def get(self, item):

        return self._find_idx(item)

    def index(self, item):
        i, _ = self._find_val(item)
        return i

    def remove(self, v):
        if self.start_node is None:
            #print("The Linked list is empty, no element to delete")
            return
        temp = self.start_node
        self._i = 0
        while True:
            if temp.data == v:

                break
            elif temp == self.start_node.prev:
                raise ValueError(f"Value not in list: {v}")

            temp = temp.next
            self._i += 1
        n = temp.next
        p = temp.prev
        n.prev = p
        p.next = n
        del temp
        self.count -= 1

    # Traversing and Displaying each element of the list
    def display(self):
        if self.start_node is None:
            #print("The list is empty")
            return
        else:
            n = self.start_node
            while n is not None:
                #print("Element is: ", n.data)
                n = n.next
        print("\n")

    def __len__(self):
        return self.count


class DCNode(DNode):
    def __init__(self, data=None):
        self.data = data
        self.prev = self
        self.next = self


class DCLLIterator(Iterator):
    def __init__(self, dcll: 'DCLL'):
        self._dcll = dcll
        self._temp = self._dcll.head
        self._i = 0

    def __next__(self):
        if self._i == 0:
            self._temp = self._temp.next
            self._i += 1
            return self._temp.prev.data
        elif self._temp != self._dcll.head:
            self._temp = self._temp.next

            return self._temp.prev.data

        else:

            raise StopIteration


class DCLL:
    """Circular Doubly Linked List
    >>> dcll=DCLL()
    >>> dcll.append(3)
    >>> dcll.append(6)
    >>> dcll.append(9)
    >>> dcll
    DCLL(3) -> 6 -> 9
    >>> dcll.insert_end(4)
    >>> dcll
    DCLL(3) -> 6 -> 9 -> 4
    >>> dcll.insert_end(5)
    >>> dcll.insert_begin(5)
    >>> dcll
    DCLL(5) -> 3 -> 6 -> 9 -> 4 -> 5
    >>> dcll.insert_after(4,6)
    >>> dcll
    DCLL(5) -> 3 -> 6 -> 4 -> 9 -> 4 -> 5
    >>> for i in dcll:
    ...     #print(i)
    5
    3
    6
    4
    9
    4
    5
    >>> a=list(dcll)
    >>> a
    [5, 3, 6, 4, 9, 4, 5]



    """
    nodetype = DCNode

    @classmethod
    def from_list(cls, seq):
        lst = cls()
        for s in seq:
            lst.append(s)

        return lst

    def __init__(self, seq=None):
        self.head = None
        self.count = 0
        self._temp = self.head

        if seq is not None:
            for s in seq:
                self.append(s)


    def reload(self):
        self._temp = self.head

    def __iter__(self):
        return DCLLIterator(self)

    def __repr__(self):
        string = ""

        if (self.head == None):
            string += f"{self.__class__.__qualname__}(<Empty>)"
            return string

        string += f"{self.__class__.__qualname__}({self.head.data})"
        temp = self.head.next
        while (temp != self.head):
            string += \
                f""" -> 
        {temp.data}"""
            temp = temp.next
        return string

    def append(self, data):
        self.insert(data, self.count)

    def insert(self, data, index):
        if (index > self.count) | (index < 0):
            raise ValueError(f"Index out of range: {index}, size: {self.count}")

        if self.head == None:
            self.head = self.nodetype(data)
            self.count = 1
            return

        temp = self.head
        if (index == 0):
            temp = temp.prev
        else:
            for _ in range(index - 1):
                temp = temp.next

        temp.next.prev = self.nodetype(data)
        temp.next.prev.next, temp.next.prev.prev = temp.next, temp
        temp.next = temp.next.prev
        if (index == 0):
            self.head = self.head.prev
        self.count += 1
        return

    def insert_node(self, data, index):
        if (index > self.count) | (index < 0):
            raise ValueError(f"Index out of range: {index}, size: {self.count}")

        if self.head == None:
            self.head = data
            self.count = 1
            return

        temp = self.head
        if (index == 0):
            temp = temp.prev
        else:
            for _ in range(index - 1):
                temp = temp.next

        temp.next.prev = data
        temp.next.prev.next, temp.next.prev.prev = temp.next, temp
        temp.next = temp.next.prev
        if (index == 0):
            self.head = self.head.prev
        self.count += 1
        return
    def remove(self, index):
        if (index >= self.count) | (index < 0):
            raise ValueError(f"Index out of range: {index}, size: {self.count}")

        if self.count == 1:
            self.head = None
            self.count = 0
            return

        target = self.head
        for _ in range(index):
            target = target.next

        if target is self.head:
            self.head = self.head.next

        target.prev.next, target.next.prev = target.next, target.prev
        self.count -= 1

    def index(self, data):
        temp = self.head
        for i in range(self.count):
            if (temp.data == data):
                return i
            temp = temp.next
        return None

    def get(self, index):
        if (index >= self.count) | (index < 0):
            raise ValueError(f"Index out of range: {index}, size: {self.count}")

        temp = self.head
        for _ in range(index):
            temp = temp.next
        return temp

    def __getitem__(self, index):
        val = self.get(index)
        if val is None:
            raise IndexError
        return val.data

    def __setitem__(self, index, val):
        val = self.get_node(index)
        val.data = val


    def size(self):
        return self.count

    def __len__(self):
        return self.count

    def get_node(self, index):
        current = self.head
        for i in range(index):
            current = current.next
            if current == self.head:
                return None
        return current

    def insert_after(self, new_value, value):
        self.count += 1
        new_node = self.nodetype(0)
        new_node.data = new_value  # Inserting the data

        # Find node having value2 and
        # next node of it
        temp = self.head
        while temp.data != value:
            #print(temp, temp.data)
            temp = temp.next
        _next = temp.next

        # insert new_node between temp and next.
        temp.next = new_node
        new_node.prev = temp
        new_node.next = _next
        _next.prev = new_node

    def insert_before_by_index(self, index, new_value):

        self.insert_after(new_value, self.get_node(index).prev.data)
    def insert_before(self, ref_node, new_node):
        self.index(ref_node)

        self.insert_after(ref_node.prev.data, new_node)

    def insert_end(self, value):
        self.count += 1
        # If the list is empty, create a
        # single node circular and doubly list
        if self.head is None:
            new_node = self.nodetype(0)
            new_node.data = value
            new_node.next = new_node.prev = new_node
            self.start = new_node
            return

        # If list is not empty

        # Find last node */
        self.last = self.head.prev

        # Create Node dynamically
        new_node = self.nodetype(0)
        new_node.data = value

        # Start is going to be next of new_node
        new_node.next = self.head

        # Make new node previous of self.start
        self.head.prev = new_node

        # Make last previous of new node
        new_node.prev = self.last

        # Make new node next of old last
        self.last.next = new_node

    def insert_begin(self, value):
        # Pointer points to last Node
        last = self.head.prev
        self.count += 1
        new_node = self.nodetype(0)
        new_node.data = value  # Inserting the data

        # setting up previous and
        # next of new node
        new_node.next = self.head
        new_node.prev = last

        # Update next and previous pointers
        # of self.start and self.last.
        last.next = self.head.prev = new_node

        # Update self.start pointer
        self.head = new_node

    def __str__(self):
        return self.__repr__()

    def __class_getitem__(cls, item):
        return type(f'{cls.__qualname__}[{item.__qualname__}]', (cls,), {'nodetype': item})

    def __contains__(self, item):
        return self.index(item) is not None


def rrange(start, end, count):
    rng = range(0, count)
    step = (end - start) * (1 / ((count - 0) - 1))

    for r in rng:
        yield start + r * step


def addtorange(rng, i):
    for r in rng:
        yield r + 1
