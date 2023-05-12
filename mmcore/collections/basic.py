import collections.abc
import dataclasses
from collections import namedtuple, deque


def chain_split_list(iterable):
    ss = deque(iterable)
    l = []
    for i in range(len(ss)):
        print(i)
        a, b, _ = ss[0], ss[1], ss.rotate()
        l.append((a, b))
    return l


class OrderedSet(collections.abc.MutableSet):
    def __init__(self, iterable=None):
        self.end = end = []
        end += [None, end, end]  # sentinel node for doubly linked list
        self.map = {}  # key --> [key, prev, next]
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


class Graph:
    ...


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


from collections.abc import Iterator, Container


class Grouper(Iterator):
    def __init__(self, iterable):
        self._itr = iterable
        self._iterable = enumerate(iterable)
        self.data = {}

    def __iter__(self):
        return self

    def __next__(self):
        i, v = self._iterable.__next__()
        self._wrp(v, i)

    def release(self):
        def wrp(s, data):
            yield data
            del s

        return wrp(self, iter(self.data))

    def get_counter(self):
        return collections.Counter(self._itr)

    def _wrp(self, key, item):
        if self.data.get(key) is None:
            self.data[key] = []
        self.data[key].append(item)


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
from dataclasses import make_dataclass, field

ParamTuple = namedtuple("ParamTuple", ["name", "type", "default"])
import strawberry


class Cond:
    def convert(self, val):
        return val

    def __call__(self, val):
        return isinstance(val, dict)


class Cond2:
    def convert(self, val):
        return dataclasses.asdict(val)

    def __call__(self, val):
        return dataclasses.is_dataclass(val)


class Cond3:
    def convert(self, val):
        return dataclasses.asdict(val)

    def __call__(self, val):
        return dataclasses.is_dataclass(val)


class Convertor:
    def __init__(self, *conds):
        self.conds = conds

    def convert(self, val):
        i = 0
        while True:
            if self.conds[i](val):

                v = self.conds[i].convert(val)
                break
            else:
                i += 1
        return v


@dataclasses.dataclass
class Param:
    name: str
    type: typing.Optional[typing.Any] = None
    default: typing.Optional[typing.Any] = None

    def __post_init__(self):
        if self.type is None:
            self.type = typing.Any

    def to_dict(self):
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


import copy


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


class BoolMask(Container):
    _masked = OrderedSet()
    name: str = None

    def __get__(self, instance, owner):
        return instance in self

    def __set__(self, instance, value: bool):
        self.add_to_mask(instance) if value else self.remove_from_mask(val=instance)

    def __contains__(self, __x: object) -> bool:
        return not self.get_id(__x) in self._masked

    def get_id(self, val):
        return val if type(val) is str else hex(id(val))

    def add_to_mask(self, val):
        k = self.get_id(val)
        self._masked.add(k)
        return k

    def extend_mask(self, val):
        *ks, = (self.get_id(v) for v in val)
        self._masked.extend(ks)
        return ks

    def set_mask(self, val):
        ks = [self.get_id(v) for v in val]
        self._masked = ks
        return ks

    def remove_from_mask(self, val):
        k = self.get_id(val)
        self._masked.remove(k)
        return k

    def removes_from_mask(self, vals):
        removed = []
        for val in vals:
            removed.append(self.remove_from_mask(val))
        return removed


from functools import wraps


def curry(func):
    """
    >>> @curry
    ... def foo(a, b, c):
    ...     return a + b + c
    >>> foo(1)
    <function __main__.foo>
    """

    @wraps(func)
    def curried(*args, **kwargs):
        if len(args) + len(kwargs) >= func.__code__.co_argcount:
            return func(*args, **kwargs)

        @wraps(func)
        def new_curried(*args2, **kwargs2):
            return curried(*(args + args2), **dict(kwargs, **kwargs2))

        return new_curried

    return curried


class ComposeMask:

    def __init__(self, masks):
        super().__init__()
        self.masks = masks

    def _per_mask(self, item):
        return (item in mask for mask in self.masks)

    def elementwise(self, item):
        return zip((mask.name for mask in self.masks), self._per_mask(item))

    def match(self, item):
        ...

    def __contains__(self, item) -> int:
        """

        @param item:
        @return:
        match self.elementwise(item)->int:
            case {"a": True, "b":False}:
            return 1:
        """
        ...


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

    def __repr__(self):
        return f'DNode(...) -> [DNode({self.data})] -> DNode(...)'


class DLLIterator(Iterator):
    def __init__(self, dcll: 'DoublyLinkedList'):
        self._dcll = dcll
        self._temp = self._dcll.start_node
        self._i = 0

    def __next__(self):

        if self._temp is not None:
            self._temp = self._temp.next
            return self._temp.prev
        else:

            raise StopIteration


class DoublyLinkedList:
    def __init__(self):
        self.start_node = None

    # Insert Element to Empty list
    def append(self, data):
        self.insert(data)

    def insert(self, data):
        if self.start_node is None:
            new_node = DNode(data)

            self.start_node = new_node
        else:
            self.insert_end(data)

    # Insert element at the end
    def insert_end(self, data):
        # Check if the list is empty
        if self.start_node is None:
            new_node = DNode(data)
            self.start_node = new_node
            return
        n = self.start_node
        # Iterate till the next reaches NULL
        while True:
            if n.next is None:
                break
            n = n.next
        new_node = DNode(data)
        n.next = new_node
        new_node.prev = n

    # Delete the elements from the start
    def delete_at_start(self):
        if self.start_node is None:
            print("The Linked list is empty, no element to delete")
            return
        if self.start_node.next is None:
            self.start_node = None
            return
        self.start_node = self.start_node.next
        self.start_prev = None;

    # Delete the elements from the end
    def delete_at_end(self):
        # Check if the List is empty
        if self.start_node is None:
            print("The Linked list is empty, no element to delete")
            return
        if self.start_node.next is None:
            self.start_node = None
            return
        n = self.start_node
        while n.next is not None:
            n = n.next
        n.prev.next = None

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
        temp = self.start_node
        self._i = 0
        while True:
            if self._i == i or (temp.next is None):
                break

            temp = temp.next
            self._i += 1

        return temp

    def __getitem__(self, item):
        return self._find_idx(item).data

    def get(self, item):

        return self._find_idx(item)

    def index(self, item):
        i, _ = self._find_val(item)
        return i

    def remove(self, v):
        if self.start_node is None:
            print("The Linked list is empty, no element to delete")
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

    # Traversing and Displaying each element of the list
    def display(self):
        if self.start_node is None:
            print("The list is empty")
            return
        else:
            n = self.start_node
            while n is not None:
                print("Element is: ", n.data)
                n = n.next
        print("\n")


class DCNode(DNode):
    def __init__(self, data=None):
        self.data = data
        self.prev = self
        self.next = self

    def __repr__(self):
        return f'DCNode({self.prev.data}) -> [DCNode({self.data})] -> DCNode({self.next.data})'


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
    ...     print(i)
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

    @classmethod
    def from_list(cls, seq):
        lst = cls()
        for s in seq:
            lst.append(s)

        return lst

    def __init__(self):

        self.head = None
        self.count = 0
        self._temp = self.head

    def reload(self):
        self._temp = self.head

    def __iter__(self):
        return DCLLIterator(self)

    def __repr__(self):
        string = ""

        if (self.head == None):
            string += "DCLL(<Empty>)"
            return string

        string += f"DCLL({self.head.data})"
        temp = self.head.next
        while (temp != self.head):
            string += f" -> {temp.data}"
            temp = temp.next
        return string

    def append(self, data):
        self.insert(data, self.count)

    def insert(self, data, index):
        if (index > self.count) | (index < 0):
            raise ValueError(f"Index out of range: {index}, size: {self.count}")

        if self.head == None:
            self.head = DCNode(data)
            self.count = 1
            return

        temp = self.head
        if (index == 0):
            temp = temp.prev
        else:
            for _ in range(index - 1):
                temp = temp.next

        temp.next.prev = DCNode(data)
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

    def insert_after(self, value1, value2):
        self.count += 1
        new_node = DCNode(0)
        new_node.data = value1  # Inserting the data

        # Find node having value2 and
        # next node of it
        temp = self.head
        while temp.data != value2:
            print(temp, temp.data)
            temp = temp.next
        _next = temp.next

        # insert new_node between temp and next.
        temp.next = new_node
        new_node.prev = temp
        new_node.next = _next
        _next.prev = new_node

    def insert_before(self, ref_node, new_node):

        self.insert_after(ref_node.prev.data, new_node)

    def insert_end(self, value):
        self.count += 1
        # If the list is empty, create a
        # single node circular and doubly list
        if self.head is None:
            new_node = DCNode(0)
            new_node.data = value
            new_node.next = new_node.prev = new_node
            self.start = new_node
            return

        # If list is not empty

        # Find last node */
        self.last = self.head.prev

        # Create Node dynamically
        new_node = DCNode(0)
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
        new_node = DCNode(0)
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
