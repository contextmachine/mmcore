import collections.abc


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
    def __init__(self, name, data=None):
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
            node = Node(data=nodes.pop(0))
            self.head = node
            for elem in nodes:
                node.next = Node(data=elem)
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
