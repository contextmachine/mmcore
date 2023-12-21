from mmcore.ds.cdll.node import Node


class CDLL:
    """

    CDLL
    ====

    Class representing a Doubly Circular Linked List.

    Methods
    -------

    __init__()
        Initializes an empty Doubly Circular Linked List.

    __repr__()
        Returns a string representation of the Doubly Circular Linked List.

    append(data)
        Appends a new element at the end of the Doubly Circular Linked List.

    append_node(node)
        Appends a new node at the end of the Doubly Circular Linked List.

    insert(data, index)
        Inserts a new element at the specified index in the Doubly Circular Linked List.

    iter_nodes()
        Returns an iterator that iterates over the nodes in the Doubly Circular Linked List.

    remove(index)
        Removes the element at the specified index from the Doubly Circular Linked List.

    index(data)
        Returns the index of the first occurrence of the specified element in the Doubly Circular Linked List.

    get(index)
        Returns the element at the specified index in the Doubly Circular Linked List.

    size()
        Returns the number of elements in the Doubly Circular Linked List.

    display()
        Prints a string representation of the Doubly Circular Linked List.

    get_node(index)
        Returns the node at the specified index in the Doubly Circular Linked List.

    """
    nodetype = Node

    def __init__(self):
        self.head = None
        self.count = 0

    def __repr__(self):
        string = ""
        if (self.head == None):
            string += "Doubly Circular Linked List Empty"
            return string
        string += f"Doubly Circular Linked List:\n{self.head.data}"
        temp = self.head.next
        while (temp != self.head):
            string += f" -> {temp.data}"
            temp = temp.next
        return string

    def append(self, data):
        self.insert(data, self.count)
        return

    def append_node(self, node):
        if self.head is None:
            self.head = node
            self.count = 1
        else:
            node.previous = self.head.previous
            self.head.previous.next = node
            self.head.previous = node
            node.next = self.head

            self.count += 1
        return

    def insert(self, data, index):
        if index > self.count or index < 0:
            raise ValueError(f"Index out of range: {index}, size: {self.count}")
        node = self.nodetype(data)
        if self.head is None:
            self.head = node
            node.next = node
            node.previous = node
        else:
            temp = self.head
            for _ in range(index - 1):
                temp = temp.next
            temp.next.previous = node
            node.next, node.previous = temp.next, temp
            temp.next = node
            if index == 0:
                self.head = node
        self.count += 1

    def iter_nodes(self):
        if self.head is not None:

            yield self.head
            temp = self.head.next
            while (temp != self.head):
                yield temp
                temp = temp.next

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
        target.previous.next, target.next.previous = target.next, target.previous
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
        return temp.data

    def size(self):
        return self.count

    def display(self):
        print(self)

    def get_node(self, index):
        if (index >= self.count) | (index < 0):
            raise ValueError(f"Index out of range: {index}, size: {self.count}")
        temp = self.head
        for _ in range(index):
            temp = temp.next
        return temp
