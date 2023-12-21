class Node:
    def __init__(self, data=None):
        self.data = data
        self.previous = self
        self.next = self

    def __repr__(self):
        return f'{self.__class__}({self.data})'
