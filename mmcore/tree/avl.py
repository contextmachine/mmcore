import numpy as np
from functools import total_ordering


@total_ordering
class AVLNode(object):
    """
    A node in an avl tree.
    """

    def __init__(self, key, data=None):
        "Construct."

        # The node's key
        self.key = key
        # The node's left child
        self.left = None
        # The node's right child
        self.right = None
        self.data = data

    def __str__(self):
        "String representation."

        return str(self.key)

    def __repr__(self):
        "String representation."
        return f"{self.__class__.__name__}(key={self.key}, data={self.data})"

    def __hash__(self):
        return self.key.__hash__()

    def __eq__(self, other):
        if isinstance(other, AVLNode):
            return self.key == other.key
        else:
            return self.key == other

    def __int__(self):
        return self.key

    def __gt__(self, other):
        if isinstance(other, AVLNode):
            return self.key > other.key
        else:
            return self.key > other

    def __le__(self, other):
        if isinstance(other, AVLNode):
            return self.key <= other.key
        else:
            return self.key <= other

class AVLPrinter:
    def __init__(self, tree: 'AVL', pos=None, buffer=None):
        self.tree = tree
        if buffer is None:
            buffer = np.zeros(
                (
                    1024,
                    1024), dtype=str)
        if pos is None:
            pos = np.array([0, 512])

        self.pos = pos
        self.buffer = buffer

    def __call__(self):
        if self.tree.height >= 1:
            self.process()
            a, b = np.where(self.buffer != '')
            self.buffer[np.where(self.buffer == '')] = ' '
            return "\n".join(''.join(i) for i in self.buffer[np.min(a):np.max(a) + 1, np.min(b) - 1:np.max(b) + 1])

        else:
            return "<Empty AVL tree>"

    def process(self):

        if self.tree.height >= 1:
            ar = list(str(self.tree.node))

            self.buffer[self.pos[0], self.pos[1]:self.pos[1] + len(ar)] = ar

            if self.tree.node.left.height >= 0:

                pos = np.copy(self.pos)
                for i in range(self.tree.height * self.tree.height):
                    pos += np.array((1, -1), dtype=int)
                    self.buffer[pos[0], pos[1]] = '/'

                AVLPrinter(self.tree.node.left, (pos[0] + 1, pos[1] - 1), self.buffer).process()

            if self.tree.node.right.height >= 0:
                pos = np.copy(self.pos)
                for i in range(self.tree.height * self.tree.height):
                    pos += np.array((1, +1), dtype=int)
                    self.buffer[pos[0], pos[1]] = '\\'

                AVLPrinter(self.tree.node.right, (pos[0] + 1, pos[1] + 1), self.buffer).process()

        elif self.tree.node is not None:
            ar = list(str(self.tree.node))

            self.buffer[self.pos[0], self.pos[1]:self.pos[1] + len(ar)] = ar


class AVL(object):
    """
    An avl tree.
    """

    def __init__(self):
        "Construct."

        # Root node of the tree.
        self.node = None
        # Height of the tree.
        self.height = -1
        # Balance factor of the tree.
        self.balance = 0


    def insert(self, key, data=None):
        """
        Insert new key into node
        """
        # Create new node
        n = AVLNode(key, data=data)

        # Initial tree
        if not self.node:
            self.node = n
            self.node.left = AVL()
            self.node.right = AVL()
        # Insert key to the left subtree
        elif key < self.node.key:
            self.node.left.insert(key, data)
        # Insert key to the right subtree
        elif key > self.node.key:
            self.node.right.insert(key, data)

        # Rebalance tree if needed
        self.rebalance()

    def append(self, data=None):
        return self.insert(self.height + 1, data)

    def rebalance(self):
        """
        Rebalance tree. After inserting or deleting a node,
        it is necessary to check each of the node's ancestors for consistency with the rules of AVL
        """

        # Check if we need to rebalance the tree
        #   update height
        #   balance tree
        self.update_heights(recursive=False)
        self.update_balances(False)

        # For each node checked,
        #   if the balance factor remains −1, 0, or +1 then no rotations are necessary.
        while self.balance < -1 or self.balance > 1:
            # Left subtree is larger than right subtree
            if self.balance > 1:

                # Left Right Case -> rotate y,z to the left
                if self.node.left.balance < 0:
                    #     x               x
                    #    / \             / \
                    #   y   D           z   D
                    #  / \        ->   / \
                    # A   z           y   C
                    #    / \         / \
                    #   B   C       A   B
                    self.node.left.rotate_left()
                    self.update_heights()
                    self.update_balances()

                # Left Left Case -> rotate z,x to the right
                #       x                 z
                #      / \              /   \
                #     z   D            y     x
                #    / \         ->   / \   / \
                #   y   C            A   B C   D
                #  / \
                # A   B
                self.rotate_right()
                self.update_heights()
                self.update_balances()

            # Right subtree is larger than left subtree
            if self.balance < -1:

                # Right Left Case -> rotate x,z to the right
                if self.node.right.balance > 0:
                    #     y               y
                    #    / \             / \
                    #   A   x           A   z
                    #      / \    ->       / \
                    #     z   D           B   x
                    #    / \                 / \
                    #   B   C               C   D
                    self.node.right.rotate_right()  # we're in case III
                    self.update_heights()
                    self.update_balances()

                # Right Right Case -> rotate y,x to the left
                #       y                 z
                #      / \              /   \
                #     A   z            y     x
                #        / \     ->   / \   / \
                #       B   x        A   B C   D
                #          / \
                #         C   D
                self.rotate_left()
                self.update_heights()
                self.update_balances()

    def update_heights(self, recursive=True):
        """
        Update tree height

        Tree height is max height of either left or right subtrees +1 for root of the tree
        """
        if self.node:
            if recursive:
                if self.node.left:
                    self.node.left.update_heights()
                if self.node.right:
                    self.node.right.update_heights()

            self.height = 1 + max(self.node.left.height, self.node.right.height)
        else:
            self.height = -1

    def update_balances(self, recursive=True):
        """
        Calculate tree balance factor

        The balance factor is calculated as follows:
            balance = height(left subtree) - height(right subtree).
        """
        if self.node:
            if recursive:
                if self.node.left:
                    self.node.left.update_balances()
                if self.node.right:
                    self.node.right.update_balances()

            self.balance = self.node.left.height - self.node.right.height
        else:
            self.balance = 0

    def rotate_right(self):
        """
        Right rotation
            set self as the right subtree of left subree
        """
        new_root = self.node.left.node
        new_left_sub = new_root.right.node
        old_root = self.node

        self.node = new_root
        old_root.left.node = new_left_sub
        new_root.right.node = old_root

    def rotate_left(self):
        """
        Left rotation
            set self as the left subtree of right subree
        """
        new_root = self.node.right.node
        new_left_sub = new_root.left.node
        old_root = self.node

        self.node = new_root
        old_root.right.node = new_left_sub
        new_root.left.node = old_root

    def search(self, key):
        current = self.node

        while current:
            if key < current.key:
                current = current.left.node
            elif key > current.key:
                current = current.right.node
            else:  # Key found
                return current
        return None

    def delete(self, key):
        """
        Delete key from the tree

        Let node X be the node with the value we need to delete,
        and let node Y be a node in the tree we need to find to take node X's place,
        and let node Z be the actual node we take out of the tree.

        Steps to consider when deleting a node in an AVL tree are the following:

            * If node X is a leaf or has only one child, skip to step 5. (node Z will be node X)
                * Otherwise, determine node Y by finding the largest node in node X's left sub tree
                    (in-order predecessor) or the smallest in its right sub tree (in-order successor).
                * Replace node X with node Y (remember, tree structure doesn't change here, only the values).
                    In this step, node X is essentially deleted when its internal values were overwritten with node Y's.
                * Choose node Z to be the old node Y.
            * Attach node Z's subtree to its parent (if it has a subtree). If node Z's parent is null,
                update root. (node Z is currently root)
            * Delete node Z.
            * Retrace the path back up the tree (starting with node Z's parent) to the root,
                adjusting the balance factors as needed.
        """
        if self.node != None:
            if self.node.key == key:
                # Key found in leaf node, just erase it
                if not self.node.left.node and not self.node.right.node:
                    self.node = None
                # Node has only one subtree (right), replace root with that one
                elif not self.node.left.node:
                    self.node = self.node.right.node
                # Node has only one subtree (left), replace root with that one
                elif not self.node.right.node:
                    self.node = self.node.left.node
                else:
                    # Find  successor as smallest node in right subtree or
                    #       predecessor as largest node in left subtree
                    successor = self.node.right.node
                    while successor and successor.left.node:
                        successor = successor.left.node

                    if successor:
                        self.node.key = successor.key

                        # Delete successor from the replaced node right subree
                        self.node.right.delete(successor.key)

            elif key < self.node.key:
                self.node.left.delete(key)

            elif key > self.node.key:
                self.node.right.delete(key)

            # Rebalance tree
            self.rebalance()

    def inorder_traverse(self):
        """
        Inorder traversal of the tree
            Left subree + root + Right subtree
        """
        result = []

        if not self.node:
            return result

        result.extend(self.node.left.inorder_traverse())
        result.append(self.node)
        result.extend(self.node.right.inorder_traverse())

        return result

    def __repr__(self):
        return AVLPrinter(self)()

    def __str__(self):
        return AVLPrinter(self)()

    def __getitem__(self, item):
        return self.search(item).data

    def __setitem__(self, item, val):
        return self.insert(item, data=val)

    def __delitem__(self, item):
        self.delete(item)




















