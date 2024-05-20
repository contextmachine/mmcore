import numpy as np

class Box:
    def __init__(self, minx, miny, minz, maxx, maxy, maxz):
        self.minx = minx
        self.miny = miny
        self.maxx = maxx
        self.maxy = maxy
        self.minz = minz
        self.maxz = maxz

    def area(self):
        return (self.maxx - self.minx) * (self.maxy - self.miny) * (self.maxz - self.minz)

    def perimeter(self):
        return 2 * (self.maxx - self.minx + self.maxy - self.miny + self.maxz - self.minz)

    def intersect(self, other):
        return not (self.minx > other.maxx or self.maxy < other.miny or
                    self.maxx < other.minx or self.miny > other.maxy or self.minz > other.maxz or self.maxz < other.minz

                    )

    def union(self, other):
        return Box(min(self.minx, other.minx), min(self.miny, other.miny), min(self.minz, other.minz),
                   max(self.maxx, other.maxx), max(self.maxy, other.maxy), max(self.maxz, other.maxz))

    def enlarge(self, other):
        self.minx = min(self.minx, other.minx)
        self.miny = min(self.miny, other.miny)
        self.minz = min(self.minz, other.minz)
        self.maxx = max(self.maxx, other.maxx)
        self.maxy = max(self.maxy, other.maxy)
        self.maxz = max(self.maxz, other.maxz)
class Node:
    def __init__(self, is_leaf=False, mbr=None):
        self.is_leaf = is_leaf
        self.children = []
        self.mbr = mbr  # Minimum bounding rectangle

    def add_child(self, node, update_mbr=True):
        self.children.append(node)
        if update_mbr:
            self.update_mbr()

    def update_mbr(self):
        if self.children:
            mbr = self.children[0].mbr
            for child in self.children[1:]:
                mbr = mbr.union(child.mbr)
            self.mbr = mbr
        else:
            self.mbr = None


class RStarTree:
    def __init__(self, max_children=4, min_children=2):
        self.root = Node(is_leaf=True)
        self.max_children = max_children
        self.min_children = min_children

    def insert(self, rectangle):
        node = self._choose_leaf(self.root, rectangle)
        node.add_child(Node(is_leaf=True, mbr=rectangle))
        if len(node.children) > self.max_children:
            self._split_node(node)

    def _choose_leaf(self, node, rectangle):
        if node.is_leaf:
            return node
        else:
            best_child = None
            best_enlargement = None
            for child in node.children:
                enlargement = child.mbr.union(rectangle).area() - child.mbr.area()
                if best_child is None or enlargement < best_enlargement:
                    best_child = child
                    best_enlargement = enlargement
            return self._choose_leaf(best_child, rectangle)

    def _split_node(self, node):
        k = len(node.children)
        axis = self._choose_split_axis(node)

        # Sort children by the selected axis
        node.children.sort(key=lambda child: getattr(child.mbr, axis[0]))

        # Choose the best distribution of entries among the two nodes
        best_distribution = None
        best_margin = None
        for i in range(self.min_children, k - self.min_children + 1):
            left = node.children[:i]
            right = node.children[i:]
            left_mbr = self._compute_mbr(left)
            right_mbr = self._compute_mbr(right)
            margin = left_mbr.area() + right_mbr.area()
            if best_margin is None or margin < best_margin:
                best_distribution = (left, right)
                best_margin = margin

        # Split node into two
        node.children = best_distribution[0]
        node.update_mbr()
        new_node = Node(is_leaf=node.is_leaf)
        new_node.children = best_distribution[1]
        new_node.update_mbr()

        if node == self.root:
            # Create a new root
            new_root = Node()
            new_root.add_child(node, update_mbr=False)
            new_root.add_child(new_node, update_mbr=False)
            self.root = new_root
        else:
            parent = self._find_parent(self.root, node)
            parent.add_child(new_node)
            if len(parent.children) > self.max_children:
                self._split_node(parent)

    def _choose_split_axis(self, node):
        # Calculate margin values for both x and y axis
        margin_x = self._compute_margin(node.children, 'x')
        margin_y = self._compute_margin(node.children, 'y')
        return ('minx', 'maxx') if margin_x <= margin_y else ('miny', 'maxy')

    def _compute_margin(self, children, axis):
        margins = 0
        k = len(children)
        children.sort(key=lambda child: getattr(child.mbr, 'min' + axis))
        for i in range(self.min_children, k - self.min_children + 1):
            left = self._compute_mbr(children[:i])
            right = self._compute_mbr(children[i:])
            margins += left.perimeter() + right.perimeter()
        return margins

    def _compute_mbr(self, children):
        mbr = children[0].mbr
        for child in children[1:]:
            mbr = mbr.union(child.mbr)
        return mbr

    def _find_parent(self, current_node, target_node):
        if current_node.is_leaf:
            return None
        for child in current_node.children:
            if child == target_node or self._find_parent(child, target_node):
                return current_node
        return None

    def search(self, rectangle):
        return self._search_subtree(self.root, rectangle)

    def _search_subtree(self, node, rectangle):
        results = []
        if node.is_leaf:
            for child in node.children:
                if child.mbr.intersect(rectangle):
                    results.append(child.mbr)
        else:
            for child in node.children:
                if child.mbr.intersect(rectangle):
                    results.extend(self._search_subtree(child, rectangle))
        return results


class Rectangle:
    def __init__(self, minx, miny, maxx, maxy):
        self.minx = minx
        self.miny = miny
        self.maxx = maxx
        self.maxy = maxy

    def area(self):
        return (self.maxx - self.minx) * (self.maxy - self.miny)

    def perimeter(self):
        return 2 * (self.maxx - self.minx + self.maxy - self.miny)

    def intersect(self, other):
        return not (self.minx > other.maxx or self.maxy < other.miny or
                    self.maxx < other.minx or self.miny > other.maxy)

    def union(self, other):
        return Rectangle(min(self.minx, other.minx), min(self.miny, other.miny),
                         max(self.maxx, other.maxx), max(self.maxy, other.maxy))

    def enlarge(self, other):
        self.minx = min(self.minx, other.minx)
        self.miny = min(self.miny, other.miny)
        self.maxx = max(self.maxx, other.maxx)
        self.maxy = max(self.maxy, other.maxy)



if __name__ == '__main__':
    # Assuming we've already defined the RStarTree, Node, and Rectangle classes as described.

    # Create an instance of R*-Tree
    rstar_tree = RStarTree(max_children=4, min_children=2)

    # Define some rectangles to insert
    rectangles = [
        Rectangle(1, 1, 2, 2),
        Rectangle(2, 3, 3, 4),
        Rectangle(3, 1, 4, 2),
        Rectangle(0, 0, 1, 1),
        Rectangle(5, 5, 6, 6)
    ]

    # Insert rectangles into the R*-Tree
    for rect in rectangles:
        rstar_tree.insert(rect)

    # Perform a search query
    search_rect = Rectangle(1.5, 1.5, 4, 4)
    results = rstar_tree.search(search_rect)

    # Print the search results
    print("Search results:")
    for result in results:
        print(search_rect.intersect(result))
        print(f"Rectangle with minx: {result.minx}, miny: {result.miny}, maxx: {result.maxx}, maxy: {result.maxy}")
