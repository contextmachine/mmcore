"""
Balanced KDTree implementation
"""
__all__=["BalancedKDTree", "BKDTree"]


def _update_size(node):
    """Recompute and update the 'size' field based on children."""
    if node is None:
        return 0
    left_size = node.left.size if node.left else 0
    right_size = node.right.size if node.right else 0
    node.size = left_size + right_size + 1
    return node.size


def _collect_points(node, buffer):
    """
    Collect all points in the subtree rooted at 'node'
    into the list 'buffer'.
    """
    if node is None:
        return
    buffer.append(node.point)
    _collect_points(node.left, buffer)
    _collect_points(node.right, buffer)


def _build_balanced(points, depth=0):
    """
    Recursively build a perfectly balanced k-d tree
    from the list of points, splitting on axis = depth % k.
    """
    if not points:
        return None
    k = len(points[0])
    axis = depth % k

    # Sort points by the coordinate in 'axis'
    points.sort(key=lambda x: x[axis])
    median_idx = len(points) // 2

    # Create node and recurse
    node = BalancedKDTree.KDNode(point=points[median_idx], axis=axis)
    node.left = _build_balanced(points[:median_idx], depth + 1)
    node.right = _build_balanced(points[median_idx + 1 :], depth + 1)
    _update_size(node)
    return node


class BalancedKDTree:
    """
    A dynamically balanced k-d tree that supports:
    - building from a list of points,
    - insertion,
    - deletion,
    - nearest neighbor and k-nearest neighbors search.
    """

    class KDNode:
        """
        A node in the k-d tree.

        Attributes:
            point (tuple or list): The actual data point (e.g., (x, y, ...)).
            axis (int): The dimension/axis this node splits on.
            left (KDNode): Left subtree (all points whose axis-value <= this node's axis-value).
            right (KDNode): Right subtree (all points whose axis-value > this node's axis-value).
            size (int): Number of nodes in the subtree rooted at this node (including itself).
        """

        __slots__ = ("point", "axis", "left", "right", "size")

        def __init__(self, point, axis, left=None, right=None):
            self.point = point
            self.axis = axis
            self.left = left
            self.right = right
            self.size = 1  # count self

    def __init__(self, points=None):
        """
        Initialize the KD-tree from an optional list of points.
        If no points are given, the tree starts empty.
        """
        self.root = None
        if points:
            # Ensure all points have the same dimension
            dim = len(points[0])
            for p in points:
                if len(p) != dim:
                    raise ValueError("All points must have the same dimension.")
            self.root = _build_balanced(points, depth=0)

    def _is_unbalanced(self, node):
        """
        Check if the subtree at 'node' is unbalanced.
        We use a simple size-based criterion: if either subtree
        is more than 'alpha' fraction of the total, we consider it unbalanced.
        """
        if node is None:
            return False
        left_size = node.left.size if node.left else 0
        right_size = node.right.size if node.right else 0
        total = node.size
        alpha = 0.7  # rebalance threshold (tweak as desired)

        # If either subtree is too big relative to the whole, unbalanced
        if left_size > alpha * total or right_size > alpha * total:
            return True
        return False

    def _rebalance_subtree(self, node, depth=0):
        """
        Rebuild the subtree rooted at 'node' from scratch
        (collect all points, re-build balanced).
        """
        if node is None:
            return None
        points = []
        _collect_points(node, points)
        return _build_balanced(points, depth)

    def _insert_rec(self, node, point, depth=0):
        """
        Recursively insert 'point' into subtree rooted at 'node'.
        Then check for balance; if unbalanced, rebuild the subtree.
        """
        if node is None:
            # Create a new leaf
            axis = depth % len(point)
            return BalancedKDTree.KDNode(point, axis)

        axis = node.axis
        if point[axis] <= node.point[axis]:
            node.left = self._insert_rec(node.left, point, depth + 1)
        else:
            node.right = self._insert_rec(node.right, point, depth + 1)

        _update_size(node)

        # Rebalance if needed
        if self._is_unbalanced(node):
            node = self._rebalance_subtree(node, depth)
        return node

    def insert(self, point):
        """
        Insert a single point into the k-d tree.
        """
        if self.root is None:
            self.root = BalancedKDTree.KDNode(point, axis=0)
            self.root.size = 1
        else:
            if len(point) != len(self.root.point):
                raise ValueError("Point dimension must match the KD-tree.")
            self.root = self._insert_rec(self.root, point, depth=0)

    def _find_min(self, node, axis, depth=0):
        """
        Return the node with the minimum value along 'axis'
        in the subtree rooted at 'node'.
        """
        if node is None:
            return None

        current_axis = node.axis

        if current_axis == axis:
            # The minimum is in the left subtree if it exists;
            # otherwise, the current node is the minimum
            if node.left is None:
                return node
            return self._find_min(node.left, axis, depth + 1)
        else:
            # The minimum could be anywhere
            left_min = self._find_min(node.left, axis, depth + 1)
            right_min = self._find_min(node.right, axis, depth + 1)

            candidates = [n for n in (node, left_min, right_min) if n is not None]
            # Return the node with the smallest value along 'axis'
            return min(candidates, key=lambda x: x.point[axis])

    def _delete_rec(self, node, point, depth=0):
        """
        Recursively delete 'point' in the subtree rooted at 'node'.
        If multiple points are identical, this deletes the first found
        in the search path.

        For a k-d tree, standard approach is:
        1) If the node to delete is found,
           - if it has a right subtree, find min in that subtree along 'node.axis',
             replace node's point with that min, and recursively delete that min.
           - else if it has a left subtree, find min in the left subtree along 'node.axis',
             replace node's point with that min, and make left->right the new left.
           - else, it's a leaf and can be removed directly (return None).
        2) If the node to delete is not found at current node, recurse left or right.
        Then we check for unbalance and possibly rebuild.
        """
        if node is None:
            return None

        current_axis = node.axis

        if node.point == point:
            # Found the node to remove
            # Case 1: If there's a right subtree, find minimum in that subtree along 'current_axis'
            if node.right is not None:
                min_node = self._find_min(node.right, current_axis, depth + 1)
                node.point = min_node.point
                node.right = self._delete_rec(node.right, min_node.point, depth + 1)
            elif node.left is not None:
                # Case 2: If there's no right subtree but a left subtree,
                # find min in the left subtree along 'current_axis'
                min_node = self._find_min(node.left, current_axis, depth + 1)
                node.point = min_node.point
                # Then, we have to remove that min from the left subtree
                node.left = self._delete_rec(node.left, min_node.point, depth + 1)
            else:
                # Case 3: Leaf node
                return None
        else:
            # Not the node to delete, search left or right
            if point[current_axis] <= node.point[current_axis]:
                node.left = self._delete_rec(node.left, point, depth + 1)
            else:
                node.right = self._delete_rec(node.right, point, depth + 1)

        _update_size(node)

        # Rebalance if needed
        if self._is_unbalanced(node):
            node = self._rebalance_subtree(node, depth)

        return node

    def delete(self, point):
        """
        Delete 'point' from the k-d tree (if it exists).
        """
        if self.root is None:
            return
        if len(point) != len(self.root.point):
            raise ValueError("Point dimension must match the KD-tree.")
        self.root = self._delete_rec(self.root, point, depth=0)

    def _distance_sq(self, p1, p2):
        """Return squared Euclidean distance between p1 and p2."""
        return sum((c1 - c2) ** 2 for c1, c2 in zip(p1, p2))

    def _nearest_rec(self, node, target, best, depth=0):
        """
        Recursively find the nearest neighbor to 'target'
        in the subtree rooted at 'node'.

        'best' is a tuple (best_node, best_dist_sq).
        """
        if node is None:
            return best

        current_dist_sq = self._distance_sq(target, node.point)
        if current_dist_sq < best[1]:
            best = (node, current_dist_sq)

        axis = node.axis
        diff = target[axis] - node.point[axis]

        # Choose which side to explore first
        if diff <= 0:
            best = self._nearest_rec(node.left, target, best, depth + 1)
            # Only explore right side if it can contain a closer point
            if diff * diff < best[1]:
                best = self._nearest_rec(node.right, target, best, depth + 1)
        else:
            best = self._nearest_rec(node.right, target, best, depth + 1)
            if diff * diff < best[1]:
                best = self._nearest_rec(node.left, target, best, depth + 1)

        return best

    def nearest_neighbor(self, point):
        """
        Return the closest point in the KD-tree to 'point'.
        If the tree is empty, returns None.
        """
        if self.root is None:
            return None
        best_node, best_dist_sq = self._nearest_rec(
            self.root, point, (None, float("inf"))
        )
        return best_node.point

    def _k_nearest_rec(self, node, target, k, heap, depth=0):
        """
        Recursively search for the k nearest neighbors.

        We maintain a max-heap (by negative distance) of size up to k.
        """
        if node is None:
            return

        dist_sq = self._distance_sq(node.point, target)
        # If we haven't found k neighbors yet, push this one
        if len(heap) < k:
            from heapq import heappush

            heappush(heap, (-dist_sq, node.point))
        else:
            # If this point is closer than the worst in the heap, replace
            if dist_sq < -heap[0][0]:
                from heapq import heapreplace

                heapreplace(heap, (-dist_sq, node.point))

        axis = node.axis
        diff = target[axis] - node.point[axis]

        # Recurse on the side of the split that target is in
        if diff <= 0:
            self._k_nearest_rec(node.left, target, k, heap, depth + 1)
            # Check other side if needed
            if diff * diff < -heap[0][0]:
                self._k_nearest_rec(node.right, target, k, heap, depth + 1)
        else:
            self._k_nearest_rec(node.right, target, k, heap, depth + 1)
            if diff * diff < -heap[0][0]:
                self._k_nearest_rec(node.left, target, k, heap, depth + 1)

    def k_nearest_neighbors(self, point, k=1):
        """
        Return the k closest points in the KD-tree to 'point'.
        If the tree is empty, returns an empty list.
        """
        if self.root is None or k <= 0:
            return []
        from heapq import heappop

        max_heap = []
        self._k_nearest_rec(self.root, point, k, max_heap, depth=0)

        # max_heap is a max-heap of size up to k (by negative dist), pop them all
        results = []
        while max_heap:
            neg_dist, pt = heappop(max_heap)
            results.append(pt)
        # The closest point is the last popped, but typically
        # we want the neighbors sorted from closest -> furthest:
        results.reverse()
        return results

BKDTree=BalancedKDTree


# ------------------------------------------------------------------------------
# Example usage (comment out or remove if you just need the class definition):
if __name__ == "__main__":
    data = [
        (2.1, 3.1),
        (5.4, 1.3),
        (9.2, 8.1),
        (4.2, 7.2),
        (8.5, 9.0),
        (1.4, 5.9),
        (3.7, 4.9),
        (6.6, 2.1),
    ]
    kd = BalancedKDTree(data)

    print("Initial nearest to (3,3):", kd.nearest_neighbor((3, 3)))
    kd.insert((3.0, 2.9))
    print("After inserting (3, 2.9), nearest to (3,3):", kd.nearest_neighbor((3, 3)))
    kd.delete((2.1, 3.1))
    print("After deleting (2.1, 3.1), nearest to (3,3):", kd.nearest_neighbor((3, 3)))

    print("3 nearest neighbors to (4, 4):", kd.k_nearest_neighbors((4, 4), k=3))
