class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            # Union by rank
            if self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            elif self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1


def group_tuples(tuples_list):
    """
    Given a list of tuples of integers, return the groups
    (connected components) of tuples such that any two tuples
    that share a digit are in the same group.
    """
    n = len(tuples_list)
    uf = UnionFind(n)

    # Map from digit -> list of indices of tuples that contain this digit
    digit_to_indices = {}

    # Step 1: Build the mapping
    for i, tup in enumerate(tuples_list):
        for num in tup:
            if num not in digit_to_indices:
                digit_to_indices[num] = []
            digit_to_indices[num].append(i)

    # Step 2: Union all tuples that share a digit
    for num, indices in digit_to_indices.items():
        # Union all indices among themselves
        first_idx = indices[0]
        for other_idx in indices[1:]:
            uf.union(first_idx, other_idx)

    # Step 3: Gather by connected components
    # find the representative (parent) of each index
    from collections import defaultdict

    root_to_group = defaultdict(list)
    for i, tup in enumerate(tuples_list):
        root = uf.find(i)
        root_to_group[root].append(tup)

    # The values in root_to_group are our groups
    return list(root_to_group.values())


# -------------- TESTING --------------

if __name__ == "__main__":
    # Example 1
    tuples1 = [(0, 1), (1, 2), (3, 4, 5), (3, 6), (2, 7)]
    groups1 = group_tuples(tuples1)
    print("Groups (Case 1):")
    for g in groups1:
        print(g)
    # Expecting two groups: [[(0,1), (1,2), (2,7)], [(3,4,5), (3,6)]]

    # Example 2
    tuples2 = [(0, 1), (1, 2), (3, 4, 5), (3, 6), (2, 6)]
    groups2 = group_tuples(tuples2)
    print("\nGroups (Case 2):")
    for g in groups2:
        print(g)
    # Expecting one group: [[(0,1), (1,2), (3,4,5), (3,6), (2,6)]]
