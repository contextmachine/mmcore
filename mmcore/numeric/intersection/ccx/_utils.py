from __future__ import annotations

import numpy as np


def _normalize_rects(rects):
    """rects: (N,4) as [s0,t0,s1,t1] (corners in any order).
       returns (N,4) as [s_lo, t_lo, s_hi, t_hi].
    """
    rects = np.asarray(rects, dtype=float).reshape(-1, 4)
    s0, t0, s1, t1 = rects.T
    s_lo = np.minimum(s0, s1)
    s_hi = np.maximum(s0, s1)
    t_lo = np.minimum(t0, t1)
    t_hi = np.maximum(t0, t1)
    return np.stack([s_lo, t_lo, s_hi, t_hi], axis=1)


def _pairwise_overlap(rects, closed=True):
    """Vectorized overlap (N,N) boolean matrix.
       closed=True: touching edges/corners count as overlap.
       closed=False: require strictly positive area intersection.
    """
    s_lo, t_lo, s_hi, t_hi = rects.T
    if closed:
        s_ov = (s_lo[:, None] <= s_hi[None, :]) & (s_lo[None, :] <= s_hi[:, None])
        t_ov = (t_lo[:, None] <= t_hi[None, :]) & (t_lo[None, :] <= t_hi[:, None])
    else:
        s_ov = (s_lo[:, None] < s_hi[None, :]) & (s_lo[None, :] < s_hi[:, None])
        t_ov = (t_lo[:, None] < t_hi[None, :]) & (t_lo[None, :] < t_hi[:, None])
    ov = s_ov & t_ov
    np.fill_diagonal(ov, False)
    return ov


def _connected_components_from_adj(adj):
    """Return component label per node using a simple BFS over a dense boolean adj matrix."""
    n = adj.shape[0]
    labels = -np.ones(n, dtype=int)
    label = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        stack = [i]
        labels[i] = label
        while stack:
            v = stack.pop()
            neigh = np.flatnonzero(adj[v])  # O(n) from the dense row
            unvisited = neigh[labels[neigh] == -1]
            labels[unvisited] = label
            stack.extend(unvisited.tolist())
        label += 1
    return labels


def _merge_by_labels(rects, labels):
    """Merge all rectangles in each connected component to their envelope."""
    s_lo, t_lo, s_hi, t_hi = rects.T
    k = labels.max() + 1
    out = np.empty((k, 4), dtype=float)
    for c in range(k):
        mask = (labels == c)
        out[c, 0] = s_lo[mask].min()
        out[c, 1] = t_lo[mask].min()
        out[c, 2] = s_hi[mask].max()
        out[c, 3] = t_hi[mask].max()
    return out


def _merge_2d_intervals(rects, *, closed=True, max_iter=None):
    """
    Merge axis-aligned 2D intervals into a non-overlapping set.

    Parameters
    ----------
    rects : array_like of shape (N, 4)
        Each row is [s0, t0, s1, t1] (two opposite corners).
    closed : bool, default True
        If True, rectangles that touch at edges/corners are merged.
        If False, only strictly overlapping interiors are merged.
    max_iter : int or None
        Safety cap on iterations (default: N).

    Returns
    -------
    merged : ndarray of shape (M, 4)
        Non-overlapping rectangles as [s_lo, t_lo, s_hi, t_hi].
    """
    rects = _normalize_rects(rects)
    if max_iter is None:
        max_iter = rects.shape[0]
    
    for _ in range(max_iter):
        adj = _pairwise_overlap(rects, closed=closed)
        labels = _connected_components_from_adj(adj)
        # If every rectangle is isolated, we're done
        if (labels.max() + 1) == rects.shape[0]:
            break
        rects = _merge_by_labels(rects, labels)
    return rects
import numpy as np

# ---------- Utilities ----------
def _normalize_minmax_nd(boxes):
    """
    boxes: (N, 2, D) or (N, 4) in degenerate cases
    returns: (N, 2, D) with boxes[:,0,:] <= boxes[:,1,:] per dim
    """
    boxes = np.asarray(boxes, dtype=float)
    if boxes.ndim == 2 and boxes.shape[1] % 2 == 0:
        # allow (N, 2D) => reshape to (N,2,D)
        D = boxes.shape[1] // 2
        boxes = boxes.reshape(-1, 2, D)
    assert boxes.ndim == 3 and boxes.shape[1] == 2, "expected (N, 2, D)"
    lo = np.minimum(boxes[:, 0, :], boxes[:, 1, :])
    hi = np.maximum(boxes[:, 0, :], boxes[:, 1, :])
    return np.stack([lo, hi], axis=1)

def _pairwise_overlap_nd_dense(boxes, *, closed=True):
    """
    Vectorized (N,N) overlap matrix without building (N,N,D) at once.
    boxes: (N, 2, D)
    """
    lo, hi = boxes[:, 0, :], boxes[:, 1, :]
    N, D = lo.shape
    ov = np.ones((N, N), dtype=bool)
    # progressively intersect per-dimension overlaps to avoid N*N*D memory
    for d in range(D):
        if closed:
            cond = (lo[:, None, d] <= hi[None, :, d]) & (lo[None, :, d] <= hi[:, None, d])
        else:
            cond = (lo[:, None, d] <  hi[None, :, d]) & (lo[None, :, d] <  hi[:, None, d])
        ov &= cond
        if not ov.any():
            break
    np.fill_diagonal(ov, False)
    return ov

def _connected_components_from_adj(adj):
    """Connected components on a dense boolean adjacency; returns labels (N,)."""
    n = adj.shape[0]
    labels = -np.ones(n, dtype=int)
    label = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        stack = [i]
        labels[i] = label
        while stack:
            v = stack.pop()
            neigh = np.flatnonzero(adj[v])
            unvisited = neigh[labels[neigh] == -1]
            labels[unvisited] = label
            # extend with Python list to avoid repeated allocations
            stack.extend(unvisited.tolist())
        label += 1
    return labels

def _merge_by_labels_nd(boxes, labels):
    """Merge each component to its envelope. boxes: (N,2,D) -> (K,2,D)."""
    lo, hi = boxes[:, 0, :], boxes[:, 1, :]
    k = labels.max() + 1
    D = lo.shape[1]
    out = np.empty((k, 2, D), dtype=float)
    for c in range(k):
        m = (labels == c)
        out[c, 0, :] = lo[m].min(axis=0)
        out[c, 1, :] = hi[m].max(axis=0)
    return out

# ---------- Dense end-to-end ----------
def merge_intervals_nd(boxes, *, closed=True, max_iter=None):
    """
    Merge axis-aligned D-dimensional intervals (hyperrectangles).

    Parameters
    ----------
    boxes : array_like, shape (N, 2, D)
        boxes[:,0,:] = mins, boxes[:,1,:] = maxs for each dimension
        (order-insensitive; will be normalized).
    closed : bool, default True
        If True, touching boundaries/corners count as overlap.
        If False, require strictly positive overlap in every dimension.
    max_iter : int or None
        Safety cap on merge rounds; default is N.

    Returns
    -------
    merged : ndarray, shape (M, 2, D)
        Non-overlapping boxes (hyperrectangles) after exhaustive merging.
    """
    boxes = _normalize_minmax_nd(boxes)
    if max_iter is None:
        max_iter = boxes.shape[0]
    curr = boxes
    for _ in range(max_iter):
        adj = _pairwise_overlap_nd_dense(curr, closed=closed)
        if not adj.any():
            break
        labels = _connected_components_from_adj(adj)
        next_boxes = _merge_by_labels_nd(curr, labels)
        if next_boxes.shape[0] == curr.shape[0]:
            # No reduction in count; nothing more to merge.
            break
        curr = next_boxes
    return curr

# ---------- Memory-friendly initial pass (blocked union–find) ----------
def _initial_union_blocked_nd(boxes, *, closed=True, block=4096):
    """
    One-pass union-find over (i,j) in blocks, without building full NxN adjacency.
    Returns component labels over the ORIGINAL boxes.
    """
    lo, hi = boxes[:, 0, :], boxes[:, 1, :]
    N, D = lo.shape
    parent = np.arange(N, dtype=int)
    rank = np.zeros(N, dtype=int)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    B = int(block)
    for i0 in range(0, N, B):
        i1 = min(i0 + B, N)
        for j0 in range(i0, N, B):
            j1 = min(j0 + B, N)
            # Start with "all true" (i-block, j-block); intersect per-dim conditions
            Ii, Jj = i1 - i0, j1 - j0
            cond = np.ones((Ii, Jj), dtype=bool)
            for d in range(D):
                A_lo = lo[i0:i1, d][:, None]
                A_hi = hi[i0:i1, d][:, None]
                B_lo = lo[j0:j1, d][None, :]
                B_hi = hi[j0:j1, d][None, :]
                if closed:
                    c = (A_lo <= B_hi) & (B_lo <= A_hi)
                else:
                    c = (A_lo <  B_hi) & (B_lo <  A_hi)
                cond &= c
                if not cond.any():
                    break
            if not cond.any():
                continue
            ii, jj = np.nonzero(cond)
            if i0 == j0:
                # keep only upper triangle (exclude dupes and self-pairs)
                m = (ii < jj)
                ii, jj = ii[m], jj[m]
            for u, v in zip(ii.tolist(), jj.tolist()):
                union(i0 + u, j0 + v)

    roots = np.array([find(i) for i in range(N)], dtype=int)
    _, labels = np.unique(roots, return_inverse=True)
    return labels

def merge_intervals_nd_blocked(boxes, *, closed=True, block=4096, max_iter=None):
    """
    Memory-friendly variant: initial block-wise union–find, then iterate densely on the reduced set.
    Good when N is large (avoids N×N allocations).
    """
    boxes = _normalize_minmax_nd(boxes)
    if max_iter is None:
        max_iter = boxes.shape[0]

    # 1) Initial union in blocks over original N
    labels0 = _initial_union_blocked_nd(boxes, closed=closed, block=block)
    merged = _merge_by_labels_nd(boxes, labels0)

    # 2) Iterate densely until stable (M is usually much < N)
    for _ in range(max_iter):
        adj = _pairwise_overlap_nd_dense(merged, closed=closed)
        if not adj.any():
            break
        labels = _connected_components_from_adj(adj)
        next_boxes = _merge_by_labels_nd(merged, labels)
        if next_boxes.shape[0] == merged.shape[0]:
            break
        merged = next_boxes
    return merged