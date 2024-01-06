import numpy as np


def aabb(points: np.ndarray):
    return np.array((np.min(points, axis=len(points.shape) - 2), np.max(points, axis=len(points.shape) - 2)))


aabb_vectorized = np.vectorize(aabb, signature='(i,j)->(k,j)')
