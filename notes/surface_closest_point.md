# Closest Point on Surface Algorithm with mmcore 
*The full code of this example can be found [(here)](../examples/surface_closest_points.py)*
## Introduction

In the world of Computer-Aided Design (CAD) and computational geometry, finding the closest point on a surface to a given point in space is a fundamental operation. This algorithm, known as the "Closest Point on Surface" algorithm, has numerous applications in areas such as collision detection, surface reconstruction, and path planning. While many CAD software packages implement this algorithm, the implementations are often closed-source or highly complex. In this chapter, we'll explore how to implement this algorithm using mmcore, a lightweight and flexible library for geometric computations.

## Understanding the Problem

Before diving into the implementation, let's clarify what we're trying to achieve. Given a surface (in our case, a ruled surface defined by two NURBS curves) and a set of points in 3D space, we want to find the corresponding points on the surface that are closest to each of these input points.

This problem can be formulated as an optimization problem: for each input point, we need to find the parameters (u, v) on the surface that minimize the distance between the surface point S(u, v) and the input point P.

## Approaches to Implementation

We'll explore two different approaches to implementing this algorithm:

1. A classic approach using spatial partitioning
2. A vectorized approach for batch processing

Both approaches leverage mmcore's capabilities, but they differ in their underlying strategies and potential applications.

### Classic Approach: Spatial Partitioning with BVH

The classic approach utilizes a Bounding Volume Hierarchy (BVH) to accelerate the search process. Here's how it works:

1. We first build a BVH for the surface, which partitions the surface into a tree-like structure of bounding volumes.
2. For each input point, we use the BVH to quickly identify the regions of the surface that are potentially closest to the point.
3. We then use a divide-and-conquer minimization algorithm (`divide_and_conquer_min_2d`) to find the exact closest point within these regions.

This approach is highly efficient due to its ability to narrow down the search area significantly. The use of BVH allows the algorithm to converge in just a few iterations, even for complex surfaces. While we rebuild the BVH tree for each call in this example, in a real-world application, you would typically build the tree once and reuse it, further improving performance.

This method is well-suited for traditional CAD applications and can handle large numbers of points efficiently. Its performance is comparable to, and in some cases may even exceed, that of the vectorized approach.

### Vectorized Approach: Batch Processing

The vectorized approach takes a different strategy, optimized for processing many points simultaneously:

1. Instead of treating each point individually, we formulate the problem as a single optimization over the entire set of input points.
2. We use a vectorized version of the divide-and-conquer algorithm (`divide_and_conquer_min_2d_vectorized`) that can minimize the distance function for all points in parallel.

This approach, despite similar performance to the classical method in this processor implementation, opens up exciting possibilities for modern computational techniques. The key notion here is to present the classical CAD algorithm in a format that is well compatible with modern computational methods, including machine learning workflows.

Although in this tutorial we use conventional NumPy arrays on the CPU, the same approach can easily be adapted to use, for example, PyTorch tensors or other ML frameworks. This would take advantage of GPU acceleration and other features of these actively developing tools.

## Implementation with mmcore

Let's look at how we can implement these approaches using mmcore. We'll start by setting up our problem:

```python
import numpy as np
from mmcore.geom.curves import NURBSpline
from mmcore.geom.surfaces import Ruled
from mmcore.numeric import scalar_dot
from mmcore.numeric.divide_and_conquer import divide_and_conquer_min_2d, divide_and_conquer_min_2d_vectorized
from mmcore.geom.bvh import contains_point
from mmcore.numeric.vectors import dot, norm

def create_ruled_from_points(points, degree=3):
    return Ruled(
        NURBSpline(np.array(points[0], dtype=float), degree=degree),
        NURBSpline(np.array(points[1], dtype=float), degree=degree),
    )


# Define control points and create the ruled surface
cpts = np.array(
    [
        [
            [15.8, 10.1, 0.0],
            [-3.0, 13.0, 0.0],
            [-11.5, 7.7, 0.0],
            [-27.8, 12.8, 0.0],
            [-34.8, 9.3, 0.0],
        ],
        [
            [15.8, 3.6, 10.1],
            [-3.0, 8.6, 15.4],
            [-11.5, 12.3, 19.6],
            [-27.8, 6.3, 16.9],
            [-34.8, 5.0, 16.7],
        ],
    ]
)  # Control points definition

surface = create_ruled_from_points(cpts, degree=3)
```

Now, let's implement our two approaches:

### Classic Approach Implementation

```python
def surface_closest_point_classic_approach(surface, pts, tol=1e-6):
    surface.build_tree(10, 10)
    def objective(u, v):
        d = surface.evaluate(np.array((u, v))) - pt
        return scalar_dot(d, d)

    uvs = np.zeros((len(pts), 2))

    for i, pt in enumerate(pts):
        objects = contains_point(surface.tree, pt)
        if len(objects) == 0:
            uvs[i] = np.array(divide_and_conquer_min_2d(objective, *surface.interval(), tol=tol))
        else:
            uvs_ranges = np.array(list(itertools.chain.from_iterable(o.uvs for o in objects)))
            uvs[i] = np.array(
                divide_and_conquer_min_2d(objective, (np.min(uvs_ranges[..., 0]), np.max(uvs_ranges[..., 0])),
                                          (np.min(uvs_ranges[..., 1]), np.max(uvs_ranges[..., 1])), tol=tol))
    return uvs
```

### Vectorized Approach Implementation

```python
def surface_closest_point_vectorized_approach(surface, pts, tol=1e-6):
    def objective(u, v):
        d = surface(np.array((u, v)).T) - pts
        return np.array(dot(d, d))

    (u_min, u_max), (v_min, v_max) = surface.interval()
    x_range = np.empty((2, len(pts)))
    x_range[0] = u_min
    x_range[1] = u_max
    y_range = np.empty((2, len(pts)))
    y_range[0] = v_min
    y_range[1] = v_max

    uvs = np.array(divide_and_conquer_min_2d_vectorized(objective, x_range=x_range, y_range=y_range, tol=tol))
    return uvs.T
```

## Performance Comparison

To compare the performance of these two approaches, we can generate a set of random points and time the execution of each method:

```python
import time

# Generate random points
pts_count = 100
min_point, max_point = aabb(cpts.reshape(-1, 3)) * 1.25
pts = np.random.uniform(min_point, max_point, size=(pts_count, 3))

# Time the classic approach
s1 = time.time()
uvs_v1 = surface_closest_point_classic_approach(surface, pts, tol=1e-5)
print('Classic approach done in:', time.time() - s1)

# Time the vectorized approach
s2 = time.time()
uvs_v2 = surface_closest_point_vectorized_approach(surface, pts, tol=1e-5)
print('Vectorized approach done in:', time.time() - s2)
```

You'll likely find that the performance of both approaches is quite similar. The classic approach, despite rebuilding the BVH tree for each call, remains highly competitive due to its efficient search space reduction. In a real-world scenario where the BVH tree is prebuilt and reused, the classic approach could potentially outperform the vectorized method for CPU-based computations.

However, it's important to note that the true potential of the vectorized approach lies in its adaptability to GPU acceleration and integration with machine learning workflows, which isn't reflected in this simple timing comparison.

## Accuracy Comparison

We can also compare the accuracy of the two approaches:

```python
# Project points onto the surface
projected_points1 = np.array(surface(uvs_v1))
projected_points2 = np.array(surface(uvs_v2))

# Compare accuracy
classic_error = np.array(norm(projected_points1 - pts))
vectorized_error = np.array(norm(projected_points2 - pts))

if np.all(classic_error < vectorized_error):
    print(f'Classic approach is more accurate by {np.average(classic_error)} units.')
elif np.all(classic_error > vectorized_error):
    print(f'Vectorized approach is more accurate by {np.average(vectorized_error)} units.')
else:
    print(f'Both approaches are about equally accurate.')
```

## Conclusion

In this chapter, we've explored two different approaches to implementing the Closest Point on Surface algorithm using mmcore. Both approaches demonstrate the flexibility and power of mmcore in tackling complex geometric problems.

The classic approach, utilizing BVH, showcases mmcore's ability to implement traditional CAD algorithms with high efficiency. Its performance is robust and scalable, making it an excellent choice for a wide range of CAD applications.

The vectorized approach, while showing similar performance in this CPU-based example, reveals mmcore's potential in bridging the gap between CAD and modern computational techniques. Its batch processing nature makes it particularly well-suited for integration with machine learning workflows, especially when adapted to use GPU-accelerated tensor operations.

By providing these implementations, mmcore offers developers the flexibility to choose the approach that best fits their specific use case:

- The classic approach is ideal for traditional CAD applications, offering high performance and the ability to handle complex surfaces efficiently.
- The vectorized approach opens up new possibilities for developers working at the intersection of CAD and machine learning, providing a pathway to leverage GPU acceleration and integrate geometric computations with neural networks and other ML models.

Both approaches demonstrate mmcore's capabilities in implementing the Closest Point on Surface algorithm. The classic approach is well-suited for traditional CAD applications, while the vectorized approach offers potential for integration with machine learning workflows and GPU acceleration. Users can choose the method that best fits their specific requirements and computational environment.

*The full code of this example can be found [(here)](../examples/surface_closest_points.py)*