# mmcore 2D cases

## Statement

Point can be:

1. **Tuple** of three floats `tuple[float, float, float]`
2. **NumPy Array** `ndarray[3, float]` or `ndarray[(n,3), float]`
3. **List** of floats length 3 `list[float]` (not recommended)

## Description

We use a three-coordinate definition for the pointk even though the representation will be treated as a two-dimensional
case. The third coordinate will be ignored. This is done for the following reasons:

- Often we need to perform an operation exactly with projections.

  For example, curves in space rarely intersect, and it is usually easy to understand this without resorting to
  intersection calculations.
- It allows us to use 2d representations as approximations.

  2d representations as approximations,in the three-dimensional case where it is allowed. For example, intersections
  of spheres, planes, or tangents to a sphere, can be obtained by decomposition into 2d cases.

- It increases uniformity.

  `mmcore` is a library working with 3D geometry in the first place. Anything that is computed in the plane in a local
  system will probably soon need to be represented in global or any other coordinates.
