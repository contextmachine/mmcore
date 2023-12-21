# PlaneLinePointSurface Class

## Description

PlaneLinePointSurface is a class in Python inheriting from PlaneSurface class. PlaneLinePointSurface represents a plane
that is defined by a line and a point. It provides functionality to find the u and v directions of the plane, the origin
point, and also to configure the properties of the line and point.

## Class Definitions and Properties

```python
class PlaneLinePointSurface(PlaneSurface):
    def __init__(self, line, point):
        super().__init__()
        self._line = None
        self._point = np.array(point)
        self.line = line

    ...
```

The class constructor accepts two parameters:

- `line`: A Line object. This is used to define one of the directions of the plane.
- `point`: A list [x, y, z] specifying a point on the plane. It is converted to a numpy ndarray datatype.

A PlaneLinePointSurface object has the following properties:

- `u_direction`: Returns the direction of the line that defines this PlaneLinePointSurface.
- `v_direction`: Returns the direction of the line from point to its projection (the closest point) on the line.
- `origin`: Returns the starting point of the line.
- `point`: Gets and sets the point property. When setting this property, it also triggers `solve()` function to update
  the internal state of the object corresponding to the newly set point.
- `line`: Gets and sets the line property. When setting this property, it also triggers `solve()` function to update the
  internal state of the object corresponding to the newly set line.

## Key Functions

```python
def solve(self):
    print(f'[solve] {self}')
    self.line2 = Line.from_ends(self.line.closest_point(self.point), self.point)
```

The `solve()` function is used to create a Line object (named line2) from the given point to the closest point on the
line. The direction of line2 will represent the second direction (v-direction) on the plane.

## Using PlaneLinePointSurface

Here is an example of how to create and use a PlaneLinePointSurface object:

```python
# Assuming Line is a class with necessary properties and methods.
line = Line.from_ends(start=np.array([0, 0, 0], float), end=np.array([1, 0, 0], float))
point = np.array([0, 1, 0], float)

# Create a PlaneLinePointSurface object
plps = PlaneLinePointSurface(line, point)

# Access properties
print(plps.u_direction)
print(plps.v_direction)
print(plps.origin)
print(plps.point)
print(plps.line)

# Change point property
new_point = np.array([0, 2, 0], float)
plps.point = new_point

# Change line property
new_line = Line.from_ends(start=np.array([0, 0, 0], float), end=np.array([2, 0, 0], float))
plps.line = new_line
```
