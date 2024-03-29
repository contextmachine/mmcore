# PolyCurve

The `PolyCurve` class represents a sequence of connected curves segments, extending a circular
doubly-linked list (CDLL) of curve segments. In the simplest case where each segment is represented by a line
PolyCurve is a polyline.
However, PolyCurve can be composed of any parametric curve segment objects.


---

> **Parametric Curve Segment Protocol** - protocol that have the evaluate method in the range (0., 1.)` and provide
> `start` and `end` descriptors with getter and setter allowing to assign the start and end point of the segment.
>
> Example: `Line`, `Arc`, ...

## Class Definition

```python

class PolyCurve(LineCDLL):
    """
    PolyCurve Class

    This class represents a circular doubly linked list (CDLL) of line segments that form a polyline.

    Attributes:
        nodetype (LineNode): The type of node used in the CDLL.

    Methods:
        __init__(self, pts=None): Initializes a PolyCurve object.
        from_points(cls, pts): Creates a new PolyCurve object from a list of points.
        solve_kd(self): Solves the KDTree for the corners of the polyline.
        insert_corner(self, value, index=None): Inserts a corner at a specified index or at the nearest index.
        corners(self): Returns an array of the corners of the polyline.
        corners(self, corners): Sets the corners of the polyline.
    """
    nodetype = LineNode
```

{
src="mmcore/geom/polycurve.py"
include-symbol="PolyCurve" collapsible="true" validate="false"
}

## Class Attributes

- `nodetype`: Specifies the type of node used in the CDLL. It's typically set to `LineNode`.

## Initialization
```python
polycurve_object = PolyCurve(segments)

```

Where `segments` is a list or an ndarray of parametric curves that compose PolyCurve.

Or you can create PolyCurve from points:

```python
polycurve_object = PolyCurve.from_points(pts)
```

Where `pts` is a list or an ndarray of points that define the PolyCurve.

For example:

```python
import numpy as np
from mmcore.geom.polycurve import PolyCurve

pts = np.array([[0, 0, 0], [1, 1, 0], [2, 2, 0]], dtype=float)
poly = PolyCurve.from_points(pts)
```

## Methods

### `from_points(cls, pts)`

Creates a new instance of PolyCurve from a list of points. This is a class method.
src="../tests/test_polygon_variable_offset.py"}
```python
@classmethod
def from_points(cls, pts):
```

Example:

```python
pts = np.array([[0, 0, 0], [1, 1, 0], [2, 2, 0]], dtype=float)
poly = PolyCurve.from_points(pts)
```

### `solve_kd(self)`

Solves KDTree for the corners of the polyline, allowing efficient point lookup queries.

```python
def solve_kd(self)
```

Example:

```python
kd = poly.solve_kd()
```

### `insert_corner(self, value, index=None)`

Inserts a corner at a specific index or at the nearest index if not specified.

```python
def insert_corner(self, value, index=None)
```

Parameters:

- `value`: The value to be inserted as a corner. It's typically a point in the form of a numpy array or list.
- `index`: The index where the corner should be inserted. If not specified, the method calculates the appropriate index
  based on the KDTree.

Example:

```python
pt = np.array([0.5, 0.5, 0.0])
poly.insert_corner(pt)
```

### `corners(self)`

Returns a numpy array of the corners of the polycurve.



