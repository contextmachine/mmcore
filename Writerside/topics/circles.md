# Circle Interactions User Guide

This guide provides instructions on how to create 2D circles, find intersection points, tangent lines, and the closest
points on the circle.
> _See the section on [point representation](2d_cases.md) if three-dimensional notation in 2D cases confused you._

## Create a Circle

Circles are created using the `Circle` class. The circle requires two arguments: the radius and the origin point (as a
list of two coordinates [x, y]).

```python
from mmcore.geom.circle import Circle

# Create a Circle object with radius 5 and origin at (0,0)
circle = Circle(5, np.array([0., 0., 0.]))
```

## Find Intersection Points of Two Circles

To find the intersection points between two circles in a 2D space, you use the `circle_intersection2d` function. It
takes two `Circle` objects as arguments.

```python
from mmcore.geom.circle import circle_intersection2d

# Create another Circle object

circle2 = Circle(7., np.array([10., 10., 0.]))

# Get the intersection points between circle and circle2
intersection = circle_intersection2d(circle, circle2)

print(intersection)
```

## Calculate Tangent Lines from a Point to a Circle

You can find the tangent lines from a point to a circle using the `tangents_lines` function.

```python
import numpy as np
from mmcore.geom.curves import tangents_lines

# Define a point
#
point = np.array([15., 15., 0.])

# Compute tangent lines from point to circle
tangent_pts = tangents_lines(point, circle)

print(tangent_pts)
```

<img alt="Tangent lines from point to circle" height="400" src="circle_tan.svg" title="Tangent Lines"/>

The results are the tangent lines from point to the circle.

## Calculate Closest Point on a Circle from a Point

You can find the closest point on a circle from any point in the 2D space using the `closest_point_on_circle_2d`
function.

```python
from mmcore.geom.curves import closest_point_on_circle_2d

# Define a point
point = np.array([20., 20., 0.])

# Compute the closest point parametr on circle from point
t = closest_point_on_circle_2d(point, circle)
# Compute the closest point from parameter
closest_pt = c1(t)
print(t, closest_pt)
```

Remember to import the required functions (`circle_intersection2d`, `tangents_lines`, `closest_point_on_circle_2d`)
from `mmcore.geom.curves` at the start of your script.

That's it! You're now able to perform various 2D geometric calculations involving circles.

