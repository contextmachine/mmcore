# Polygon Operations using mmcore.geom.polygon

This guide walks you through how to use the mmcore.geom.polygon library in Python to perform operations on polygons.
Here are the three main operations: Union, Intersection, and Difference.

## Prerequisites

The guide assumes that you're using Python 3.11.6 and have the required Python packages installed.

## Step 1. Import Required Modules

Begin by importing the necessary modules. Here we need `Polygon` and `PolygonCollection` from the `mmcore.geom.polygon`
module, and numpy for numerical operations.

```python
from mmcore.geom.polygon import Polygon, PolygonCollection
import numpy as np
```

## Step 2. Create Polygon Instances

After importing the required classes, we can create polygon instances. Each polygon in this case is created from a list
of points.

```python
points = [[[x1, y1, z1], [x2, y2, z2], ..., [xn, yn, zn]], ..., [points for polygon n]]
polygons = [Polygon(np.array(p)) for p in
            points]  # Convert each list of points to a numpy array and create a Python list of Polygon objects
```

<img alt="Right 3d Plane" height="400"  src="clipping1.svg"/>

Here, `points` is a list of lists, where each sublist represents a separate polygon and consists of a series
of [x, y, z] coordinates defining the points of the polygon.

## Step 3. Inspect Polygon Properties

The `Polygon` class has attributes that let you see various properties of the polygon such as the corners. Here's how to
get the no. of corners in the first polygon:

```python
>> > len(polygons[0].corners)
11 >> > polygons[0].angles
array([2.82562238, 3.14159265, 1.57079633, 1.57079633, 1.57079633, 3.14159265, 1.57079633, 1.57079633, 1.57079633,
       1.57079633, 1.25482605]
      ) >> > np.degrees(polygons[0].angles)
array([161.89623697, 180., 90., 90., 90., 180., 90., 90., 90., 90., 71.89623697])

>> > polygons[0].area
1.4920610000000067
``` 

## Step 4. Refine Polygons

<img alt="Right 3d Plane" height="400"  src="clipping6.svg"/>

You can refine the corners of a polygon using the `refine()` method. This method redefines the corners of the polygon
minimizing the number of corners while maintaining the same shape:

```python
polygons[0].refine()
```

<img alt="Right 3d Plane" height="400"  src="clipping7.svg"/>

## Step 5. Perform Polygon Operations

### Union

The union of two or more polygons includes all the points in the involved polygons. The `+` operator is overloaded to
perform the union operation on polygons:

```python
figure1 = polygons[1] + polygons[2]  
```

<img alt="Right 3d Plane" height="400"  src="clipping2.svg"/>

### Intersection

The intersection includes only the points common to all the involved polygons. In mmcore, the `|` operator performs the
intersection:

```python
figure2 = figure1[0] | polygons[0]  
```

<img alt="Right 3d Plane" height="400"  src="clipping3.svg"/>

### Difference

The `-` operator computes the difference of two polygons, giving you the part of the first polygon that doesn't exist in
the second:

Here, figure4 consists of points that are in the first polygon but not in figure1 (the previously computed union).

```python
figure4 = polygons[0] - figure1[
    0]  # Here, figure4 consists of points that are in the first polygon but not in figure1 (the previously computed union).
```

<img height="400" src="clipping4.svg"/> 

Here, figure5 consists of points that are in figure1 (the previously computed union) but not in the first polygon.

```python
figure5 = figure1[0] - polygons[
    0]  # Here, figure5 consists of points that are in figure1 (the previously computed union) but not in the first polygon.
```

<img src="clipping5.svg" height="400" /> 

## Step 6. Print Resulting

You can print out the result using the print function. We use `*` operator to unpack the polygons and use `'\n'` as
separator for readability:

```python
print(*figure1, sep='\n', end='\n\n')  # prints each Polygon in figure1 on separate lines.
```

By running these operations, you can manipulate and analyze polygons in Python programmatically using
the `mmcore.geom.polygon` module.

