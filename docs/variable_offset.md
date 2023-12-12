# User Guide for `polygon_variable_offset` Function

The `polygon_variable_offset` is a function that calculates the offset of a polygon with variable distances for each
side. This function resides under the `mmcore.geom.parametric.algorithms` module.

Below are the details about its usage:

## Importing the Function

To use the `polygon_variable_offset` function in your script, import it from its module:

```python
from mmcore.geom.parametric.algorithms import polygon_variable_offset
```

## Function Signature

The `polygon_variable_offset` function takes two parameters:

```python
from numpy import ndarray


def polygon_variable_offset(points: ndarray, dists: ndarray): ...
```

### Parameters

- `points`: A 2D numpy array, where each item is an array of three coordinates representing a point on the polygon in 3D
  space. The polygon is defined by these points in the order they are provided.

- `dists`: A 2D numpy array, where each item is a pair of distances representing how far each side of the polygon should
  be offset. The number of items should match the number of points defining the polygon.

Below is the description of parameters:

1. `points`: This is a 2D numpy array of shape `(n, 3)`. Each item in the array represents a point on the polygon
   specified as `[x, y, z]`.

2. `dists`: This is a 2D numpy array of shape `(n, 2)`. Each item in `dists` corresponds to the distance for offsetting
   each side of the polygon. Negative values offset to the inside of the polygon.

## Example Usage

Let's assume we have a polygon defined by 5 points and we want each side to be offset by a specific distance. Here is
how you could use the `polygon_variable_offset` function:

```python
import numpy as np
from mmcore.geom.parametric.algorithms import polygon_variable_offset

# Define the polygon points
pts = np.array([[33.049793, -229.883303, 0],
                [132.290583, -165.409427, 0],
                [48.220282, 27.548631, 0],
                [-115.077733, -43.599024, 0],
                [-44.627307, -205.296759, 0]])

# Define the offset distances for each side
dists = np.zeros((5, 2))  # initially all set to zero
dists[0] = 4  # We want the first side to be offset by 4 units
dists[2] = 1  # We want the third side to be offset by 1 unit
dists[-1] = 2  # We want the last side to be offset by 2 units

# Now we can call the function
*res, = polygon_variable_offset(pts, dists)

# Print the result as a numpy array
print(np.array(res))
```

The results are the points on the offset polygon.

![img.png](/Users/andrewastakhov/PycharmProjects/mmcore/Writerside/images/img_2.png)
## Error Handling

If the `points` parameter is empty, contains non-2D points, or if the `dists` parameter is not of the same length
as `points` or is not a 2D array, the `polygon_variable_offset` function will raise a ValueError with an appropriate
error message.

