# Refine

## Overview

The PlaneRefine class is a Python class used for performing operations involving vector computation on 3D coordinates.
The transformations performed by this class consist of unitization of a vector and calculation of the cross product of
the vectors. These operations are carried out according to the order of axes specified in the class constructor.

## Class Definition

```python

class PlaneRefine:
    """
    The `PlaneRefine` class is used to perform vector operations on 3D coordinates.

    Attributes:
        axis (str): The axis along which operations are performed.
        _axis_indices (dict): A dictionary mapping each axis ('x', 'y', 'z') to a number (0, 1, 2)
        next (PlaneRefine): A recursive instance of PlaneRefine for the subsequent axis if provided.

    Methods:
        __init__(*axis: Tuple[str], axis_indices: Dict[str, int] = None)
            Constructor for PlaneRefine class. Raises a ValueError for more than 3 axes.

        __call__(xyz: np.ndarray, inplace: bool = False)
           Perform unitization of the array and the computation of the cross product depending on the axis.
           Can modify the input array in-place and applies recursion if a next PlaneRefine instance exists.
    """

```

{
src="../mmcore/geom/plane/refine.py" validate="true"   include-symbol="PlaneRefine" collapsible="true" href="
../mmcore/geom/plane/refine.py"
}

## How the Class Works

The PlaneRefine class takes in a sequence of axes ('x', 'y', 'z') during its instantiation. These axes represent the
order in which the operations (unitization and cross product computation) are performed in the 3-dimensional numpy array
the instance takes as input when called.
When PlaneRefine object is invoked with a 3x3 numpy array, it performs calculations on this array. First, it transforms
every vector in the numpy array, scaling them to a unit vector (a vector of length 1). Next, it replaces the vector
along the designated axis with a vector that is orthogonal to the plane formed by the vectors on the remaining two axes.
This is done using the cross product operation. If more axes were specified during the initialization of the PlaneRefine
instance, it recursively executes the same operations for the next axis in the sequence.

## Usage

Here's a step-by-step guide on how to use it.

### Step 1: Import necessary modules

Import the `PlaneRefine` class from its module, along with `numpy` to create ndarrays.

```python
import numpy as np
from mmcore.geom.plane.refine import PlaneRefine  # Replace 'module' with the actual module name
```

### Step 2: Instantiate the class

Create an instance of the `PlaneRefine` class. The constructor accepts one or more axes ('x', 'y', 'z') indicating the
order of operations.

```python
refine_zy = PlaneRefine('z', 'y')  # Operations will be performed along 'z' and 'y' axes in sequence
```

Here, the operations will first be performed on the 'z' axis, followed by the 'y' axis.

### Step 3: Prepare your 3D coordinates data

You need to have your 3-dimensional coordinate data ready in a 3x3 numpy ndarray.

```python
pln1 = np.random.random((3, 3))  # Generate a random 3x3 numpy array for this example
```

You can replace this with your actual 3D coordinates data.

### Step 4: Apply PlaneRefining operations

Invoke the created `PlaneRefine` object with your coordinates ndarray. This will apply the sequence of operations on the
ndarray.

```python
pln1_refined = refine_zy(pln1)  # Apply operations on pln1
```

The result, `pln1_refined`, is the transformed version of `pln1`, with operations of unitization and orthogonal vectors
applied in order of 'z' and 'y' axis.

<img alt="Right 3d Plane" height="400"  src="Screenshot 2024-01-07at010208.png"/>

The vector x remains **unchanged**. It coincides completely with the incoming value.

```python

print(np.allclose(pln1_refined[0], pln1[0]))

```

Output:

```
True
```

### Step 5: Modify original ndarray (optional)

If you want the operations to be applied on the original ndarray itself, pass the argument `inplace=True` while invoking
the `PlaneRefine` object.

```python
refine_zy(pln1, inplace=True)  # Apply operations directly on pln1


```

## Considerations:

- The sequence of axes passed during instantiation matters. `PlaneRefine('x', 'y')` will give different results
  from `PlaneRefine('y', 'x')`.

- The operations performed by `PlaneRefine` — vector unitization and cross product — are linear algebra operations
  applicable on 3D

