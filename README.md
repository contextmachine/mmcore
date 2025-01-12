# mmcore

[![poetry-build](https://github.com/contextmachine/mmcore/actions/workflows/poetry-build.yml/badge.svg)](https://github.com/contextmachine/mmcore/actions/workflows/poetry-build.yml)
[![Docker](https://github.com/contextmachine/mmcore/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/contextmachine/mmcore/actions/workflows/docker-publish.yml) 
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![pip downloads](https://img.shields.io/pypi/dm/mmcore)](https://pypi.python.org/project/mmcore)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mmcore.svg)](https://pypi.python.org/project/mmcore)

![](notes/images/img.png)

<!-- TOC -->
* [mmcore](#mmcore)
  * [Overview](#overview)
  * [Key Features](#key-features)
  * [Installation](#installation)
    * [Using pip (Python 3.9+)](#using-pip-python-39)
    * [PyPy Support](#pypy-support)
    * [Docker](#docker)
  * [Project Structure](#project-structure)
    * [Core Modules](#core-modules)
    * [Additional Components](#additional-components)
  * [Getting Started](#getting-started)
  * [Implementation Examples](#implementation-examples)
    * [1. Parametric Representations](#1-parametric-representations)
    * [2.Geometry Construction](#2geometry-construction)
    * [3. Basic CAD Algorithms](#3-basic-cad-algorithms)
      * [Closest Point Example](#closest-point-example)
    * [4. Advance CAD Algorithms](#4-advance-cad-algorithms)
      * [SSX Example](#ssx-example)
    * [5. Implicit Representation](#5-implicit-representation)
    * [6. CAD Algorithms for Implicits](#6-cad-algorithms-for-implicits)
  * [Dependencies](#dependencies)
    * [Core Requirements](#core-requirements)
    * [Optional Components](#optional-components)
  * [Recent Updates (v0.52.0)](#recent-updates-v0520)
  * [Known Deprecations](#known-deprecations)
  * [Contributing](#contributing)
  * [License](#license)
<!-- TOC -->

## Overview

mmcore is a modern CAD engine written in Python with performance-critical parts implemented in Cython. The main goal is to make advanced CAD capabilities as accessible and easy to use as popular scientific computing libraries.

The library provides a comprehensive set of geometric modeling tools, numerical algorithms, and optimization methods specifically designed for CAD applications. It features efficient implementations of NURBS geometry, surface analysis, intersection algorithms, and more.

**Note:** mmcore is under active development and does not currently guarantee backwards compatibility. The API may change significantly between versions.

## Key Features

- **Geometric Modeling**
  - Complete NURBS curves and surfaces implementation
  - Advanced surface analysis with fundamental forms
  - Comprehensive intersection algorithms
  - Implicit geometry support with boolean operations
  - Primitive shapes and surface analysis tools

- **Numerical Methods**
  - General purpose optimization algorithms (Newton method, divide-and-conquer)
  - Robust numerical integration (RK45 and alternatives)
  - Interval arithmetic support
  - Advanced intersection algorithms for curves and surfaces
  - CAD-specific computational geometry algorithms
  
- **Performance Optimization**
  - Critical algorithms implemented in C/C++/Cython 
  - Fastest NURBS implementation in python.
  - BVH (Bounding Volume Hierarchy) for efficient spatial queries
  - Vectorized operations outperforming numpy for 2D-4D cases

    
## Installation

### Using pip (Python 3.9+)

```bash
python3 -m pip install --user --force-reinstall git+https://github.com/contextmachine/mmcore.git@tiny
```

### PyPy Support
```bash
pypy3 -m pip install --user --force-reinstall git+https://github.com/contextmachine/mmcore.git@tiny
```

### Docker
```bash
docker pull ghcr.io/contextmachine/mmcore.git:tiny
```

## Project Structure

### Core Modules

- **mmcore.geom**: Geometric primitives and operations
  - `nurbs.pyx/pxd`: NURBS curves and surfaces implementation
  - `implicit`: Implicit geometry with boolean operations
  - `bvh`: Spatial acceleration structures

- **mmcore.numeric**: Algorithms and computations
  - `algorithms`: Optimization and fundamental CAD algorithms
  - `integrate`: Numerical integration (RK45 and others)
  - `interval`: Interval arithmetic implementation
  - `intersections`: Comprehensive intersection algorithms (curve x curve, curve x surface, surface x surface)
  - `vectors`: High-performance vector operations
- **mmcore.construction**: Construction operations

  - `ruled`: Implements a fabric function to construct Ruled NURBSSurface from a two NURBSCurve.
  
### Additional Components

- **mmcore.api**: High-level interface for common operations (WIP)
- **mmcore.renderer**: Visualization capabilities  (WIP)
- **mmcore.topo**: Topological operations and mesh handling  (WIP)

## Getting Started
1. Start with short introduction in the [Implementation Examples](#implementation-examples)

2. Check the basic examples in `examples/`:
   - `surface_closest_points.py`: Surface analysis and optimization
   - `primitives/`: Basic geometric shape creation
   - `ssx/`: Surface-surface intersection examples
   - `implicit_intersections.py`: Working with implicit geometry

2. Check the short introduction in the [Implementation Examples](#implementation-examples):
   - [surface_closest_point.md](./notes/surface_closest_point.md): Detailed algorithm explanations
   - Additional implementation examples and best practices

## Implementation Examples
### 1. Parametric Representations

We recommend using NURBS as parametric representations, Although procedural parametric representations are also supported for many operations, due to their properties NURBS representations can be used in algorithms requiring strict robustness.
```python
from mmcore.geom.nurbs import NURBSCurve,NURBSSurface
```

### 2.Geometry Construction
This creates a simple NURBS curve of degree 3 on 10 control points:
```python
import numpy as np
from mmcore.geom.nurbs import NURBSCurve
curve = NURBSCurve(np.random.random((10,3))) # This 
```

```python

from mmcore.geom.nurbs import NURBSCurve
from mmcore.construction.ruled import ruled
# Create forming curves
curve1=NURBSCurve(...)
curve2=NURBSCurve(...)
# Create a ruled surface
surface = ruled(curve1,curve2)

```
### 3. Basic CAD Algorithms

#### Closest Point Example
This example demonstrates such a base operation as closest point on surface
```python
import numpy as np
from mmcore.numeric.closest_point import closest_point_on_surface_batched
# Surface construction
surface=...

# Create a random 3d points
points = np.random.random((100,3))

# Find closest points on surface:
closest_points = closest_point_on_surface_batched(surface, points)

```
You can find a detailed algorithm explanation here [surface_closest_point.md](./notes/surface_closest_point.md)

### 4. Advance CAD Algorithms
Algorithms for finding all intersections are fundamental in CAD.  In mmcore there are robust implementations for parametric NURBS objects:
- **CCX** (Curve Curve Intersection)  
- **CSX** (Curve Surface Intersection)
- **SSX** (Surface Surface Intersection)

Also in experimental mode there are implementations for implicit and procedural objects. However, at the moment it is not guaranteed to find all intersections in the general case (we are working on it).
#### SSX Example

```python
from mmcore.numeric.intersection.ssx import ssx
# Surfaces construction
surface1=...
surface2=...

# Perform Surface and Surface Intersection
result = ssx(surface1,surface2,tol=0.001)
```

You can find full examples at [examples/ssx/nurbs_nurbs_intersection_1.py](examples/ssx/nurbs_nurbs_intersection_1.py) and [examples/ssx/nurbs_nurbs_intersection_2.py](examples/ssx/nurbs_nurbs_intersection_2.py).
To display the output of the algorithm install the viewer plugin with `pip install "mmcore[renderer]"` or 
`pip install "mmcore[all]"`.


### 5. Implicit Representation
Implicit representations are less common in commercial frame systems, but like parametric representations have long been well developed in computer graphics. General implicits are implemented in mmcore. It means that the algorithms are suitable for working with any implicits and not only with such widespread implicit forms as SDF. 

To create your own implicit object class, all you need to do is inherit from one of the base classes and override the `implicit(self,point)` and `bounds(self)` methods:
```python
from mmcore.geom.implicit import Implicit2D

class Circle(Implicit2D):
    def __init__(self, center, radius):
        super().__init__()
        self.center = center
        self.radius = radius
    def bounds(self):
        return self.center-self.radius,self.center+self.radius # min point , max point
    def implicit(self, xy:'ndarray[float, (2,)] | ndarray[float, (3,)]'):
        x, y = xy
        return (x - self.center[0])**2 + (y - self.center[1])**2 - self.radius**2 # circle implicit equation
```
When these methods are implemented, all other methods will be generated automatically. 

An example of applying the standard algorithm for finding the intersection curve between two implicits to an implicit cylinder and a custom implicit based on a point cloud:

![](notes/images/cloud-intersection41.gif)

### 6. CAD Algorithms for Implicits
Closest point, intersections, and others algorithms are also available for implicits in both 2d and 3d cases. Intersection algorithms in 3D are particularly interesting because they show good accuracy and performance, in some cases surpassing commercial packages. 

**Performance Benchmarks:**

Comparison with Rhino 8 for intersection curves computation between solid tubes:

| Task Size | CPU Cores | Rhino 8 (sec.) | mmcore (sec.) | Speed Ratio |
|-----------|-----------|----------------|---------------|-------------|
| 1         | 1         | 0.027         | 0.033         | 0.82x       |
| 100       | 1         | 2.685         | 1.571         | 1.71x       |
| 100       | 10        | 0.938         | 0.275         | 3.41x       |
| 1000      | 10        | 13.4          | 2.313         | 5.79x       |

*In mmcore we use implicit representations, in Rhino tubes are represented by BReps.*

Results show mmcore excels particularly in parallel processing and batch operations.

<img src="notes/images/implicit_tubes_intersection.png" width="300"/>

*Figure: Visualization of tube intersection test case*

At the same time, I would like to remind you that these algorithms are currently not guaranteed in the general case and are under active development.

You can find the full code for this example here [examples/ssx/implicit_intersections.py](examples/ssx/implicit_intersections.py)

## Dependencies

### Core Requirements
- Python >= 3.9
- numpy
- scipy
- earcut
- pyquaternion
- more-itertools

### Optional Components
- Development: Cython
- Visualization: plotly, kaleido, pyopengl, pyrr, glfw
- Interactive: IPython


## Recent Updates (v0.52.0)

1. Enhanced ray and geometry intersection capabilities
   - Added efficient ray-plane intersection algorithms
   - Implemented Möller–Trumbore triangle-segment intersection
   - Enhanced AABB utilities with ray/segment intersections
   
2. Improved BVH implementation and spatial queries
   - Optimized bounding volume hierarchy operations
   - Enhanced spatial acceleration structures
   - Improved query performance for ray-casting and intersections

3. Added new geometric construction utilities
   - Implemented ruled surface construction between NURBS curves
   - Added Union-Find data structure for topology operations
   - Enhanced NURBS knot manipulation and degree elevation

4. Performance optimizations
   - Streamlined Cython configurations for numeric utilities
   - Improved intersection algorithm efficiency
   - Enhanced NURBS operations performance

## Known Deprecations

1. Use `mmcore.numeric.vectors` instead of `mmcore.geom.vec` for vector operations
2. Prefer `NURBSCurve` over `NURBSSpline` for better algorithms and serialization
3. For curve-surface intersection, use `mmcore/numeric/intersections/csx/_ncsx.py`
4. Surface-surface intersection (SSX) implementation is currently reliable only for NURBS surfaces

## Contributing

Contributions are welcome! Please note:

1. The project is under active development
2. Breaking changes may occur between versions
3. Test all changes thoroughly before submitting
4. Follow the existing code style and documentation patterns

## License

Licensed under the Apache License, Version 2.0 - see [LICENSE](LICENSE) for details.