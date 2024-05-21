# mmcore

Some useful hints:
- `mmcore.api` - User API like common CAD systems.

- `mmcore.geom` - Data structures representing geometric objects
  - `mmcore.geom.curves` - Representations of curves in parametric form
  - `mmcore.geom.curves.curve` - Basic curve class in parametric form that implements many useful methods
  - `mmcore.geom.curves.bspline` - B-Spline curves and NURBS curves
  - `mmcore.geom.surfaces` - Representations of surfaces in parametric form
  - `mmcore.geom.implicit` - Implicit representations of curves and surfaces  
  - `mmcore.geom.bvh` - Implementation of BVH tree construction and queries . This can be extremely useful for intersectionb nearest neighbor search and other spatial tasks. 

- `mmcore.numeric` - Implementation of fundamental numeric CAD algorithms and some data structures. 
  - `mmcore.numeric.fdm` - Automatic differentiation using Finite Difference Method.
  - `mmcore.numeric.plane` - Useful procedures for working with planes and coordinate systems
  - `mmcore.numeric.closest_point` - Currently includes implementation of the algorithm for finding the closest point on a parametric curve.
  - `mmcore.numeric.curve_intersection` - Algorithm for finding all intersection points between two curves. All cases (Implicit-Implicit,Parametric-Implicit,Parametric-Parametric) are provided.


