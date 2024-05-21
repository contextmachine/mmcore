# mmcore

Some useful hints:
- `mmcore.api` - User API like common CAD systems.

- `mmcore.geom` - Data structures representing geometric objects
  - `mmcore.geom.curves` - Representations of curves in parametric form
  - `mmcore.geom.curves.curve` - Basic curve class in parametric form that implements many useful methods
  
  - `mmcore.geom.curves.bspline` - **B-Spline** curves and **NURBS** curves
  - `mmcore.geom.surfaces` - Representations of surfaces in parametric form
  - `mmcore.geom.implicit` - Implicit representations [[1]](https://en.wikipedia.org/wiki/Implicit_curve)  of curves and surfaces  
  - `mmcore.geom.implivit.dc` - **Adaptive Dual Contouring** algorithm [[2]](https://www.cs.wustl.edu/~taoju/research/interfree_paper_final.pdf) , so far in 2D. 
  - `mmcore.geom.implivit.tree` - Octree approximation for shapes in implicit form (like libfive [[3]](https://github.com/libfive/libfive/blob/master/libfive/src/tree/tree.cpp)), so far in 2D.  
  - `mmcore.geom.bvh` - Implementation of BVH tree construction and queries. This can be extremely useful for intersectionb nearest neighbor search and other spatial tasks. 
  
- `mmcore.numeric` - Implementation of fundamental numeric CAD algorithms and some data structures. 
  - `mmcore.numeric.fdm` - Automatic differentiation using **Finite Difference Method**.
  - `mmcore.numeric.plane` - Useful procedures for working with planes and coordinate systems
  - `mmcore.numeric.closest_point` - Currently includes implementation of the algorithm for finding the **Closest Point on Curve**, **Parametric Surface Foot Point Algorithm** [[4]](Geometry and Algorithms for COMPUTER AIDED DESIGN, p. 94), etc.
  - `mmcore.numeric.curve_intersection` - Algorithm for finding all intersection points between two curves. All cases (Implicit-Implicit,Parametric-Implicit,Parametric-Parametric) are provided.


## References
1. https://en.wikipedia.org/wiki/Implicit_curve
2. https://en.wikipedia.org/wiki/Implicit_curve
3. https://github.com/libfive/libfive/blob/master/libfive/src/tree/tree.cpp
4. Geometry and Algorithms for COMPUTER AIDED DESIGN, p. 94
