<a id="mmcore.numeric.curve_intersection"></a>

# mmcore.numeric.curve_intersection



<!-- TOC -->
* [mmcore.numeric.curve_intersection](#mmcorenumericcurve_intersection)
  * [Overview](#overview)
  * [Curve Intersection Methods](#curve-intersection-methods-)
      * [Method utilization depending on the supported protocol of the first and second curve.](#method-utilization-depending-on-the-supported-protocol-of-the-first-and-second-curve)
      * [Methods comparison](#methods-comparison)
  * [PI (Parametric Implicit Intersection)](#pi-parametric-implicit-intersection)
    * [Example](#example)
      * [Intersection with implicit curve.](#intersection-with-implicit-curve)
      * [Intersection with implicit surface.](#intersection-with-implicit-surface)
  * [PP (Parametric Parametric Intersection)](#pp-parametric-parametric-intersection)
    * [Example](#example-1)
  * [II (Implicit Implicit Intersection)](#ii-implicit-implicit-intersection)
      * [Note](#note)
    * [Example](#example-2)
<!-- TOC -->

## Overview
Methods for finding all intersection points for two given curves
## Curve Intersection Methods 
There are three types of curve intersections based on their representations.

**PP** -  [Parametric x Parametric](#pp-parametric-parametric-intersection)\
**PI** - [Parametric x Implicit](#pi-parametric-implicit-intersection)\
**II** - [Implicit x Implicit](#ii-implicit-implicit-intersection)


#### Method utilization depending on the supported protocol of the first and second curve.

|                           | Parametric Only | Implicit Only | Parametric & Implicit |
|---------------------------|:---------------:|:-------------:|:---------------------:|
| **Parametric Only**       |      `PP`       |     `PI`      |         `PI`          | 
| **Implicit Only**         |      `PI`       |     `II`      |         `PI`          | 
| **Parametric & Implicit** |      `PI`       |     `PI`      |         `PI`          | 
  
#### Methods comparison
 

|                             | Parametric x Parametric (PP) | Parametric x Implicit (PI) | Implicit x Implicit  (II) |
|-----------------------------|:----------------------------:|:--------------------------:|:-------------------------:|
| **Complexity**              |             Hard             |            Easy            |           Easy            |
| **3D support**              |             True             |            True            |           False           | 
| **High dimension support**  |            False             |  True (implicit surface)   |           False           |
| **Multi component support** |            False             |           False            |           True            |
| **Returns Cartesian**       |             True             |            True            |           True            | 
| **Returns Parametric**      |        First & Second        |           First            |           False           | 



<a id="mmcore.numeric.curve_intersection.curve_pii"></a>


PI (Parametric Implicit Intersection)
-----
```python
def curve_pii(curve,
              implicit: Callable[[ArrayLike], float],
              step: float = 0.5,
              default_tol=1e-3) -> list[float]
```


The function finds the parameters where the parametric curve intersects some implicit form. It can be a curve,
surface, ..., anything that has the form f(x)=0 where x is a vector denoting a point in space.

**Arguments**:

- `curve`: the curve in parametric form
- `implicit` (`callable`): the implicit function like `f(x) = 0` where x is a vector denoting a point in space.
- `step`: Step with which the range of the curve parameter is initially split for estimation. defaults to 0.5
- `default_tol` (`float`): The error for computing the intersection in the parametric space of a curve.
If the curve is a spline, the error is calculated automatically and the default_tolerance value is ignored.

**Returns**:

`list[float]
This algorithm is the preferred algorithm for curve crossing problems.
Its solution is simpler and faster than the general case for PP (parametric-parametric) or II (implict-implict)
intersection. You should use this algorithm if you can make one of the forms parametric and the other implicit.

### Example
#### Intersection with implicit curve.

```python
>>> import numpy as np
>>> from mmcore.numeric.curve_intersection import curve_pii
>>> from mmcore.geom.curves import NURBSpline
```

1. Define the implict curve.
    ```python
    >>> def cassini(x, a=1.1, c=1.0):
    ...     return ((x[0] ** 2 + x[1] ** 2) * (x[0] ** 2 + x[1] ** 2)
    ...              - 2 * c * c * (x[0] ** 2 - x[1] ** 2)
    ...              - (a ** 4 - c ** 4))
    ```

 2. Сreate the parametric curve:
   ```python
    >>> spline = NURBSpline(np.array([(-2.3815177882733494, 2.2254910228438045, 0.0),
    ...                               (-1.1536662710614194, 1.3922103249454953, 0.0),
    ...                               (-1.2404122859674858, 0.37403957301406443, 0.0),
    ...                               (-0.82957856158065857, -0.24797333823516698, 0.0),
    ...                               (-0.059146886557566614, -0.78757517340047745, 0.0),
    ...                               (1.4312784414267623, -1.2933712167625511, 0.0),
    ...                               (1.023775628607696, -0.58571247602345811, 0.0),
    ...                               (-0.40751426943615976, 0.43200382009529514, 0.0),
    ...                               (-0.091810780095197053, 2.0713419737806906, 0.0)]
    ...                           ),
    ...                           degree=3)
    
```

3. Calculate the parameters in which the parametric curve intersects the implicit curve.
    ```python
    >>> t = curve_pii(spline,cassini)
    >>> t
    [0.9994211794774993, 2.5383824909241675, 4.858054223961756, 5.551602752306095]
    ```
4. Now you can evaluate the points by passing a list of parameters to a given parametric curve.
   ```python
    >>> spline(t)
   array([[-1.15033465,  0.52553561,  0.        ],
      [-0.39017002, -0.53625519,  0.        ],
      [ 0.89756699, -0.59935844,  0.        ],
      [-0.01352083,  0.45838774,  0.        ]])
   ```

#### Intersection with implicit surface.

1. Define the implicit surface.
    ```python
    >>> def genus2(x):
    ...     return 2 * x[1] * (x[1] ** 2 - 3 * x[0] ** 2) * (1 - x[2] ** 2) + (x[0] ** 2 + x[1] ** 2) ** 2 - (
    ...         9 * x[2] ** 2 - 1) * (1 - x[2] ** 2)
    ```
2. Repeat the items from the previous example
    ```python
    >>> t = curve_pii(spline, genus2)
    >>> t
    [0.6522415161474464, 1.090339012572083]

    >>> spline(t)
    array([[-1.22360866,  0.96065424,  0.        ],
       [-1.13538864,  0.43142482,  0.        ]]) # list of parameters that implicit function is intersects
    ```



<a id="mmcore.numeric.curve_intersection.curve_ppi"></a>

PP (Parametric Parametric Intersection)
-----
```python
def curve_ppi(curve1,
              curve2,
              tol: float = 0.001,
              tol_bbox=0.1,
              bounds1=None,
              bounds2=None,
              eager=True) -> list[tuple[float, float]]
```

Intersection for the two parametric curves.
curve1 and curve2 can be any object with a parametric curve interface.
However, in practice it is worth using only if both curves do not have implicit representation,
most likely they are two B-splines or something similar.
Otherwise it is much more efficient to use PII (Parametric Implict Intersection).

The function uses a recursive divide-and-conquer approach to find intersections between two curves.
It checks the AABB overlap of the curves and recursively splits them until the distance between the curves is within
the specified tolerance or there is no overlap. The function returns a list of tuples
representing the parameter values of the intersections on each curve.

Обратите внимание! Этот метод продолжает "Разделяй и властвуй" пока расстояние не станет меньше погрешности.
Вы можете значительно ускорить поиск, начиная метод ньютона с того момента где для вас это приемлимо.
Однако имейте ввиду что для правильной сходимости вы уже должны быть в "низине" с одним единственым минимумом.

**Arguments**:

- `curve1`: first curve
- `curve2`: second curve
- `bounds1`: [Optional] custom bounds for first NURBS curve. By default, the first NURBS curve interval.
- `bounds2`: [Optional] custom bounds for first NURBS curve. By default, the second NURBS curve interval.
- `tol`: A pair of points on a pair of Euclidean curves whose Euclidean distance between them is less than tol will be considered an intersection point

**Returns**:

`list[tuple[float, float]] | list`

### Example

```python
>>> first = NURBSpline(
...    np.array(
...        [
...            (-13.654958030023677, -19.907874497194975, 0.0),
...            (3.7576433265207765, -39.948793039632903, 0.0),
...            (16.324284871574083, -18.018771519834026, 0.0),
...            (44.907234268165922, -38.223959886390297, 0.0),
...            (49.260384607302036, -13.419216444520401, 0.0),
...        ]
...    )
... )
>>> second= NURBSpline(
...     np.array(
...         [
...             (40.964758489325661, -3.8915666456564679, 0.0),
...             (-9.5482124270650726, -28.039230791052990, 0.0),
...             (4.1683178868166371, -58.264878428828240, 0.0),
...             (37.268687446662931, -58.100608604709883, 0.0),
...         ]
...     )
... )



>>> intersections = curve_ppi(first, second, 0.001)
>>> print(intersections)
[(0.600738525390625, 0.371673583984375)]`: List containing all intersections, or empty list if there are no intersections. Where intersection
is the tuple of the parameter values of the intersections on each curve.
```

<a id="mmcore.numeric.curve_intersection.curve_iii"></a>

II (Implicit Implicit Intersection)
-----

```python
def curve_iii(curve1,
              curve2,
              tree: ImplicitTree2D = None,
              rtol=None,
              atol=None)
```


Intersection for the two implicit curves. Requirements: both curves must have an `implicit` method
with the following signature: `(xy:np.ndarray[2, dtype[float]]) -> float`

**Arguments**:

- `curve1`: first curve
- `curve2`: second curve
- `tree` (`ImplicitTree2D`): [Optional] The ImplicitTree2D object representing descretizations of primitives. If None, a new treewill be constructed from the union of the curves.
- `rtol`: [Optional] see Note.
- `atol`: [Optional] see Note.

**Returns**:

List containing all intersection points.

#### Note
**atol, rtol :**
These values will be passed to np.allclose at and will affect whether two close intersections are
considered as one. If None, the values set in np.allclose will be used.
In general, curve_iii does not require passing an error parameter, and the solution can be considered "exact".

### Example

```python
>>> import numpy as np
>>> from mmcore.geom.implicit import Implicit2D

>>> class Circle2D(Implicit2D):
...    def __init__(self, origin=(0.0, 0.0), radius=1):
...        super().__init__(autodiff=True)
...        self.origin = np.array(origin, dtype=float)
...        self.radius = radius
...
...    def bounds(self):
...        return self.origin - self.radius, self.origin + self.radius
...
...    def implicit(self, v):
...        return np.linalg.norm(v - self.origin) - self.radius
...

>>> c1,c2=Circle2D((0.,1),2),Circle2D((3.,3),3)
>>> intersections = curve_iii(c1,c2)
>>> intersections
[[1.8461538366698576, 0.2307692265580236], [0.0, 2.999999995]]

>>> np.array(intersections)
array([[1.84615384, 0.23076923],
       [0.        , 3.        ]])

```

