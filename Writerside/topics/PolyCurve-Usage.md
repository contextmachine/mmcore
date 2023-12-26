# How to use the PolyCurve class

In this tutorial, we will mainly focus on how to use the `PolyCurve` class from the `mmcore.geom.polycurve` module.

### Step 1: Creating a PolyCurve

To create a `PolyCurve` object, first import the `PolyCurve` class and numpy:

```python
from mmcore.geom.polycurve import PolyCurve
import numpy as np
```

#### Create the `PolyCurve`

> By default, the constructor accepts a set of segments, in case you want to create an object from complex curves.
> If you just want to create a polyline, look at the `PolyCurve.from_points` method (next example).

Import classes for segments objects:

```python
from mmcore.geom.line import Line
```

Create polycurve segments

```python
a, b, c, d = np.array([[(-293.90192934239769, 552.79553618743603, 0.0), (-38.355428662402574, 559.56153315045117, 0.0)],
                       [(-101.33124654891867, 382.60468950238948, 0.0), (-98.728940024682473, 632.94657713391121, 0.0)],
                       [(-237.64193845280525, 484.77128576271491, 0.0), (-38.355428662402574, 465.35803697309967, 0.0)],
                       [(-292.34054542785589, 382.60468950238948, 0.0), (-115.90416308464137, 643.98494013701122, 0.0)]]
                      )
l1 = Line.from_ends(*a)
l2 = Line.from_ends(*b)
l3 = Line.from_ends(*c)
l4 = Line.from_ends(*d)

```

```python
pcurve = PolyCurve([l1, l2, l3, l4])
print(pcurve)
``` 

Output will like:

```
PolyCurve[LineNode](length=4) at 0x170061390
```

Corners will estimate by segment intersections

```python
print(pcurve.corners)
```

Output will like:

```
array([[-175.33969084,  555.93465859,    0.        ],
       [ -99.50860916,  557.94240617,    0.        ],
       [-100.40818896,  471.40282984,    0.        ],
       [-224.25642889,  483.46735291,    0.        ]])
```

#### Create the `PolyCurve` from points

Then provide points for the `PolyCurve` in a numpy array:

```python
points = np.array([(-22.047791358681653, -0.8324885498102903, 0.0), (-9.3805108456226147, -28.660718210796471, 0.0),
                   (22.846252925543098, -27.408177802862003, 0.0), (15.166676249569946, 2.5182225098112045, 0.0)]
                  )

```

```python
polycurve = PolyCurve.from_points(points)

```

### Step 2: Inserting a single corner

To insert a corner into the `PolyCurve`, use `insert_corner`. Let's add a new corner point:

```python
corner_point = np.array((24.707457249539218, -8.0614698399460814, 0.0))
polycurve.insert_corner(corner_point)
```

Now, when you print the `corners`, you will see the inserted corner:

```python
print(polycurve.corners)
```

### Step 3: Inserting multiple corners

We can also insert multiple corners into our `PolyCurve`. For that, we use `insert_corner` inside a `for` loop:

```python
for corner in [(-8.7294742050094118, 9.9843044974317401, 0.0), (-31.500187542229803, -6.5544241369704661, 0.0),
               (-21.792672908993747, -25.849607543773036, 0.0), (33.695960118022263, -10.149799927057904, 0.0),
               (19.793840396350852, -36.875426633374516, 0.0), (7.3298709907144382, 9.6247669184230027, 0.0),
               (26.625054397516983, -0.44228529382181825, 0.0), (-34.376488174299752, -18.778701823267753, 0.0),
               (0.85819456855706555, 14.418601305206250, 0.0)]:
    polycurve.insert_corner(corner)
```

### Step 4: Evaluate PolyCurve

To evaluate the `PolyCurve` at certain points, use the `PolyCurve` instance as a callable and pass in an array of
evaluation points:

```python
evaluated_points = polycurve(np.linspace(0, 4, 10))
```

### Step 5: Evaluate Node on PolyCurve

You can also use the `evaluate_node` method to evaluate the `PolyCurve` at certain nodes:

```python
points_on_curve = polycurve.evaluate_node(np.linspace(0, 4, 10))
```

### Step 6: Changing a Corner Point

You can change the position of a corner point using the `set_corner` method:

```python
polycurve.set_corner(0, [-15., 0., 0.])
```

In the above code, we are changing the first corner point to `[-15., 0., 0.]`.

Now, when you print you points, you will see the changes:

```python
print(np.array(points))
```

And that concludes this brief tutorial on how to use the `PolyCurve` class.

