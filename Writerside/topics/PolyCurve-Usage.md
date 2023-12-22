# How to use the PolyCurve class

In this tutorial, we will mainly focus on how to use the `PolyCurve` class from the `mmcore.geom.polycurve` module.

### Step 1: Creating a PolyCurve

To create a `PolyCurve` object, first import the `PolyCurve` class and numpy:

```python
from mmcore.geom.polycurve import PolyCurve
import numpy as np
```

Then provide points for the `PolyCurve` in a numpy array:

```python
points = np.array([(-22.047791358681653, -0.8324885498102903, 0.0), (-9.3805108456226147, -28.660718210796471, 0.0),
    (22.846252925543098, -27.408177802862003, 0.0), (15.166676249569946, 2.5182225098112045, 0.0)]
        )
```

Now create the `PolyCurve`:

```python
polycurve = PolyCurve(points)
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

