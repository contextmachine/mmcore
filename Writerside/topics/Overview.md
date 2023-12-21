# Overview

### PDE

The `PDE` class encapsulates a single-variable Partial Differential Equation. It provides an interface for defining the
equation and calculating the derivative using numerical methods. The class supports different numerical methods wrapped
in the `PDEMethodEnum` enumeration.

```python
pde = PDE(func, method=PDEMethodEnum.central, h=0.001)
```

**Parameters:**

- `func`: A function to represent the Partial Differential Equation. This function should take a single argument (
  representing time) and return a single value or a vector of values.
- `method`: The numerical method to use for derivative calculation. Defaults to `PDEMethodEnum.central`.
- `h`: Step size for the numerical method. Defaults to 0.001.

**Methods:**

- `call(t)`: This class can be called with a single argument `t` (representing time) to evaluate the PDE at that point
  in time.
- `tan(t)`: Returns the tangent vector to the PDE at a given time value `t`.
- `normal(t)`: Returns the normal vector to the PDE at a given time value `t`.
- `plane(t)`: Returns a plane defined by the PDE at a given time value `t`.

### Offset

The `Offset` class extends `PDE` to define an offset curve for a given PDE. It can be used to calculate an offset at any
point along the curve.

```python
offset = Offset(func, distance, evaluate_range=(0, 1))
```

**Parameters:**

- `func`: A function to represent the derivative.
- `distance`: The offset distance, can either be a scalar or a function.
- `evaluate_range`: Two-element tuple specifying the parameter range over which to calculate. Defaults to (0, 1).

**Methods:**

- `call(t)`: This class can be called with a single argument `t` (representing time) to evaluate the offset at that
  point in time.

### PDE2D

The `PDE2D` class represents a PDE function in two variables. It is used to define and solve PDEs numerically in higher
dimensions.

```python
pde2d = PDE2D(func, method=PDE2DMethodEnum.methods, h=0.001)
```

**Parameters:**

- `func`: A function to represent the PDE. This function should take two arguments (representing two-dimensional space)
  and return a single value or a vector of values.
- `method`: This numerical method to use for derivative calculation. Defaults to `PDE2DMethodEnum.methods`.
- `h`: Step-size for the numerical method. Defaults to 0.001.

**Methods**

- `call(u, v)`: This class can be called with two arguments `u` and `v` to evaluate the PDE at that point in the
  two-dimensional space.
- `normal(u, v)`: Returns the normal vector to the PDE at a given point in the two-dimensional space.
- `plane(u, v)`: Returns a plane defined by the PDE at a given point in the two-dimensional space.

### Example Usage

```python
# Define the mathematical function
def func(t):
    return t ** 2


# Create a PDE object
pde = PDE(func)

# Evaluate the PDE
result = pde(1.0)

# Compute the offset
offset = Offset(func, 1.0)
offset_result = offset(1.0)
```