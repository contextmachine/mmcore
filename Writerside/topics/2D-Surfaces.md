# 2D Surfaces

## Module Description

This Python module implements multiple classes that represent 2D surfaces. Each class represents a specific type of 2D
surface and associated operations. The key classes are

1. `Surface2D`: Base class implementing common functionality for all 2D surfaces.
2. `OffsetSurface2D`: Class representing a surface formed by moving away from an existing surface by a specified
   distance.
3. `PlaneSurface2D`: Class representing a planar surface defined by its origin and 'u' & 'v' directions.
4. `PlaneLinePointSurface`: Class representing a planar surface defined by a line and a point.
5. `PlaneLineDirectionSurface`: Class representing a planar surface defined by a line and a direction.
6. `LineLineSurface2D`: Class representing a surface created by sweeping a line along another line.

## Class Examples

A few examples on how to use these classes are provided below.

```python
import numpy as np

# Assuming Line is a class with necessary properties and methods.
line1 = Line(start=[0, 0, 0], end=[1, 0, 0])
line2 = Line(start=[0, 1, 0], end=[1, 1, 0])
point = [0, 0, 0]
direction = np.array([0, 0, 1])

# 1. Surface2D
# No direct usage as it serves as a base class

# 2. OffsetSurface2D
offsetSurface = OffsetSurface2D(distance=2.5, parent=someOtherSurface2Dobject)

# 3. PlaneSurface
planeSurface = PlaneSurface2D()
planeSurface.origin = np.array([0, 0, 0])
planeSurface.xaxis = np.array([1, 0, 0])
planeSurface.yaxis = np.array([0, 1, 0])

# 4. PlaneLinePointSurface
planeLinePointSurface = PlaneLinePointSurface(line1, point)

# 5. PlaneLineDirectionSurface
planeLineDirectionSurface = PlaneLineDirectionSurface(line1, direction)

# 6. LineLineSurface2D 
lineLineSurface = LineLineSurface2D(line1, line2)
```

