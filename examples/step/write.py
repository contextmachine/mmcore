"""This example demonstrates how to use the `NURBSSurface` class from the `mmcore` package to create NURBS (Non-Uniform
Rational B-Splines) surfaces and how to write them to a STEP file using `StepWriter`."""
import numpy as np
from mmcore.geom.nurbs import NURBSSurface
from mmcore.compat.step.step_writer import StepWriter

# 1. **Defining Control Points**:
#     - `pts1`: Control points for the first NURBS surface. Organized into a 2-dimensional array.
#     - `pts2`: Control points for the second NURBS surface. Similarly organized.

pts1 = np.array(
    [
        [-25.0, -25.0, -10.0],
        [-25.0, -15.0, -5.0],
        [-25.0, -5.0, 0.0],
        [-25.0, 5.0, 0.0],
        [-25.0, 15.0, -5.0],
        [-25.0, 25.0, -10.0],
        [-15.0, -25.0, -8.0],
        [-15.0, -15.0, -4.0],
        [-15.0, -5.0, -4.0],
        [-15.0, 5.0, -4.0],
        [-15.0, 15.0, -4.0],
        [-15.0, 25.0, -8.0],
        [-5.0, -25.0, -5.0],
        [-5.0, -15.0, -3.0],
        [-5.0, -5.0, -8.0],
        [-5.0, 5.0, -8.0],
        [-5.0, 15.0, -3.0],
        [-5.0, 25.0, -5.0],
        [5.0, -25.0, -3.0],
        [5.0, -15.0, -2.0],
        [5.0, -5.0, -8.0],
        [5.0, 5.0, -8.0],
        [5.0, 15.0, -2.0],
        [5.0, 25.0, -3.0],
        [15.0, -25.0, -8.0],
        [15.0, -15.0, -4.0],
        [15.0, -5.0, -4.0],
        [15.0, 5.0, -4.0],
        [15.0, 15.0, -4.0],
        [15.0, 25.0, -8.0],
        [25.0, -25.0, -10.0],
        [25.0, -15.0, -5.0],
        [25.0, -5.0, 2.0],
        [25.0, 5.0, 2.0],
        [25.0, 15.0, -5.0],
        [25.0, 25.0, -10.0],
    ]
)
pts1 = pts1.reshape((6, len(pts1) // 6, 3))
pts2 = np.array(
    [
        [25.0, 14.774795467423544, 5.5476189978794661],
        [25.0, 10.618169208735296, -15.132510312735601],
        [25.0, 1.8288992061686002, -13.545426491756078],
        [25.0, 9.8715747661086723, 14.261864686419623],
        [25.0, -15.0, 5.0],
        [25.0, -25.0, 5.0],
        [15.0, 25.0, 1.8481369394623908],
        [15.0, 15.0, 5.0],
        [15.0, 5.0, -1.4589623860307768],
        [15.0, -5.0, -1.9177595746260625],
        [15.0, -15.0, -30.948650572598954],
        [15.0, -25.0, 5.0],
        [5.0, 25.0, 5.0],
        [5.0, 15.0, -29.589097491066767],
        [3.8028908181980938, 5.0, 5.0],
        [5.0, -5.0, 5.0],
        [5.0, -15.0, 5.0],
        [5.0, -25.0, 5.0],
        [-5.0, 25.0, 5.0],
        [-5.0, 15.0, 5.0],
        [-5.0, 5.0, 5.0],
        [-5.0, -5.0, -27.394523521151221],
        [-5.0, -15.0, 5.0],
        [-5.0, -25.0, 5.0],
        [-15.0, 25.0, 5.0],
        [-15.0, 15.0, -23.968082282285287],
        [-15.0, 5.0, 5.0],
        [-15.0, -5.0, 5.0],
        [-15.0, -15.0, -18.334465891060319],
        [-15.0, -25.0, 5.0],
        [-25.0, 25.0, 5.0],
        [-25.0, 15.0, 14.302789083068138],
        [-25.0, 5.0, 5.0],
        [-25.0, -5.0, 5.0],
        [-25.0, -15.0, 5.0],
        [-25.0, -25.0, 5.0],
    ]
)

pts2 = pts2.reshape((6, len(pts2) // 6, 3))

# 2. **Creating NURBS Surfaces**:
#     - Create two `NURBSSurface` objects using the defined control points.
surf1 = NURBSSurface(pts1, (3, 3))
surf2 = NURBSSurface(pts2, (3, 3))

# 3. **Writing to STEP File**:
#     - Initialize a `StepWriter`.
#     - Add both NURBS surfaces to the `StepWriter`.
#     - Write the surfaces to a STEP file named `example.step`.
step_writer = StepWriter()
step_writer.add_nurbs_surface(surf1, color=(0.5,0.5,0.5),name='surf1')
step_writer.add_nurbs_surface(surf2, color=(0.1,0.5,0.5), name='surf2')

with open('example.step', 'w') as f:
    step_writer.write(f)
