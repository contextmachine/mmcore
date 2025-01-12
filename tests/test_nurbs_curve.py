from mmcore.geom.nurbs import NURBSCurve
import numpy as np

# Create a simple NURBS curve for testing
control_points = np.array([
    [0.0, 0.0, 0.0],  # First control point
    [1.0, 1.0, 0.0],  # Second control point
    [2.0, 0.0, 0.0],  # Third control point
    [3.0, 1.0, 0.0]   # Fourth control point
])

# Create the curve
curve = NURBSCurve(control_points, degree=3)

# Get the parameter range
tmin, tmax = curve.interval()
print(f"Parameter range: [{tmin}, {tmax}]")

# Test points:
test_points = [
    tmin - 0.1,  # Before start
    tmin,        # Start
    0.333,       # Inside range
    0.666,       # Inside range
    tmax,        # End
    tmax + 0.1   # After end
]

print("\nEvaluation results:")
print("-" * 50)
for t in test_points:
    point = curve.evaluate(t)
    print(f"t = {t:6.3f}: {point}")