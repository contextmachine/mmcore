from __future__ import annotations

import numpy as np

from mmcore.geom.nurbs import NURBSCurve, split_curve_multiple

from mmcore.numeric.algorithms.quicksort import unique

from mmcore.numeric.vectors import scalar_norm
from mmcore.numeric.intersection.ccx import ccx
from mmcore.numeric.vectors import scalar_cross as cross
from typing import Optional, Tuple, List, Sequence

from enum import Enum


class BooleanOperationType(int, Enum):
    UNION = 0
    INTERSECTION = 1
    DIFFERENCE = 2


def unique_with_tolerance(arr, tolerance):
    # Sort the array
    sorted_arr = np.sort(arr)
    # Initialize the list to hold unique values
    unique_values = []
    # Start the first group
    current_group = [sorted_arr[0]]

    for value in sorted_arr[1:]:
        if abs(value - current_group[-1]) <= tolerance:
            # If the value is within the tolerance, add it to the current group
            current_group.append(value)
        else:
            # If the value is outside the tolerance, finalize the current group
            unique_values.append(np.mean(current_group))  # or np.median(current_group)
            # Start a new group
            current_group = [value]

    # Don't forget to add the last group
    unique_values.append(np.mean(current_group))  # or np.median(current_group)

    return np.array(unique_values)


def curve_boolean_operation(curve1: NURBSCurve, curve2: NURBSCurve,
                            operation: BooleanOperationType = BooleanOperationType.UNION) -> List[NURBSCurve]:
    """
    Performs a boolean operation between two closed NURBS curves and returns the result segments.

    Args:
        curve1 (NURBSCurve): The first closed NURBS curve.
        curve2 (NURBSCurve): The second closed NURBS curve.
        operation (BooleanOperationType, optional): The type of boolean operation to perform. Defaults to BooleanOperationType.UNION.

    Returns:
        List[NURBSCurve]: A list of NURBS curves resulting from the boolean operation.

    Raises:
        ValueError: If an unknown operation type is provided.

    Usage Example:
        curve1 = NURBSCurve([...])
        curve2 = NURBSCurve([...])
        result = curve_boolean_operation(curve1, curve2, BooleanOperationType.INTERSECTION)
    """
    # Verify curves are closed?

    # Find all curve intersections
    first, second = zip(*ccx(curve1, curve2))

    # Split both curves at intersection points
    segments1 = split_curve_multiple(curve1, np.unique(first))
    segments2 = split_curve_multiple(curve2, np.unique(second))


    segments = []
    min_2, max_2 = np.array(curve2.bbox())
    min_1, max_1 = np.array(curve1.bbox())


    for seg in segments1:
        t0, t1 = seg.interval()  # Use start of segment

        mid = np.array(seg.evaluate((t0 + t1) / 2))
        ray = NURBSCurve(np.array([mid, mid + np.array([1, 0, 0.]) * scalar_norm(max_2 - min_2) * 2]), degree=1)
        rtct = list(zip(*ccx(ray, curve2, tol=1e-7)))
        if len(rtct) == 0:
            rt = ()
        else:
            rt, ct = rtct
            rt = unique_with_tolerance(rt, 1e-5)
        a_inside_b = len(rt) % 2
        if not a_inside_b:
            if operation == BooleanOperationType.UNION:
                segments.append(seg)
            elif operation == BooleanOperationType.INTERSECTION:
                pass
            elif operation == BooleanOperationType.DIFFERENCE:
                segments.append(seg)
            else:
                raise ValueError(f"Unknown operation: {operation}")


        else:
            if operation == BooleanOperationType.UNION:
                pass
            elif operation == BooleanOperationType.INTERSECTION:
                segments.append(seg)
            elif operation == BooleanOperationType.DIFFERENCE:
                pass
            else:
                raise ValueError(f"Unknown operation: {operation}")

    # Classify segments of curve2 relative to curve1

    for seg in segments2:
        t0, t1 = seg.interval()  # Use start of segment

        mid = np.array(seg.evaluate((t0 + t1) / 2))
        ray = NURBSCurve(np.array([mid, mid + np.array([1, 0, 0.]) * scalar_norm(max_1 - min_1) * 2]), degree=1)
        rtct = list(zip(*ccx(ray, curve1, tol=1e-7)))
        if len(rtct) == 0:
            rt = ()
        else:
            rt, ct = rtct
            rt = unique_with_tolerance(rt, 1e-5)
        b_inside_a = len(rt) % 2
        if b_inside_a:
            if operation == BooleanOperationType.UNION:
                pass
            elif operation == BooleanOperationType.INTERSECTION:
                segments.append(seg)
            elif operation == BooleanOperationType.DIFFERENCE:
                segments.append(seg)
            else:
                raise ValueError(f"Unknown operation: {operation}")

        else:
            if operation == BooleanOperationType.UNION:
                segments.append(seg)
            elif operation == BooleanOperationType.INTERSECTION:
                pass
            elif operation == BooleanOperationType.DIFFERENCE:
                pass
            else:
                raise ValueError(f"Unknown operation: {operation}")

    return segments
