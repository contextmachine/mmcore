"""Supported source data for SSX control-hull and rational certificates."""
import numpy as np


def validate_control_data(points, weights=None, *, context):
    """Reject unsupported data before any destructive hull exclusion."""
    points = np.asarray(points)
    if (points.ndim != 3 or points.shape[-1] != 3
            or not points.shape[0] or not points.shape[1]
            or not np.all(np.isfinite(points))):
        raise ValueError(f'{context}: control coordinates must be a finite, nonempty (nu,nv,3) array')
    if weights is not None:
        weights = np.asarray(weights)
        if (weights.shape != points.shape[:2]
                or not np.all(np.isfinite(weights))
                or not np.all(weights > 0.)):
            raise ValueError(f'{context}: weights must be finite and strictly positive, with shape (nu,nv)')


def validate_bezier_surface(net, *, rational, context):
    dimensions = 4 if rational else 3
    if net.ndim != 3 or net.shape[-1] != dimensions:
        raise ValueError(f'{context}: expected a (nu,nv,{dimensions}) control net')
    validate_control_data(net[...,:3],net[...,3] if rational else None,
                          context=context)
