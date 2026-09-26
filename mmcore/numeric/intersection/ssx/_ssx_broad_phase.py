"""Cheap, tolerance-padded control-hull exclusions for NURBS patch pairs.

PCA/OBB directions and GJK only propose separating axes. An exclusion
always comes from projections of both complete Cartesian control nets.
Positive rational weights, validated by the NURBS entry point, put each
surface inside its control hull. Ambiguous arithmetic retains the pair.
"""
from dataclasses import dataclass

import numpy as np

from mmcore.numeric.algorithms import cygjk

# Source checkouts with an older native extension still get the axis filter.
# A rebuilt extension adds the optional, independently checked GJK proposal.
_gjk_axis = getattr(cygjk, 'gjk_separating_axis', None)
_ROUNDOFF = 128. * np.finfo(float).eps
_PROJECTION_ENTRIES = 262144
_GJK_MAX_CONTROL_PRODUCT = 65536  # bound the native driver's quadratic cache


@dataclass
class _PatchBounds:
    offsets: np.ndarray
    origins: np.ndarray
    axes: np.ndarray
    coordinate_scale: np.ndarray
    lookup: np.ndarray


def _prepare_patches(patches, indices):
    """Cache only participating patches; decomposition preserves degree."""
    indices = np.unique(indices)
    points = np.stack([np.asarray(patches[i].control_points, dtype=float)
                       .reshape(-1, 3) for i in indices])
    # Half first avoids overflow in min+max. The origin need not be exact:
    # the same stored origin is used for every control point in a patch.
    origins = .5*points.min(axis=1) + .5*points.max(axis=1)
    offsets = points - origins[:, None, :]
    scale = np.max(np.abs(offsets), axis=(1, 2))
    finite = np.isfinite(offsets).all(axis=(1, 2)) & np.isfinite(scale)
    axes = np.zeros((len(indices), 3, 3))
    if np.any(finite):
        scaled = offsets[finite] / np.maximum(scale[finite, None, None],
                                              np.finfo(float).tiny)
        scaled -= scaled.mean(axis=1, keepdims=True)
        covariance = np.einsum('nki,nkj->nij', scaled, scaled)
        try:
            _, vectors = np.linalg.eigh(covariance)
            axes[finite] = vectors.transpose(0, 2, 1)
        except np.linalg.LinAlgError:
            # Failed fitting provides no axes, not an exclusion.
            pass
    lookup = np.full(len(patches), -1, dtype=np.intp)
    lookup[indices] = np.arange(len(indices))
    return _PatchBounds(offsets, origins, axes,
                        np.max(np.abs(points), axis=1), lookup)


def _separated_on_axes(first, second, delta, axes, coordinate_scale, atol):
    """Check complete support intervals, including projection roundoff.

    The local offsets and origin difference avoid subtracting two large
    projected world coordinates. The error cushion also includes their
    subtractions at the original coordinate scale. Axis-fitting accuracy
    is immaterial: every finite nonzero direction is a valid test axis.
    """
    lengths = np.linalg.norm(axes, axis=-1)
    valid = np.isfinite(lengths) & (lengths > np.finfo(float).tiny)
    axes = axes / np.where(valid, lengths, 1.)[..., None]
    def interval(points, shift=None):
        # Also chunk the control axis: even one unusually high-degree pair
        # must not allocate an unrestricted controls-by-directions tensor.
        block = max(1, _PROJECTION_ENTRIES // (len(points)*axes.shape[1]))
        low = np.full(axes.shape[:2], np.inf)
        high = np.full(axes.shape[:2], -np.inf)
        for begin in range(0, points.shape[1], block):
            projection = np.einsum('npi,nai->npa', points[:, begin:begin+block], axes)
            if shift is not None:
                projection += shift[:, None, :]
            low = np.minimum(low, projection.min(axis=1))
            high = np.maximum(high, projection.max(axis=1))
        return low, high

    low_a, high_a = interval(first)
    low_b, high_b = interval(second, np.einsum('ni,nai->na', delta, axes))
    valid &= (np.isfinite(low_a) & np.isfinite(high_a)
              & np.isfinite(low_b) & np.isfinite(high_b))
    # Multiply by epsilon before summing, so a finite world magnitude does
    # not overflow merely while computing its (much smaller) error bound.
    error = np.einsum('ni,nai->na',
                      np.maximum(1., coordinate_scale)*_ROUNDOFF, np.abs(axes))
    clearance = np.nextafter(2.*atol + error, np.inf)
    gap = np.maximum(low_b-high_a, low_a-high_b)
    return np.any(valid & np.isfinite(gap) & (gap > clearance), axis=1)


def _filter_hull_pairs(patches1, patches2, pairs, atol, *, stats=None,
                       use_gjk=True):
    """Retain possibly contacting AABB candidates in their original order.

    Both hulls have an ``atol`` cushion. A PCA frame is cached once per
    participating patch. Its six face directions and nine cross directions
    supply inexpensive OBB-style axes, but projections use the actual
    controls rather than the looser fitted boxes. GJK may propose one more
    axis; neither its Boolean verdict nor an unconverged distance is used.
    """
    if stats is not None:
        stats.update(input_pairs=len(pairs), axis_rejected=0, gjk_rejected=0,
                     gjk_calls=0, candidate_pairs=len(pairs), cached_patches=(0, 0))
    if len(pairs) == 0 or not np.isfinite(atol) or atol < 0.:
        return list(pairs)
    indices = np.asarray(pairs, dtype=np.intp)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
        first = _prepare_patches(patches1, indices[:, 0])
        second = _prepare_patches(patches2, indices[:, 1])
        ai = first.lookup[indices[:, 0]]
        bi = second.lookup[indices[:, 1]]
        if stats is not None:
            stats['cached_patches'] = (len(first.origins), len(second.origins))
        # Keep projection temporaries bounded for high-degree control nets.
        count = max(first.offsets.shape[1], second.offsets.shape[1])
        batch_size = max(1, min(1024, _PROJECTION_ENTRIES // (15*count)))
        keep = np.ones(len(pairs), dtype=bool)
        for begin in range(0, len(pairs), batch_size):
            end = min(begin+batch_size, len(pairs))
            a, b = ai[begin:end], bi[begin:end]
            fa, fb = first.axes[a], second.axes[b]
            cross = np.cross(fa[:, :, None, :], fb[:, None, :, :]).reshape(-1, 9, 3)
            axes = np.concatenate((fa, fb, cross), axis=1)
            delta = second.origins[b]-first.origins[a]
            scale = np.maximum(first.coordinate_scale[a], second.coordinate_scale[b])
            keep[begin:end] = ~_separated_on_axes(
                first.offsets[a], second.offsets[b], delta, axes, scale, atol)
        if stats is not None:
            stats['axis_rejected'] = int(np.count_nonzero(~keep))

        product = first.offsets.shape[1]*second.offsets.shape[1]
        if use_gjk and _gjk_axis is not None and product <= _GJK_MAX_CONTROL_PRODUCT:
            for index in np.flatnonzero(keep):
                a, b = ai[index], bi[index]
                delta = second.origins[b]-first.origins[a]
                pa, pb = first.offsets[a], second.offsets[b]+delta
                if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
                    continue
                if stats is not None:
                    stats['gjk_calls'] += 1
                axis = _gjk_axis(pa, pb, tol=1e-12, max_iter=25)
                if axis is None:
                    continue
                axis = np.asarray(axis, dtype=float)
                if axis.shape != (3,) or not np.isfinite(axis).all():
                    continue
                scale = np.maximum(first.coordinate_scale[a], second.coordinate_scale[b])
                separated = _separated_on_axes(
                    first.offsets[a:a+1], second.offsets[b:b+1], delta[None, :],
                    axis[None, None, :], scale[None, :], atol)[0]
                if separated:
                    keep[index] = False
                    if stats is not None:
                        stats['gjk_rejected'] += 1
    if stats is not None:
        stats['candidate_pairs'] = int(np.count_nonzero(keep))
    return [pair for pair, retained in zip(pairs, keep) if retained]


def _filter_patch_pairs(patches1, patches2, pairs, atol, *, stats=None,
                        use_gjk=True, refine=True):
    """Filter whole patch owners, with bounded refinement of loose hulls.

    Refinement subdivides bounds only. The surviving original patch pair,
    its domain, requested accuracy, and narrow-phase allowance stay intact.
    """
    kept = _filter_hull_pairs(patches1, patches2, pairs, atol,
                              stats=stats, use_gjk=use_gjk)
    if refine and kept and np.isfinite(atol) and atol >= 0.:
        from mmcore.numeric.intersection.ssx._ssx_hull_refine import _refine_patch_pairs

        def filter_children(first, second, candidates, clearance):
            return _filter_hull_pairs(first, second, candidates, clearance,
                                      use_gjk=use_gjk)

        kept = _refine_patch_pairs(patches1, patches2, kept, atol,
                                   filter_hulls=filter_children, stats=stats)
        if stats is not None:
            stats['candidate_pairs'] = len(kept)
    return kept
