"""Bounded, private child hulls for excluding original NURBS patch pairs.

Subdivision only tightens enclosures.  Surviving pairs still identify the
original patches, so neither the narrow-phase domains nor its tolerance change.
"""
from dataclasses import dataclass

import numpy as np


_PAIR_BATCH_SIZE = 1024
_MAX_CHILD_PAIRS = 65536
_MAX_SPLIT_WORK = 262144
_ROUNDOFF = 128. * np.finfo(float).eps


@dataclass(frozen=True)
class _HullPatch:
    control_points: np.ndarray
    weights: np.ndarray


def _split_hull(patch):
    """Return two positive-weight midpoint hulls, or decline refinement.

    Normalizing all weights by their maximum preserves rational geometry and
    prevents multiplying Cartesian coordinates by weights larger than one.
    Subnormal normalized weights are declined before homogeneous arithmetic.
    """
    points = np.asarray(patch.control_points, dtype=float)
    weights = np.asarray(patch.weights, dtype=float)
    if (points.ndim != 3 or points.shape[-1] != 3
            or weights.shape != points.shape[:2]
            or not weights.size or not np.isfinite(points).all()
            or not np.isfinite(weights).all() or np.any(weights <= 0.)):
        return None
    with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
        weights = weights / weights.max()
        if not np.isfinite(weights).all() or np.any(weights < np.finfo(float).tiny):
            return None
        degree = np.array(points.shape[:2], dtype=int) - 1
        if not np.any(degree > 0):
            return None
        scaled = points / max(1., float(np.max(np.abs(points))))
        curvature = np.zeros(2)
        for axis in range(2):
            if degree[axis] >= 2:
                difference = np.diff(scaled, n=2, axis=axis)
                curvature[axis] = (degree[axis] * (degree[axis] - 1)
                                   * np.max(np.linalg.norm(difference, axis=-1)))
        if np.any(curvature > 0.):
            axis = int(np.argmax(curvature))
        else:
            lengths = [float(np.max(np.linalg.norm(np.diff(scaled, axis=a), axis=-1)))
                       if degree[a] else -1. for a in range(2)]
            axis = int(np.argmax(lengths))
        # Midpoint de Casteljau is quadratic in the split degree. A large
        # single-span input must not spend unbounded setup work here before
        # the narrow phase can apply its own allowance.
        if points.shape[axis] * weights.size > _MAX_SPLIT_WORK:
            return None
        homogeneous = np.concatenate((points * weights[..., None], weights[..., None]), axis=-1)
        if not np.isfinite(homogeneous).all():
            return None
        work = np.moveaxis(homogeneous, axis, 0)
        left, right = [work[0].copy()], [work[-1].copy()]
        while len(work) > 1:
            # Half before addition also handles finite coordinates near the
            # largest representable value without overflowing their sum.
            work = .5 * work[:-1] + .5 * work[1:]
            left.append(work[0].copy())
            right.append(work[-1].copy())
        children = []
        for controls in (left, right[::-1]):
            net = np.moveaxis(np.stack(controls), 0, axis)
            child_weights = net[..., -1]
            if (not np.isfinite(net).all()
                    or np.any(child_weights < np.finfo(float).tiny)):
                return None
            child_points = net[..., :3] / child_weights[..., None]
            if not np.isfinite(child_points).all():
                return None
            children.append(_HullPatch(child_points, child_weights))
    return tuple(children)


def _refine_patch_pairs(patches1, patches2, pairs, atol, *, filter_hulls,
                        stats=None, max_depth=2):
    """Return an ordered subset of original pairs, using at most two levels.

    ``filter_hulls`` must conservatively retain possible contacts at its
    supplied tolerance.  Failed construction leaves that node unsplit;
    exhausted comparison capacity retains every affected original parent.
    """
    original = list(pairs)
    depth_limit = min(2, max(0, int(max_depth)))
    if stats is not None:
        stats.update(refinement_input_pairs=len(original), refinement_rejected=0,
                     refinement_depth=0, refinement_child_pairs=0,
                     refinement_cached_children=0, refinement_fallback_pairs=0)
    if not original or depth_limit == 0 or not np.isfinite(atol) or atol < 0.:
        return original

    nodes, lookups = [], []
    coordinate_scale, max_degree = 1., 0
    for side, patches in enumerate((patches1, patches2)):
        ids = dict.fromkeys(pair[side] for pair in original)
        lookup = {index: i for i, index in enumerate(ids)}
        side_nodes = [patches[index] for index in ids]
        for patch in side_nodes:
            points = np.asarray(patch.control_points, dtype=float)
            if not np.isfinite(points).all():
                return original
            coordinate_scale = max(coordinate_scale, float(np.max(np.abs(points))))
            max_degree = max(max_degree, max(points.shape[:2]) - 1)
        nodes.append(side_nodes)
        lookups.append(lookup)
    active = [(lookups[0][a], lookups[1][b], owner)
              for owner, (a, b) in enumerate(original)]
    caches = [{}, {}]
    protected = set()
    cached_children = comparisons = completed_depth = 0

    def children(side, index):
        nonlocal cached_children
        if index not in caches[side]:
            split = _split_hull(nodes[side][index])
            if split is None:
                caches[side][index] = (index,)
            else:
                start = len(nodes[side])
                nodes[side].extend(split)
                caches[side][index] = (start, start + 1)
                cached_children += 2
        return caches[side][index]

    for depth in range(1, depth_limit + 1):
        # This cushion is used only for hull exclusions.  The caller sends
        # original patches and its original atol to the actual SSX solver.
        error = (_ROUNDOFF * (max_degree + 3) * depth) * coordinate_scale
        child_atol = float(atol) + error
        if not np.isfinite(child_atol):
            protected.update(owner for _, _, owner in active)
            break
        pending, next_active = [], []
        level_comparisons = 0

        def flush():
            if not pending:
                return
            candidate_pairs = [(a, b) for a, b, _ in pending]
            try:
                kept = set(filter_hulls(nodes[0], nodes[1], candidate_pairs, child_atol))
            except (ArithmeticError, ValueError, np.linalg.LinAlgError):
                protected.update(owner for _, _, owner in pending)
            else:
                next_active.extend(item for item in pending if item[:2] in kept)
            pending.clear()

        for position, (a, b, owner) in enumerate(active):
            if owner in protected:
                continue
            ca, cb = children(0, a), children(1, b)
            count = len(ca) * len(cb)
            if level_comparisons + count > _MAX_CHILD_PAIRS:
                protected.update(item[2] for item in active[position:])
                break
            pending.extend((aa, bb, owner) for aa in ca for bb in cb)
            level_comparisons += count
            if len(pending) >= _PAIR_BATCH_SIZE:
                flush()
        flush()
        comparisons += level_comparisons
        completed_depth = depth
        active = [item for item in next_active if item[2] not in protected]
        if not active:
            break
    retained = protected | {owner for _, _, owner in active}
    result = [pair for owner, pair in enumerate(original) if owner in retained]
    if stats is not None:
        stats.update(refinement_rejected=len(original) - len(result),
                     refinement_depth=completed_depth,
                     refinement_child_pairs=comparisons,
                     refinement_cached_children=cached_children,
                     refinement_fallback_pairs=len(protected))
    return result
