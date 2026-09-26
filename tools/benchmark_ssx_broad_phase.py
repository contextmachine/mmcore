"""Compare NURBS SSX candidate filtering with the same narrow-phase solver.

Run from any directory, for example::

    .venv/bin/python tools/benchmark_ssx_broad_phase.py \
        examples/ssx/nurbs_nurbs_intersection_2.py --baseline
    .venv/bin/python tools/benchmark_ssx_broad_phase.py \
        examples/ssx/nurbs_nurbs_intersection_2.py

Fixture files are trusted Python source. Their data declarations are executed,
stopping before the usual example CLI, result computation, or main guard.
"""
from pathlib import Path
import argparse
import ast
import dataclasses
import hashlib
import json
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from mmcore.numeric.intersection.ssx import _nssx5 as adapter


def load_surfaces(path, first=None, second=None):
    path = Path(path).resolve()
    nodes = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.If):
            break
        if isinstance(node, ast.Import) and any(a.name == 'logging' for a in node.names):
            break
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in ('args', 'result') for t in node.targets):
            break
        if isinstance(node, ast.ImportFrom) and node.module == 'examples.ssx.common_helpers':
            continue
        nodes.append(node)
    namespace = {'__name__': 'benchmark_fixture', '__file__': str(path)}
    sys.path.insert(0, str(path.parent))
    try:
        exec(compile(ast.Module(nodes, type_ignores=[]), str(path), 'exec'), namespace)
    finally:
        sys.path.pop(0)
    if first is None and second is None:
        first, second = ('s1', 's2') if 's1' in namespace else ('surf1', 'surf2')
    if first is None or second is None:
        raise ValueError('provide both --first and --second')
    return namespace[first], namespace[second]


def geometry_digest(result):
    """Hash all returned geometry, excluding work counters and allowances."""
    digest = hashlib.sha256()

    def token(data):
        digest.update(len(data).to_bytes(8, 'big'))
        digest.update(data)

    def visit(value):
        if isinstance(value, np.ndarray):
            token(b'array')
            token(repr((value.dtype.str, value.shape)).encode())
            token(value.tobytes())
        elif dataclasses.is_dataclass(value):
            token(type(value).__qualname__.encode())
            for item in dataclasses.fields(value):
                token(item.name.encode())
                visit(getattr(value, item.name))
        elif isinstance(value, dict):
            token(b'dict')
            token(str(len(value)).encode())
            for key in sorted(value):
                visit(key)
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            token(type(value).__name__.encode())
            token(str(len(value)).encode())
            for item in value:
                visit(item)
        elif value is None or isinstance(value, (str, bool, int, float, np.generic)):
            scalar = value.item() if isinstance(value, np.generic) else value
            token(repr((type(scalar).__name__, scalar)).encode())
        else:
            raise TypeError(f'unsupported result value: {type(value).__name__}')

    visit({key: result[key] for key in ('branches', 'points', 'singularities', 'overlap_regions')})
    return digest.hexdigest()


def run_once(surfaces, atol, *, baseline=False, use_gjk=True, refine=True):
    original_filter = adapter._filter_patch_pairs
    original_solver = adapter.bez_ssx
    original_decompose = adapter.decompose_surface
    stats = {'bez_ssx_calls': 0, 'bez_ssx_seconds': 0., 'nonempty_pairs': 0,
             'decomposition_seconds': 0., 'patch_counts': []}

    def decompose(*args, **kwargs):
        start = time.perf_counter()
        result = original_decompose(*args, **kwargs)
        stats['decomposition_seconds'] += time.perf_counter()-start
        stats['patch_counts'].append(len(result))
        return result

    def filter_pairs(first, second, pairs, tolerance):
        start = time.perf_counter()
        if baseline:
            result = pairs
            stats.update(input_pairs=len(pairs), candidate_pairs=len(pairs))
        else:
            result = original_filter(first, second, pairs, tolerance,
                                     stats=stats, use_gjk=use_gjk, refine=refine)
        stats['filter_seconds'] = time.perf_counter()-start
        return result

    def solve(*args, **kwargs):
        start = time.perf_counter()
        result = original_solver(*args, **kwargs)
        stats['bez_ssx_seconds'] += time.perf_counter()-start
        stats['bez_ssx_calls'] += 1
        stats['nonempty_pairs'] += int(any(result[key] for key in
            ('branches', 'points', 'singularities', 'overlap_regions')))
        return result

    adapter.decompose_surface = decompose
    adapter._filter_patch_pairs = filter_pairs
    adapter.bez_ssx = solve
    start, cpu_start = time.perf_counter(), time.process_time()
    try:
        result = adapter.nurbs_ssx(*surfaces, atol=atol)
    finally:
        adapter.decompose_surface = original_decompose
        adapter._filter_patch_pairs = original_filter
        adapter.bez_ssx = original_solver
    stats.update(seconds=time.perf_counter()-start, cpu_seconds=time.process_time()-cpu_start,
                 geometry_digest=geometry_digest(result), complete=result['complete'],
                 reasons=result['status']['reasons'],
                 branches=len(result['branches']), points=len(result['points']),
                 singularities=len(result['singularities']), regions=len(result['overlap_regions']))
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fixture', type=Path)
    parser.add_argument('--first')
    parser.add_argument('--second')
    parser.add_argument('--atol', type=float, default=1e-3)
    parser.add_argument('--baseline', action='store_true', help='bypass only the new candidate filters')
    parser.add_argument('--no-gjk', action='store_true')
    parser.add_argument('--no-refinement', action='store_true')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error('--repeat must be positive')
    surfaces = load_surfaces(args.fixture, args.first, args.second)
    trials = []
    for _ in range(args.repeat):
        trial = run_once(surfaces, args.atol, baseline=args.baseline,
                         use_gjk=not args.no_gjk, refine=not args.no_refinement)
        trials.append(trial)
        print(json.dumps(trial), flush=True)
    report = dict(fixture=str(args.fixture.resolve()), atol=args.atol,
                  baseline=args.baseline, use_gjk=not args.no_gjk,
                  refinement=not args.no_refinement, source=adapter.__file__,
                  median_seconds=statistics.median(r['seconds'] for r in trials), trials=trials)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
