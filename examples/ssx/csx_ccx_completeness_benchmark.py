"""Reproduce CSX substrate timings and topology against a git revision.

Run from the repository with the interpreter that can import its native
extensions, for example::

    python examples/ssx/csx_ccx_completeness_benchmark.py \
        --baseline-ref 76735f9 --output docs/ssx_completeness_benchmark.json

The baseline reloads the two solver modules from git, including the nested
CCX implementation. Other shared support and native modules come from the
current checkout; this is a solver-module comparison, not a historical
installation benchmark. Every CSX fixture gets the same tolerance, cell
cap, output cap, and watchdog. The end-to-end SSX case11 row uses the
existing 60,000-cell regression allowance.
"""
import argparse
from collections import Counter
import importlib.util
import hashlib
import json
import platform
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_revision_module(ref, relative, name, directory):
    source = subprocess.run(['git', 'show', f'{ref}:{relative}'], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout
    path = Path(directory)/f'{name}.py'
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fixtures(csx):
    from examples.ssx.bez_ssx5_case10 import S1 as A10, S2 as B10
    from examples.ssx.bez_ssx5_case11 import S1 as A11, S2 as B11
    result = []
    for label, a, b in [('case10', A10, B10), ('case11', A11, B11)]:
        for surface, other, direction in [(a, b, 'A-B'), (b, a, 'B-A')]:
            for axis in (0, 1):
                left, _ = csx._subdivide_surface(surface, axis, .5)
                curve = left[-1, :, :] if axis == 0 else left[:, -1, :]
                result.append((f'{label} {direction} axis{axis}', curve, other))
    plane = np.array([[[0., 0., 0.], [0., 1., 0.]],
                      [[1., 0., 0.], [1., 1., 0.]]])
    result.append(('near-plane-nonzero-gap',
                   np.array([[.25, 0., -1/65536], [.25, 1., -1/65536]]), plane))
    folded = np.array([[[x, v, 0.] for v in (0., 1.)] for x in (.25, -.25, .25)])
    result.append(('folded-two-regular-preimages',
                   np.array([[1/16, .5, -.5], [1/16, .5, .5]]), folded))
    degree = 10
    tangent = np.column_stack((np.full(degree+1, .25),
                               np.linspace(0., 1., degree+1),
                               [(-1.)**i/1024 for i in range(degree+1)]))
    result.append(('degree10-single-multiple-root', tangent, plane))
    return result


def ccx_fixtures():
    a = np.array([[0., 0., 0.], [.5, 0., 0.], [1., 1., 0.]])
    b = np.array([[.5, .25, 0.], [.5, -.125, 0.], [.5, .5, 0.]])
    cancellation_a = np.array([[-5., -19., 0.], [21., 18., 0.], [-37., -17., 0.]])*2**20
    cancellation_b = np.array([[-29., 12., 0.], [-25., 14., 0.], [79., -40., 0.]])*2**20
    degree = 10
    tangent = np.column_stack((np.linspace(0., 1., degree+1),
                               [(-1.)**i*2.**-degree for i in range(degree+1)],
                               np.zeros(degree+1)))
    line = np.array([[0., 0., 0.], [1., 0., 0.]])
    return [('same-u-two-partner-preimages', a, b),
            ('generic-quadratic-squared-cancellation', cancellation_a, cancellation_b),
            ('curve-line-degree10-multiple-root', tangent, line)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ref', default='76735f9')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--timeout', type=int, default=15)
    parser.add_argument('--update-ssx', action='store_true',
                        help='Refresh only the SSX row in an existing output; retain raw rows and history.')
    args = parser.parse_args()
    previous = None
    if args.update_ssx:
        if args.output is None or not args.output.exists():
            parser.error('--update-ssx requires an existing --output JSON')
        previous = json.loads(args.output.read_text())
        if previous['baseline_ref'] != args.baseline_ref:
            parser.error('existing output has a different baseline reference')
    source_paths = sorted(set((ROOT/'mmcore/numeric/intersection').rglob('*.py'))
                          | set((ROOT/'mmcore/numeric').glob('*.py')))
    def source_hashes():
        return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in source_paths}
    initial_hashes = source_hashes()
    current_ref = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
        capture_output=True, text=True, check=True).stdout.strip()
    from mmcore.numeric.intersection.csx import _bez_csx4 as current
    from mmcore.numeric.intersection.ccx import _bez_ccx4 as current_ccx
    from mmcore.numeric.intersection.ssx import _bez_ssx5 as ssx

    def timeout(*_):
        raise TimeoutError

    signal.signal(signal.SIGALRM, timeout)
    rows = []
    ccx_rows = []
    if not args.update_ssx:
        with tempfile.TemporaryDirectory(prefix='mmcore-csx-baseline-') as directory:
            old_csx = load_revision_module(args.baseline_ref,
                'mmcore/numeric/intersection/csx/_bez_csx4.py', '_benchmark_old_csx', directory)
            old_ccx = load_revision_module(args.baseline_ref,
                'mmcore/numeric/intersection/ccx/_bez_ccx4.py', '_benchmark_old_ccx', directory)
            old_csx.bez_ccx_v4 = old_ccx.bez_ccx
            modes = [('baseline', old_csx, {}), ('current-public', current, {}),
                     ('current-exact', current, {'tolerance_tier': False})]
            for name, curve, surface in fixtures(current):
                for mode, module, kwargs in modes:
                    row = {'name': name, 'mode': mode, 'ccx_calls': 0,
                           'ccx_cells': 0, 'ccx_seconds': 0.}
                    nested = module.bez_ccx_v4

                    def counted(*call_args, **call_kwargs):
                        row['ccx_calls'] += 1
                        started = time.perf_counter()
                        try:
                            result = nested(*call_args, **call_kwargs)
                        finally:
                            row['ccx_seconds'] += time.perf_counter()-started
                        row['ccx_cells'] += int(result.get('cells_processed', 0))
                        return result

                    module.bez_ccx_v4 = counted
                    signal.alarm(args.timeout)
                    started = time.perf_counter()
                    try:
                        result = module.bez_csx(curve, surface, atol=1e-3,
                            rational=False, max_cells=10_000, max_results=4096, **kwargs)
                        row.update(cells=result['cells_processed'], roots=len(result['isolated']),
                                   overlaps=len(result['overlaps']),
                                   complete=not result['budget_exhausted'] and result['boundary_topology_complete'],
                                   truncation_cause=result.get('truncation_cause'))
                    except TimeoutError:
                        row['timeout'] = True
                    finally:
                        signal.alarm(0)
                        module.bez_ccx_v4 = nested
                        row['seconds'] = time.perf_counter()-started
                    rows.append(row)
                    print(json.dumps(row), flush=True)

            for name, first, second in ccx_fixtures():
                for version, module in [('baseline', old_ccx), ('current', current_ccx)]:
                    for tier in (True, False):
                        row = {'name': name, 'mode': version+('-public' if tier else '-exact')}
                        signal.alarm(args.timeout)
                        started = time.perf_counter()
                        try:
                            result = module.bez_ccx(first, second, atol=1e-3,
                                rational=False, max_cells=10_000, max_results=4096,
                                tolerance_tier=tier)
                            row.update(cells=result['cells_processed'],
                                       roots=len(result['isolated']), overlaps=len(result['overlaps']),
                                       root_certifications=dict(Counter(r.get('certification') for r in result['isolated'])),
                                       complete=not result['budget_exhausted'] and result['boundary_topology_complete'],
                                       truncation_cause=result.get('truncation_cause'))
                        except TimeoutError:
                            row['timeout'] = True
                        finally:
                            signal.alarm(0)
                            row['seconds'] = time.perf_counter()-started
                        ccx_rows.append(row)
                        print(json.dumps(row), flush=True)

    from examples.ssx.bez_ssx5_case11 import S1, S2
    signal.alarm(args.timeout)
    started = time.perf_counter()
    ssx_row = {'name': 'case11-end-to-end', 'mode': 'current-exact',
               'raw_csx_calls': 0, 'raw_csx_cells': 0, 'raw_csx_seconds': 0.,
               'nested_ccx_calls': 0, 'nested_ccx_cells': 0, 'nested_ccx_seconds': 0.,
               'source_cut_calls': 0, 'source_cut_cells': 0, 'source_cut_seconds': 0.}
    from mmcore.numeric.intersection.ssx import _ssx_planar_cut as source_cut_module
    original_raw_csx = ssx.bez_csx
    original_nested_ccx = current.bez_ccx_v4
    original_source_cut = source_cut_module.exact_source_planar_cut

    def count_engine(function, prefix):
        def measured(*call_args, **call_kwargs):
            ssx_row[prefix+'_calls'] += 1
            began = time.perf_counter()
            try:
                result = function(*call_args, **call_kwargs)
            finally:
                ssx_row[prefix+'_seconds'] += time.perf_counter()-began
            if result is not None:
                ssx_row[prefix+'_cells'] += int(result.get('cells_processed', 0))
            return result
        return measured

    ssx.bez_csx = count_engine(original_raw_csx, 'raw_csx')
    current.bez_ccx_v4 = count_engine(original_nested_ccx, 'nested_ccx')
    source_cut_module.exact_source_planar_cut = count_engine(original_source_cut, 'source_cut')
    try:
        result = ssx.bez_ssx(S1, S2, atol=1e-3, rational=False,
                            max_cells=60_000, max_csx_calls=10_000)
        ssx_row.update(complete=result['complete'], branches=len(result['branches']),
                       closed=[bool(branch.closed) for branch in result['branches']],
                       status=result['status'])
    except TimeoutError:
        ssx_row['timeout'] = True
    finally:
        signal.alarm(0)
        ssx.bez_csx = original_raw_csx
        current.bez_ccx_v4 = original_nested_ccx
        source_cut_module.exact_source_planar_cut = original_source_cut
        ssx_row['seconds'] = time.perf_counter()-started
    print(json.dumps(ssx_row), flush=True)
    final_hashes = source_hashes()
    output = {'baseline_ref': args.baseline_ref,
              'current_ref': current_ref,
              'current_source_sha256': initial_hashes,
              'sources_changed_during_benchmark': [path for path, digest in initial_hashes.items()
                                                    if final_hashes.get(path) != digest],
              'baseline_scope': 'CSX and nested CCX source modules; current shared/native support',
              'call_graph': ['SSX -> CSX -> CCX', 'CCX does not call CSX'],
              'python': sys.version, 'platform': platform.platform(),
              'csx_parameters': {'atol': .001, 'max_cells': 10000, 'max_results': 4096,
                                 'watchdog_seconds': args.timeout},
              'rows': rows, 'ccx_rows': ccx_rows, 'ssx': ssx_row}
    ssx_row['current_ref'] = current_ref
    ssx_row['source_sha256'] = initial_hashes
    ssx_row['sources_changed_during_run'] = [path for path,digest in initial_hashes.items()
                                             if final_hashes.get(path) != digest]
    ssx_row['parameters'] = dict(atol=.001,rational=False,max_cells=60000,
                                max_csx_calls=10000,max_depth='default',watchdog_seconds=args.timeout)
    if previous is not None:
        output = previous
        output.setdefault('ssx_history', []).append(output['ssx'])
        output['ssx'] = ssx_row
    output['call_graph'] = ['SSX -> exact source cut or source-residual CSX',
                            'public/default CSX -> boundary CCX', 'CCX does not call CSX']
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
