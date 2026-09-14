"""Time an exact curved-graph circle under invertible world-coordinate mixes.

This is a separate ordinary SSX integration probe, not a comparison with
historical native extensions. Run after other test jobs finish for useful
local timing samples. Every variant uses the same tolerance and work cap.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def source_pair():
    first = np.array([[[i/2., j/2., i*j/4.] for j in range(3)] for i in range(3)])
    second = first.copy()
    square = (1., -1., 1.)
    second[..., 2] += [[square[i]+square[j]-.5 for j in range(3)] for i in range(3)]
    return first, second


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=30)
    args = parser.parse_args()
    from mmcore.numeric.intersection.ssx import _bez_ssx5 as ssx
    from mmcore.numeric.intersection.csx import _bez_csx4 as csx
    from examples.ssx.ssx5_analytic_audit import distances_to_polylines

    files = sorted(set((ROOT/'mmcore/numeric/intersection').rglob('*.py'))
                   | set((ROOT/'mmcore/numeric').glob('*.py')))
    def hashes():
        return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in files}
    before = hashes()
    report = dict(git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                       cwd=ROOT, text=True).strip(), source_hashes_before=before,
                  python=sys.version, parameters=dict(atol=.001, rational=False,
                       max_cells=60000, max_csx_calls=10000, max_depth='default',
                       watchdog_seconds=args.timeout),
                  source_definition='S1=(s,t,s*t); S2=(u,v,u*v+4*(u-.5)^2+4*(v-.5)^2-.5)',
                  rows=[])
    matrices = [('identity', np.eye(3)),
                ('mixed', np.array([[1., 1., 0.], [0., 1., 1.], [1., 0., 1.]])),
                ('sheared', np.array([[1., 2., 1.], [0., 1., 1.], [1., 1., 1.]]))]
    original = (ssx.bez_csx, csx.bez_ccx_v4)
    def expired(*_):
        raise TimeoutError
    signal.signal(signal.SIGALRM, expired)
    for label, matrix in matrices:
        for swap in (False, True):
            first, second = (net@matrix.T for net in source_pair())
            if swap:
                first, second = second, first
            row = dict(world_transform=label, matrix=matrix.tolist(), swapped=swap,
                       raw_csx_calls=0, raw_csx_cells=0, raw_csx_seconds=0.,
                       nested_ccx_calls=0, nested_ccx_cells=0, nested_ccx_seconds=0.)
            def count(function, prefix):
                def measured(*a, **kw):
                    row[prefix+'_calls'] += 1
                    start = time.perf_counter()
                    try:
                        result = function(*a, **kw)
                    finally:
                        row[prefix+'_seconds'] += time.perf_counter()-start
                    if result is not None:
                        row[prefix+'_cells'] += int(result.get('cells_processed', 0))
                    return result
                return measured
            ssx.bez_csx = count(original[0], 'raw_csx')
            csx.bez_ccx_v4 = count(original[1], 'nested_ccx')
            started = time.perf_counter()
            signal.alarm(args.timeout)
            try:
                result = ssx.bez_ssx(first, second, atol=.001, rational=False,
                                     max_cells=60000, max_csx_calls=10000)
                row.update(complete=bool(result['complete']), status=result['status'],
                           branches=len(result['branches']),
                           closed=[bool(branch.closed) for branch in result['branches']])
            except TimeoutError:
                result = None
                row['timeout'] = True
            finally:
                signal.alarm(0)
                ssx.bez_csx, csx.bez_ccx_v4 = original
                row['solve_seconds'] = time.perf_counter()-started
            if result is not None:
                paths = [np.asarray(branch.curve[1]) for branch in result['branches']]
                angles = np.linspace(0., 2*np.pi, 4097)
                s, t = .5+np.sqrt(1/8)*np.cos(angles), .5+np.sqrt(1/8)*np.sin(angles)
                reference = np.column_stack((s, t, s*t))@matrix.T
                row['coverage_error'] = (float(distances_to_polylines(reference, paths).max())
                                         if paths else None)
                row['output_length'] = sum(float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) for p in paths)
                row['reference_length'] = float(np.linalg.norm(np.diff(reference, axis=0), axis=1).sum())
                travels = []
                for path in paths:
                    source_xyz = np.linalg.solve(matrix, path.T).T
                    theta = np.arctan2(source_xyz[:, 1]-.5, source_xyz[:, 0]-.5)
                    delta = np.arctan2(np.sin(np.diff(theta)), np.cos(np.diff(theta)))
                    travels.append((float(delta.sum()), float(np.abs(delta).sum())))
                row['angular_travel'] = travels
                row['coverage_pass'] = bool(row['complete'] and row['branches'] == 1
                    and row['closed'] == [True] and row['coverage_error'] < .004
                    and abs(row['output_length']-row['reference_length']) < 8*np.pi*.001
                    and abs(abs(travels[0][0])-2*np.pi) < 1e-6
                    and abs(travels[0][1]-2*np.pi) < 1e-6)
            report['rows'].append(row)
            print(json.dumps(row), flush=True)
    report['source_hashes_after'] = hashes()
    report['sources_unchanged'] = before == report['source_hashes_after']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    return 0 if report['sources_unchanged'] and all(row.get('coverage_pass') for row in report['rows']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
