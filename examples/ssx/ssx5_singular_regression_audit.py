"""Run singular regressions with independent watchdogs and source manifests.

Example (use the repository's configured Python environment)::

    python examples/ssx/ssx5_singular_regression_audit.py \
        --output /tmp/singular-audit.json --timeout 30 --workers 2

The JSON contains every node outcome, failure output, log hash, and the
complete before/after Python source manifest for mmcore and SSI fixtures.
Existing output/log paths are refused so a later run cannot erase history.
The process exits nonzero for test failures or watchdogs, after saving all
outcomes. This is a diagnostic runner, not a change to pytest expectations.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=30.)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--test-file', default='tests/test_bez_ssx5_singular.py',
                        help='Repository-relative pytest file; default is the full singular suite.')
    parser.add_argument('--log-dir', type=Path,
                        help='Default: OUTPUT_STEM-logs beside the JSON.')
    args = parser.parse_args(argv)
    if args.timeout <= 0 or not args.timeout < float('inf') or args.workers <= 0:
        parser.error('timeout must be finite and positive; workers must be positive')
    test_path = (ROOT/args.test_file).resolve()
    try:
        test_file = str(test_path.relative_to(ROOT))
    except ValueError:
        parser.error('test-file must be inside this repository')
    if not test_path.is_file():
        parser.error(f'test-file does not exist: {test_file}')
    output = args.output.resolve()
    logs = (args.log_dir.resolve() if args.log_dir is not None
            else output.with_name(output.stem+'-logs'))
    if output.exists() or logs.exists():
        parser.error('output and log-dir must be new paths; preserve previous audit history')
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT)+(os.pathsep+env['PYTHONPATH']
                                  if env.get('PYTHONPATH') else '')
    command = [sys.executable, '-m', 'pytest', test_file, '--collect-only', '-q']
    collected = subprocess.run(command, cwd=ROOT, env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               timeout=max(30., args.timeout))
    if collected.returncode:
        parser.exit(collected.returncode, collected.stdout)
    nodes = [line.strip() for line in collected.stdout.splitlines()
             if line.startswith(test_file+'::')]
    if not nodes:
        parser.error('pytest collected no nodes from the requested file')

    paths = sorted(set(ROOT.glob('mmcore/**/*.py'))
                   | set(ROOT.glob('examples/ssx/**/*.py')) | {test_path})
    lock = threading.Lock()
    manifests = {}

    def snapshot():
        hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in paths}
        key = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        with lock:
            manifests[key] = hashes
        return key

    started = datetime.now(timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=False)
    # Reserve the output exclusively, then use atomic replacements for
    # progress updates so readers cannot observe half a JSON document.
    with output.open('x') as stream:
        stream.write('{}\n')
    (logs/'nodes.json').write_text(json.dumps(nodes, indent=2)+'\n')
    results = []

    def save(done=False):
        with lock:
            source_manifests = dict(manifests)
        payload = {
            'started_utc': started,
            'finished_utc': datetime.now(timezone.utc).isoformat() if done else None,
            'collected': len(nodes), 'watchdog_seconds': args.timeout,
            'workers': args.workers, 'python': sys.executable,
            'python_version': sys.version, 'platform': platform.platform(),
            'repository': str(ROOT), 'test_file': test_file,
            'runner': str(Path(__file__).relative_to(ROOT)),
            'command': [sys.executable, str(Path(__file__).relative_to(ROOT)),
                        '--output', str(output), '--timeout', str(args.timeout),
                        '--workers', str(args.workers), '--test-file', test_file,
                        '--log-dir', str(logs)],
            'complete': done,
            'counts': {status: sum(row['status'] == status for row in results)
                       for status in ('pass', 'fail', 'timeout')},
            'source_manifests': source_manifests,
            'results': sorted(results, key=lambda row: row['index']),
        }
        temporary = output.with_name(output.name+'.writing')
        temporary.write_text(json.dumps(payload, indent=2)+'\n')
        temporary.replace(output)

    def run(item):
        index, node = item
        before = snapshot()
        start = time.monotonic()
        log = logs/f'{index:03d}.txt'
        with log.open('w') as stream:
            try:
                result = subprocess.run(
                    [sys.executable, '-m', 'pytest', node, '-q', '--tb=short'],
                    cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                    timeout=args.timeout)
                status = 'pass' if result.returncode == 0 else 'fail'
                code = result.returncode
            except subprocess.TimeoutExpired:
                status, code = 'timeout', None
        after = snapshot()
        body = log.read_text()
        return {
            'index': index, 'node': node, 'status': status, 'exit': code,
            'seconds': time.monotonic()-start, 'log': str(log),
            'source_snapshot': before, 'source_end_snapshot': after,
            'source_changed_during_test': before != after,
            'output': body[-16000:],
            'log_sha256': hashlib.sha256(body.encode()).hexdigest(),
        }

    save()
    print(f'collected {len(nodes)}; watchdog {args.timeout:g}s; workers {args.workers}', flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run, item) for item in enumerate(nodes)]):
            row = future.result()
            results.append(row)
            save()
            print(row['index'], row['status'], round(row['seconds'], 2), row['node'], flush=True)
    save(True)
    counts = {status: sum(row['status'] == status for row in results)
              for status in ('pass', 'fail', 'timeout')}
    print('totals', counts, flush=True)
    return int(any(row['status'] != 'pass' for row in results))


if __name__ == '__main__':
    raise SystemExit(main())
