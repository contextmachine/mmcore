"""Shared status ledger for the NURBS-level intersection adapters (ledger L52).

`_nccx4` and `_ncsx4` each carried a hand-rolled copy of the same aggregate
status machinery (~110 lines, copy-pasted and already drifting: comment
wording, key sets, and expression formatting had diverged while the intended
semantics stayed identical). Divergent budget accounting is where a large
fraction of the L4x review findings came from — this module is the single
implementation both adapters now delegate to.

Contract (unchanged from the twins):
- one call-wide ledger dict tracks cells/results processed against the
  aggregate allowances; `complete` / `budget_exhausted` /
  `boundary_topology_complete` summarize honesty flags across span pairs;
- every candidate dispatch charges at least ONE cell — an AABB-fast-rejected
  span reporting zero solver cells must not let an arbitrarily large BVH
  candidate set bypass the aggregate allowance;
- overproduced results are truncated to the remaining result allowance and
  the truncation marks the ledger exhausted;
- with ``return_status=False`` any incompleteness raises RuntimeError
  (fail-fast opt-in; the default since L41 is always-return-status).

Adapter-specific concerns stay in the adapters: CSX's ``parameter_fibers``
list field and its fiber->global parameter mapping.
"""
from __future__ import annotations


def new_status(max_cells, max_results, extra_list_fields=()):
    status = {
        'complete': True,
        'budget_exhausted': False,
        'boundary_topology_complete': True,
        'cells_processed': 0,
        'max_cells': int(max_cells),
        'results_processed': 0,
        'max_results': int(max_results),
        'partial_results': 0,
    }
    for field in extra_list_fields:
        status[field] = []
    return status


def mark_incomplete(status, context, return_status, message):
    status['complete'] = False
    status['budget_exhausted'] = True
    status['partial_results'] += 1
    if not return_status:
        raise RuntimeError(f"{context}: {message}; pass return_status=True "
                           "to receive explicit partial status")


def remaining_allowances(status):
    return (
        max(0, status['max_cells'] - status['cells_processed']),
        max(0, status['max_results'] - status['results_processed']),
    )


def consume_bezier_status(
    result, status, *, incomplete_message, return_status,
    cell_allowance, result_allowance,
    list_keys=('isolated', 'overlaps'),
):
    """Aggregate and sanitize one span result under call-wide allowances.

    Returns ``(sanitized_result, incomplete)``; raises RuntimeError with
    ``incomplete_message`` when incomplete and ``return_status`` is False.
    """
    result = dict(result)
    reported_cells = max(0, int(result.get('cells_processed', 0)))
    charged_cells = max(1, reported_cells)
    cells_overrun = charged_cells > cell_allowance
    status['cells_processed'] += min(charged_cells, cell_allowance)

    kept = 0
    result_overrun = False
    for key in list_keys:
        values = list(result.get(key, ()) or ())
        remaining = max(0, result_allowance - kept)
        if len(values) > remaining:
            values = values[:remaining]
            result_overrun = True
        result[key] = values
        kept += len(values)
    status['results_processed'] += kept

    exhausted = (bool(result.get('budget_exhausted', False))
                 or cells_overrun or result_overrun)
    topology_complete = bool(result.get('boundary_topology_complete', True))
    incomplete = exhausted or not topology_complete

    status['budget_exhausted'] |= exhausted
    status['boundary_topology_complete'] &= topology_complete
    if incomplete:
        status['complete'] = False
        status['partial_results'] += 1
        if not return_status:
            raise RuntimeError(incomplete_message)
    return result, incomplete
