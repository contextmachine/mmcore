"""Shared solver-level work budget (ledger L52 — the 8-way accounting merge).

The 2026-07-12 budget review (§10 finding 15) found the same soft-budget
accounting hand-rolled EIGHT times with already-divergent charge semantics:
``_SSXSoftBudget``, ``BernsteinZeroBudget``, the closest-point inline
counters, the ``bez_ccx``/``bez_csx`` inline counters, the ``_nccx4``/
``_ncsx4`` adapter twins (unified first, into
``mmcore/numeric/intersection/_adapter_status.py``), and the
``_ssx5_singular`` closures.  Divergent budget accounting is where a large
fraction of the L4x findings came from; this module is the single home for
the solver-level mechanics, migrated one accounting at a time.

Charge-semantics registry (the EXPLICIT reconciliation the ledger demands —
these families deliberately differ and must not be silently averaged):

- **check-then-charge, all-or-nothing, latching** (``SoftWorkBudget``,
  c3's ``_spend``): a denied amount is never partially spent; once
  exhausted every later charge fails fast WITHOUT re-marking reasons.
  Used where the charge protects the *next* unit of work.
- **clamp-and-charge with a min-1 floor** (`_adapter_status
  .consume_bezier_status`, the closest-point NURBS aggregators): a
  reported spend is billed at ``min(max(1, reported), allowance)`` — every
  dispatch costs at least one unit so a large candidate set cannot bypass
  the aggregate allowance, and overruns clamp rather than deny.  Used
  where the work has ALREADY happened and the ledger only reconciles it.
- **charge-at-completion after truncation** (``BernsteinZeroBudget``
  result accounting): results are counted once, at top level, after
  clamping to the remaining allowance.  Node work in the same object is
  ordinary check-then-charge.
- **shared-remainder threading** (``bez_ccx``/``bez_csx`` locals): phases
  and bounded sub-ledgers draw ``min(remaining, tier)`` from one
  down-counter; a sub-phase never receives a fresh allowance.

The schema-v2 REASON_* vocabulary lives here with the ledger that stamps
it.  ``complete == (not reasons)`` by construction: ``mark_incomplete``
requires a reason, and only root-cause transitions record one.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Callable, Optional


# Reason strings published in result['status']['reasons'] (schema v2,
# 2026-07-12 review doc §6). `complete` is the one bit consumers act on;
# `reasons` says WHY it is False and which reaction can help.
#
# Work family — a resource knob can help:
REASON_WORK_BUDGET = "work_budget"          # shared cell/CSX ledger or a CSX per-call tier ran dry (max_cells / max_csx_calls / *_csx_max_cells)
REASON_OUTPUT_CAP = "output_cap"            # max_output_items reached
REASON_POSTPROCESS_CAP = "postprocess_cap"  # max_postprocess_work reached
REASON_DEPTH_LIMIT = "depth_limit"          # max_depth ceiling left a crossing-bearing cell unresolved
# Structural family — raising budgets cannot help:
REASON_PARAMETER_FIBER = "parameter_fiber"  # positive-dimensional preimage of a boundary point (collapsed edge)
REASON_OVERLAP_REGION = "overlap_region_unsupported"  # 2-D coincidence region detected; retired by L28's SSXOverlapRegion
REASON_TANGENTIAL_ZONE = "unresolved_tangential_zone"  # truncated Δ/Φ tangency enumeration or Φ-loop path not certified
REASON_MULTIPLICITY = "unresolved_multiplicity"  # rank-deficient Δ-root / crossing cluster whose local dimension is unproven
REASON_TRACE_UNVERIFIED = "trace_unverified"  # marched continuation failed the strict Ψ-zero path certificate


@dataclass
class SoftWorkBudget:
    """One shared work budget for an entire solver call.

    Local solver limits are still useful backstops, but they do not compose:
    SSX can invoke thousands of CSX/zero-dimensional searches and each search
    used to receive a fresh allowance.  This object is deliberately tiny and
    callback-friendly so nested solvers spend from the same counter.

    Every transition into a partial state records one of the REASON_*
    strings; ``result_fields`` publishes them as ``status['reasons']`` with
    the invariant ``complete == (not reasons)``. Only root-cause transitions
    record a reason — denials that merely echo an already-exhausted state do
    not re-mark, so ``reasons`` stays a list of causes, not a cascade log.
    """

    max_cells: int
    max_csx_calls: int
    max_output_items: int = 1_024
    max_postprocess_work: Optional[int] = None
    cells_processed: int = 0
    csx_calls: int = 0
    output_items: int = 0
    postprocess_work: int = 0
    postprocess_exhausted: bool = False
    exhausted: bool = False
    incomplete: bool = False
    cell_counts: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)
    # (reason, stuv_global) records for STRUCTURAL marks whose location a
    # later pass can re-examine — ledger L28: an `unresolved_multiplicity`
    # ambiguity whose root lies INSIDE a certified overlap region is a
    # region-interior sample of the 2-D C2 set (resolved by the region),
    # so the assembler may retire it; marks outside any region stay.
    structural_sites: list = field(default_factory=list)

    def __post_init__(self):
        if self.max_postprocess_work is None:
            self.max_postprocess_work = max(0, int(self.max_cells))
        else:
            self.max_postprocess_work = max(
                0, int(self.max_postprocess_work))

    def _add_reason(self, reason: str) -> None:
        if reason not in self.reasons:
            self.reasons.append(reason)

    def charge_cells(self, amount: int = 1, source: str = "nested") -> bool:
        amount = max(0, int(amount))
        if self.exhausted:
            return False
        if self.cells_processed + amount > self.max_cells:
            self.exhausted = True
            self._add_reason(REASON_WORK_BUDGET)
            return False
        self.cells_processed += amount
        self.cell_counts[source] = self.cell_counts.get(source, 0) + amount
        return True

    def charge_csx_call(self) -> bool:
        if self.exhausted:
            return False
        if self.csx_calls >= self.max_csx_calls:
            self.exhausted = True
            self._add_reason(REASON_WORK_BUDGET)
            return False
        self.csx_calls += 1
        return True

    def charge_postprocess(self, amount: int = 1) -> bool:
        """Charge bounded assembly/filter work after the search phase.

        This counter is separate from subdivision cells so a hard-stopped
        search can still assemble its certified partial fragments. It is
        nevertheless call-wide and finite, preventing postprocessing from
        becoming a second unbounded phase.
        """
        amount = max(0, int(amount))
        if self.postprocess_exhausted:
            return False
        if self.postprocess_work + amount > self.max_postprocess_work:
            self.postprocess_exhausted = True
            self.exhausted = True
            self.incomplete = True
            self._add_reason(REASON_POSTPROCESS_CAP)
            return False
        self.postprocess_work += amount
        return True

    @property
    def remaining_cells(self) -> int:
        return max(0, self.max_cells - self.cells_processed)

    @property
    def remaining_postprocess_work(self) -> int:
        return max(0, self.max_postprocess_work - self.postprocess_work)

    def mark_exhausted(self, reason: str = REASON_WORK_BUDGET) -> None:
        self.exhausted = True
        self._add_reason(reason)

    def mark_incomplete(self, reason: str) -> None:
        """Record a partial local result without stopping independent work.

        ``reason`` is mandatory: the caller names the root cause (one of the
        REASON_* strings) so ``status['reasons']`` can steer the consumer
        (raise a knob / wait for typed machinery / accept the resolution
        limit) instead of conflating everything into one budget flag.
        """
        self.incomplete = True
        self._add_reason(reason)

    def retire_reason(self, reason: str) -> None:
        """Remove a structural reason that this same call has since RESOLVED.

        Reasons are published only at :meth:`result_fields`; a condition
        recorded mid-search that later machinery fully represents (ledger
        L28: `overlap_region_unsupported` once every piece of overlap
        evidence is covered by a certified region) may be retired before
        publication.  Hard exhaustion is never retirable — `exhausted`
        always keeps its `work_budget` reason, so `reasons` can only become
        empty when the ledger never ran dry.
        """
        if reason in self.reasons:
            self.reasons.remove(reason)
        if not self.reasons and not self.exhausted:
            self.incomplete = False

    def append_output(self, target: list, value, source: str) -> bool:
        """Append one intermediate/output entity under the global cap."""
        if self.output_items >= self.max_output_items:
            self.mark_incomplete(REASON_OUTPUT_CAP)
            return False
        target.append(value)
        self.output_items += 1
        return True

    def extend_output(self, target: list, values, source: str) -> bool:
        complete = True
        for value in values:
            if not self.append_output(target, value, source):
                complete = False
                break
        return complete

    def result_fields(self) -> dict:
        return {
            "complete": not (self.exhausted or self.incomplete),
            "status": {
                "reasons": list(self.reasons),
                "work": {
                    "cells_processed": int(self.cells_processed),
                    "csx_calls": int(self.csx_calls),
                    "max_cells": int(self.max_cells),
                    "max_csx_calls": int(self.max_csx_calls),
                    "output_items": int(self.output_items),
                    "max_output_items": int(self.max_output_items),
                    "postprocess_work": int(self.postprocess_work),
                    "max_postprocess_work": int(self.max_postprocess_work),
                    "cell_counts": dict(self.cell_counts),
                },
            },
        }


@dataclass
class BernsteinZeroBudget:
    """Scoped work/result budget for nested Bernstein zero searches.

    ``classify_sq_dist_net`` calls the 1-D zero-finder without budget
    arguments.  A context-local budget lets CCX/CSX bound those Phase-1
    calls without a process-global mutable limit (and without changing the
    public classifier API).  ``nodes`` counts recursive solver invocations
    (ordinary check-then-charge).  Results are charged only when a
    top-level boundary solve returns — the charge-at-completion-after-
    truncation family in this module's registry — so shared subdivision
    endpoints do not consume the result allowance repeatedly.  The
    remaining result allowance is nevertheless propagated through
    recursion so a capped solve stops before materializing an unbounded
    result list.
    """

    max_nodes: int
    max_results: int
    nodes: int = 0
    results: int = 0
    active_depth: int = 0
    exhausted: bool = False

    def enter(self) -> bool:
        if self.nodes >= self.max_nodes:
            self.exhausted = True
            return False
        self.nodes += 1
        self.active_depth += 1
        return True

    def leave(self) -> None:
        self.active_depth -= 1

    def remaining_results(self) -> int:
        return max(0, self.max_results - self.results)

    def cap_top_level_results(self, values: list) -> list:
        remaining = self.remaining_results()
        if len(values) > remaining:
            self.exhausted = True
            values = values[:remaining]
        self.results += len(values)
        return values


@dataclass
class LatchingSpend:
    """Check-then-charge, all-or-nothing, latching work ledger.

    Extraction of the ``_spend`` closure from ``c3_pass``
    (``_ssx5_singular``): a local cap plus an OPTIONAL external hook
    (typically :func:`charge_hook` over a shared :class:`SoftWorkBudget`).
    Semantics preserved exactly:

    - a denied amount is never partially spent;
    - once exhausted, every later spend fails fast;
    - the external hook is consulted only AFTER the local check passes, so
      a local denial never phantom-charges the shared ledger — and a
      hook denial leaves the LOCAL counter unspent (the shared ledger has
      latched anyway, so the double-count is moot and matches the closure).
    """

    max_work: int
    charge_external: Optional[Callable[[int], bool]] = None
    work_processed: int = 0
    exhausted: bool = False
    external_exhausted: bool = False

    def __post_init__(self):
        self.max_work = max(0, int(self.max_work))

    def spend(self, amount: int = 1) -> bool:
        amount = max(0, int(amount))
        if self.exhausted:
            return False
        if self.work_processed + amount > self.max_work:
            self.exhausted = True
            return False
        if (self.charge_external is not None
                and not self.charge_external(amount)):
            self.exhausted = True
            self.external_exhausted = True
            return False
        self.work_processed += amount
        return True


_ZERO_BUDGET: ContextVar[Optional[BernsteinZeroBudget]] = ContextVar(
    "bernstein_zero_budget", default=None,
)


@contextmanager
def bernstein_zero_budget(max_nodes: int, max_results: int):
    """Bound all nested ``find_bernstein_zeros_1d`` calls in a scope."""

    budget = BernsteinZeroBudget(
        max_nodes=max(0, int(max_nodes)),
        max_results=max(0, int(max_results)),
    )
    token = _ZERO_BUDGET.set(budget)
    try:
        yield budget
    finally:
        _ZERO_BUDGET.reset(token)


def charge_hook(budget: Optional[SoftWorkBudget],
                source: str) -> Optional[Callable[..., bool]]:
    """Bind a cell-ledger charge callback to ``budget``, or None.

    Single implementation of the guarded-lambda pattern that was repeated at
    every ``charge_box=``/``charge_work=`` wiring site::

        (lambda n: b.charge_cells(n, src)) if b is not None else None

    Consumers accept ``Optional[Callable[[int], bool]]``; the None path
    means "no shared ledger — local caps bind alone".
    """
    if budget is None:
        return None

    def _charge(amount: int = 1) -> bool:
        return budget.charge_cells(amount, source)

    return _charge
