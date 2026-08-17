"""Contract tests for mmcore.numeric._work_budget (ledger L52, slice 5).

The shared budget module is the single implementation of the solver-level
work-ledger mechanics that were previously hand-rolled 8 ways (review doc
2026-07-12 §10 finding 15).  These tests pin the EXACT semantics extracted
from `_bez_ssx5._SSXSoftBudget` — the migration must not shift any of them:

- charge_cells is check-then-charge and all-or-nothing (a denied amount is
  not partially spent);
- denials that echo an already-exhausted ledger do NOT re-mark reasons
  (reasons is a cause list, not a cascade log);
- zero-amount charges are true no-ops that succeed;
- postprocess has its own latch so a hard-stopped search can still afford
  assembly, and its cap defaults to max_cells;
- retire_reason never clears hard exhaustion;
- complete == (not reasons) holds through mark/retire cycles;
- charge_hook reproduces the guarded-lambda Optional threading pattern.
"""
import pytest

from mmcore.numeric._work_budget import (
    SoftWorkBudget,
    charge_hook,
    REASON_WORK_BUDGET,
    REASON_OUTPUT_CAP,
    REASON_POSTPROCESS_CAP,
    REASON_DEPTH_LIMIT,
    REASON_PARAMETER_FIBER,
    REASON_OVERLAP_REGION,
    REASON_TANGENTIAL_ZONE,
    REASON_MULTIPLICITY,
    REASON_TRACE_UNVERIFIED,
)


def test_charge_cells_is_all_or_nothing():
    b = SoftWorkBudget(max_cells=10, max_csx_calls=5)
    assert b.charge_cells(8, "a")
    # 8 + 5 > 10: the whole charge is denied, nothing is spent.
    assert not b.charge_cells(5, "a")
    assert b.cells_processed == 8
    assert b.exhausted
    assert b.reasons == [REASON_WORK_BUDGET]


def test_echo_denials_do_not_cascade_reasons():
    b = SoftWorkBudget(max_cells=1, max_csx_calls=5)
    assert b.charge_cells(1, "a")
    assert not b.charge_cells(1, "a")   # root-cause denial
    assert not b.charge_cells(1, "b")   # echo — must not re-mark
    assert not b.charge_csx_call()      # echo through the other ledger
    assert b.reasons == [REASON_WORK_BUDGET]


def test_zero_amount_charge_is_a_successful_noop():
    b = SoftWorkBudget(max_cells=0, max_csx_calls=1)
    assert b.charge_cells(0, "a")
    assert b.cells_processed == 0
    assert not b.exhausted


def test_cell_counts_ledger_tallies_per_source():
    b = SoftWorkBudget(max_cells=100, max_csx_calls=5)
    b.charge_cells(3, "ssx")
    b.charge_cells(4, "csx")
    b.charge_cells(2, "ssx")
    assert b.cell_counts == {"ssx": 5, "csx": 4}


def test_csx_call_ledger_is_separate_from_cells():
    b = SoftWorkBudget(max_cells=100, max_csx_calls=2)
    assert b.charge_csx_call()
    assert b.charge_csx_call()
    assert not b.charge_csx_call()
    assert b.exhausted
    assert b.cells_processed == 0
    assert b.reasons == [REASON_WORK_BUDGET]


def test_postprocess_latch_survives_search_exhaustion():
    b = SoftWorkBudget(max_cells=4, max_csx_calls=1)
    b.mark_exhausted()
    # Search phase is dead, but assembly still has its own allowance
    # (defaulting to max_cells).
    assert b.charge_postprocess(3)
    assert not b.charge_postprocess(3)
    assert b.postprocess_exhausted
    assert REASON_POSTPROCESS_CAP in b.reasons
    # And the latch fails fast afterwards without re-marking.
    reasons_after = list(b.reasons)
    assert not b.charge_postprocess(1)
    assert b.reasons == reasons_after


def test_postprocess_cap_defaults_to_max_cells():
    b = SoftWorkBudget(max_cells=7, max_csx_calls=1)
    assert b.max_postprocess_work == 7
    b2 = SoftWorkBudget(max_cells=7, max_csx_calls=1, max_postprocess_work=3)
    assert b2.max_postprocess_work == 3


def test_retire_reason_never_clears_hard_exhaustion():
    b = SoftWorkBudget(max_cells=1, max_csx_calls=1)
    b.mark_incomplete(REASON_OVERLAP_REGION)
    assert not b.result_fields()["complete"]
    b.retire_reason(REASON_OVERLAP_REGION)
    assert b.result_fields()["complete"]

    b.mark_incomplete(REASON_OVERLAP_REGION)
    b.mark_exhausted()  # hard exhaustion
    b.retire_reason(REASON_OVERLAP_REGION)
    fields = b.result_fields()
    assert not fields["complete"]
    assert fields["status"]["reasons"] == [REASON_WORK_BUDGET]


def test_complete_iff_no_reasons_through_mark_retire_cycle():
    b = SoftWorkBudget(max_cells=100, max_csx_calls=5)
    for step in range(3):
        fields = b.result_fields()
        assert fields["complete"] == (not fields["status"]["reasons"])
        b.mark_incomplete(REASON_MULTIPLICITY)
        fields = b.result_fields()
        assert fields["complete"] == (not fields["status"]["reasons"])
        b.retire_reason(REASON_MULTIPLICITY)


def test_output_cap_denies_and_marks_incomplete():
    b = SoftWorkBudget(max_cells=100, max_csx_calls=5, max_output_items=2)
    out = []
    assert b.append_output(out, "x", "point")
    assert b.append_output(out, "y", "point")
    assert not b.append_output(out, "z", "point")
    assert out == ["x", "y"]
    assert REASON_OUTPUT_CAP in b.reasons
    assert not b.result_fields()["complete"]


def test_extend_output_stops_at_first_denial():
    b = SoftWorkBudget(max_cells=100, max_csx_calls=5, max_output_items=2)
    out = []
    assert not b.extend_output(out, ["a", "b", "c"], "fragment")
    assert out == ["a", "b"]


def test_result_fields_schema_v2_shape():
    b = SoftWorkBudget(max_cells=10, max_csx_calls=3)
    b.charge_cells(2, "ssx")
    b.charge_csx_call()
    fields = b.result_fields()
    assert fields["complete"] is True
    work = fields["status"]["work"]
    assert work == {
        "cells_processed": 2,
        "csx_calls": 1,
        "max_cells": 10,
        "max_csx_calls": 3,
        "output_items": 0,
        "max_output_items": 1024,
        "postprocess_work": 0,
        "max_postprocess_work": 10,
        "cell_counts": {"ssx": 2},
    }


def test_remaining_properties_clamp_at_zero():
    b = SoftWorkBudget(max_cells=2, max_csx_calls=1)
    b.charge_cells(2, "a")
    assert b.remaining_cells == 0
    b.mark_exhausted()
    assert b.remaining_cells == 0
    assert b.remaining_postprocess_work == 2


def test_charge_hook_binds_source_and_none_path():
    assert charge_hook(None, "phi") is None
    b = SoftWorkBudget(max_cells=3, max_csx_calls=1)
    hook = charge_hook(b, "phi")
    assert hook(2)
    assert b.cell_counts == {"phi": 2}
    assert not hook(2)
    assert b.exhausted
    # default amount is 1, matching charge_cells
    b2 = SoftWorkBudget(max_cells=3, max_csx_calls=1)
    hook2 = charge_hook(b2, "singular")
    assert hook2()
    assert b2.cells_processed == 1


def test_reason_vocabulary_is_stable():
    # The budget-contract gate scans REASON_* names; the vocabulary is part
    # of the public schema and must not drift silently.
    #
    # This used to assert only the individual names, which cannot detect the
    # thing it exists to detect: a NEW reason drifted in unnoticed twice
    # (`unresolved_singular_set` at L52 slice 9, `trace_point_cap` at P2
    # 2026-07-25).  Pin the closed SET so adding a reason forces a
    # deliberate update here — and, by the checklist below, everywhere else
    # a reason has to be registered.
    import mmcore.numeric._work_budget as wb

    expected = {
        "work_budget", "output_cap", "postprocess_cap", "depth_limit",
        "parameter_fiber", "overlap_region_unsupported",
        "unresolved_tangential_zone", "unresolved_multiplicity",
        "trace_unverified", "trace_point_cap", "unresolved_singular_set",
    }
    actual = {getattr(wb, n) for n in dir(wb) if n.startswith("REASON_")}
    assert actual == expected, (
        "reason vocabulary changed. A new reason must ALSO be added to: "
        "examples/ssx/nurbs_ssx5_coverage_check.py STRUCTURAL_REASONS (if "
        "structural), the bez_ssx public docstring reason list, and the "
        "schema-v2 listing in "
        "docs/superpowers/plans/2026-07-12-ssx5-budget-review-and-overlap-contract.md"
        f"\nadded={actual - expected} removed={expected - actual}")

    # Individual pins kept: the SET check catches additions, these catch a
    # value being edited in place.
    assert REASON_WORK_BUDGET == "work_budget"
    assert REASON_OUTPUT_CAP == "output_cap"
    assert REASON_POSTPROCESS_CAP == "postprocess_cap"
    assert REASON_DEPTH_LIMIT == "depth_limit"
    assert REASON_PARAMETER_FIBER == "parameter_fiber"
    assert REASON_OVERLAP_REGION == "overlap_region_unsupported"
    assert REASON_TANGENTIAL_ZONE == "unresolved_tangential_zone"
    assert REASON_MULTIPLICITY == "unresolved_multiplicity"
    assert REASON_TRACE_UNVERIFIED == "trace_unverified"


def test_structural_reasons_set_covers_every_structural_reason():
    """The harness's structural/resource split must not miss a reason.

    `trace_point_cap` was added to the engine and to `_work_budget`'s
    documented structural family, but not to the coverage harness's closed
    set — so the tier it names was still counted as a resource FAIL, which
    is the exact misclassification P2 set out to remove.
    """
    import importlib.util
    import pathlib

    path = (pathlib.Path(__file__).parent.parent
            / "examples" / "ssx" / "nurbs_ssx5_coverage_check.py")
    src = path.read_text()
    # Cheap textual check: importing the harness pulls heavy fixtures.
    for reason in ("trace_point_cap", "unresolved_singular_set",
                   "trace_unverified"):
        assert f"'{reason}'" in src.split("STRUCTURAL_REASONS")[1][:400], reason


def test_bernstein_zero_budget_nodes_are_check_then_charge():
    from mmcore.numeric._work_budget import BernsteinZeroBudget

    b = BernsteinZeroBudget(max_nodes=2, max_results=10)
    assert b.enter() and b.enter()
    assert b.nodes == 2 and b.active_depth == 2
    assert not b.enter()          # denied, nothing spent
    assert b.nodes == 2
    assert b.exhausted
    b.leave()
    b.leave()
    assert b.active_depth == 0


def test_bernstein_zero_budget_results_charge_at_completion_after_clamp():
    from mmcore.numeric._work_budget import BernsteinZeroBudget

    b = BernsteinZeroBudget(max_nodes=10, max_results=3)
    kept = b.cap_top_level_results([1.0, 2.0])
    assert kept == [1.0, 2.0]
    assert b.results == 2 and not b.exhausted
    # Overproduction truncates silently and charges only what was kept.
    kept = b.cap_top_level_results([3.0, 4.0, 5.0])
    assert kept == [3.0]
    assert b.results == 3
    assert b.exhausted
    assert b.remaining_results() == 0


def test_bernstein_zero_budget_scope_sets_and_resets_contextvar():
    from mmcore.numeric._work_budget import (
        bernstein_zero_budget, _ZERO_BUDGET)

    assert _ZERO_BUDGET.get() is None
    with bernstein_zero_budget(5, 5) as outer:
        assert _ZERO_BUDGET.get() is outer
        with bernstein_zero_budget(1, 1) as inner:
            assert _ZERO_BUDGET.get() is inner
        assert _ZERO_BUDGET.get() is outer
    assert _ZERO_BUDGET.get() is None


def test_latching_spend_is_all_or_nothing_and_latches():
    from mmcore.numeric._work_budget import LatchingSpend

    led = LatchingSpend(max_work=5)
    assert led.spend(3)
    assert not led.spend(3)          # would overflow: denied, latched
    assert led.work_processed == 3
    assert led.exhausted
    assert not led.spend(1)          # latched: fails fast
    assert not led.external_exhausted


def test_latching_spend_zero_amount_is_a_noop():
    from mmcore.numeric._work_budget import LatchingSpend

    led = LatchingSpend(max_work=0)
    assert led.spend(0)
    assert led.work_processed == 0 and not led.exhausted


def test_latching_spend_external_hook_charged_after_local_check():
    from mmcore.numeric._work_budget import LatchingSpend

    calls = []

    def hook(n):
        calls.append(n)
        return len(calls) < 2   # second external charge denied

    led = LatchingSpend(max_work=100, charge_external=hook)
    assert led.spend(4)
    assert calls == [4]
    assert not led.spend(5)          # external denial
    assert led.work_processed == 4   # local spend NOT recorded on denial
    assert led.exhausted and led.external_exhausted
    # local denial must never consult the external hook
    led2 = LatchingSpend(max_work=2, charge_external=hook)
    calls.clear()
    assert not led2.spend(3)
    assert calls == []


def test_down_counter_pairs_remaining_and_processed():
    from mmcore.numeric._work_budget import DownCounter

    c = DownCounter(10)
    c.spend(3)
    c.spend(4)
    assert c.remaining == 3 and c.processed == 7
    # spend is deliberately unchecked (the bez_ccx/bez_csx family keeps
    # denial policy at each site); overdraw drives remaining negative and
    # the site's own guard reacts.
    c.spend(5)
    assert c.remaining == -2 and c.processed == 12


def test_down_counter_clamps_construction_and_tiers():
    from mmcore.numeric._work_budget import DownCounter

    c = DownCounter(-5)
    assert c.remaining == 0 and c.processed == 0
    c2 = DownCounter(10_000)
    assert c2.tier(2_000) == 2_000       # bounded sub-allowance
    c2.spend(9_500)
    assert c2.tier(2_000) == 500         # drawn from the remainder
    c2.spend(600)
    assert c2.tier(2_000) == 0           # never negative


def test_reconcile_reported_bills_at_least_the_floor():
    from mmcore.numeric._work_budget import reconcile_reported

    # A fast-rejected span reporting zero work still costs one unit — an
    # arbitrarily large candidate set must not bypass an aggregate
    # allowance.
    assert reconcile_reported(0, 100) == (1, False)
    assert reconcile_reported(7, 100) == (7, False)


def test_reconcile_reported_clamps_overruns_instead_of_denying():
    from mmcore.numeric._work_budget import reconcile_reported

    # The work already happened; the ledger only reconciles it.
    assert reconcile_reported(150, 100) == (100, True)
    assert reconcile_reported(1, 0) == (0, True)
    # floor=0 disables the minimum charge (for ledgers without the
    # anti-starvation rule).
    assert reconcile_reported(0, 100, floor=0) == (0, False)


def test_bern_zero_1d_reexports_are_the_same_objects():
    # ccx/csx and the tests import these via _bern_zero_1d; the move must
    # preserve identity (the solver reads the SAME ContextVar object).
    from mmcore.numeric import _bern_zero_1d as bz
    from mmcore.numeric import _work_budget as wb

    assert bz.BernsteinZeroBudget is wb.BernsteinZeroBudget
    assert bz.bernstein_zero_budget is wb.bernstein_zero_budget
    assert bz._ZERO_BUDGET is wb._ZERO_BUDGET


def test_bez_ssx5_reexports_are_the_same_objects():
    # Behavior-preservation contract: existing consumers access these via
    # the _bez_ssx5 module namespace (tests, budget-contract gate).
    from mmcore.numeric.intersection.ssx import _bez_ssx5 as ssx5
    from mmcore.numeric import _work_budget as wb

    assert ssx5._SSXSoftBudget is wb.SoftWorkBudget
    for name in dir(wb):
        if name.startswith("REASON_"):
            assert getattr(ssx5, name) == getattr(wb, name)
