"""Import-health gate for mmcore.

Imports every tracked module under mmcore/ and asserts the failure set is a
SUBSET of the benign optional-dependency allowlist below.  Subset, never
equality: a run with extras installed (plotly, torch, ...) legitimately fixes
allowlisted leaves, and that must not fail the gate.

Exit 0: every failure (if any) is allowlisted.
Exit 1: at least one non-allowlisted module fails to import.

Run from the repo root: ``python tools/check_imports.py``
(baseline command of docs/RESTRUCTURE.md section 7, made a tracked file).
"""
from __future__ import annotations

import importlib
import io
import subprocess
import sys
import warnings

# The 6 benign optional-dependency leaves (docs/RESTRUCTURE.md section 1).
# These fail with ModuleNotFoundError on their optional import when the
# extra is not installed; that is correct behaviour, not debt.
ALLOWED_FAILURES = {
    "mmcore.extras.occ.geom_int",            # OCC (not pip-installable)
    "mmcore.extras.renderer.renderer2d",     # plotly
    "mmcore.extras.renderer.renderer3d",     # glfw
    "mmcore.extras.rhino",                   # rhino3dm
    "mmcore.extras.torch.algorithms.implicit_point",  # torch
    "mmcore.extras.torch.numeric",           # torch
}


def tracked_modules() -> list[str]:
    files = subprocess.run(
        ["git", "ls-files", "mmcore"], capture_output=True, text=True, check=True
    ).stdout.split()
    mods = []
    for f in files:
        if f.endswith(".py"):
            m = f[:-3].replace("/", ".")
            mods.append(m[:-9] if m.endswith(".__init__") else m)
    return sorted(set(mods))


def main() -> int:
    warnings.filterwarnings("ignore")
    failures: dict[str, str] = {}
    for m in tracked_modules():
        out, err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = io.StringIO()
        try:
            importlib.import_module(m)
        except BaseException as ex:  # noqa: BLE001 - import-time errors of any kind
            failures[m] = f"{type(ex).__name__}: {ex}"
        finally:
            sys.stdout, sys.stderr = out, err

    unexpected = {m: e for m, e in failures.items() if m not in ALLOWED_FAILURES}
    allowed = sorted(set(failures) & ALLOWED_FAILURES)

    print(f"import failures: {len(failures)} "
          f"({len(allowed)} allowlisted, {len(unexpected)} unexpected)")
    for m in allowed:
        print(f"  [allowed]    {m} -> {failures[m]}")
    for m in sorted(unexpected):
        print(f"  [UNEXPECTED] {m} -> {unexpected[m]}")

    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main())
