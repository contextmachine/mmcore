# SSX branch preservation audit and repair

The requested outcome is reliable Bézier/NURBS intersection with principled branch discovery and an explanation of excessive CSX work. The baseline is commit `76735f9`, preserved in the main checkout; implementation uses `codex/ssx-completeness`. The main checkout's local `bern.py` edit is outside this work.

## Design

Keep the public result schema and shared finite budgets. Audit every search termination against a geometric exclusion, a complete local topology certificate, or a returned unresolved parameter region. Discovering one component does not certify its complement. Never approximate rational representation metadata with geometric tolerances. Topology operations must consider paired surface parameters as well as world geometry.

Separate three concerns: algebraic zero-set discovery, numerical curve approximation, and output assembly. Sampling/proximity checks can validate approximation but cannot independently prove absence of another component. Do not tune constants to fixtures or describe passing cases as a universal theorem.

Use the existing subdivision framework after repairing invalid decisions. Replacing the entire solver would discard substantial singular/overlap functionality before proving a replacement. Raising work caps would conceal repeated work and does not repair correctness. Any optimization must preserve exhaustive candidate/root accounting.

## Verification

Construct polynomial graph-vs-plane families with prescribed line/circle factors and rational surfaces with analytically derived zeros. Check every known component against entire polyline segments, residuals on both surfaces, topology, and representation invariance (surface swap, parameter reversal, knot insertion, weights). Use independent code review of exclusion assumptions alongside these checks. Run existing SSX, CSX, and CCX suites with bounded subprocess execution and retain before/after profiles.

## Limits of claims

A finite floating-point solver is not a proof procedure for arbitrary singular algebraic geometry. A universal completeness claim requires validated coefficient enclosures, certified root isolation and continuation, and certified singular-set topology. Record which obligations are established and which remain instead of substituting a larger regression suite for a proof. An unresolved result must preserve verified output and expose its unresolved complement.
