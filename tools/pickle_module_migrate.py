"""Lossless module-path migration for pickle fixtures.

Rewrites the module paths embedded in a pickle (e.g.
``mmcore.nurbs._nurbs_eval`` -> ``mmcore.nurbs._nurbs_eval``) by
load-with-remap + re-dump.  No geometry is recomputed: the ssx engine is
never run.  Only the class-reference strings change; every value
(ints, floats, ndarray buffers, dtypes, list/tuple structure) is carried
through unchanged.

REAL-RENAME USAGE (run this AFTER the source rename lands, with the new
package importable):

    python pickle_module_migrate.py migrate \
        --map mmcore.nurbs._nurbs_eval=mmcore.nurbs._nurbs_eval \
        --out-dir /some/out/dir \
        examples/ssx/nurbs_nurbs_intersection_{5,6,8,10,11}.pkl

then verify each pair:

    python pickle_module_migrate.py verify OLD.pkl NEW.pkl \
        --map mmcore.nurbs._nurbs_eval=mmcore.nurbs._nurbs_eval

`verify` loads OLD through the remap and NEW natively and deep-compares
the two object trees (ndarray: shape+dtype+bit-exact buffer; NamedTuple:
type name + field-by-field; containers: recursively).

Why load+re-dump and not a byte patch: protocol-5 SHORT_BINUNICODE
carries a length prefix, so an in-place string substitution of a
different-length module path corrupts the stream.  Re-dumping is exact
for these fixtures because their payload is only
int / float / str / list / tuple / NamedTuple / np.ndarray.
"""
from __future__ import annotations

import argparse
import io
import pickle
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# the migration primitive
# ---------------------------------------------------------------------------

class RemappingUnpickler(pickle.Unpickler):
    """Unpickler that rewrites module paths before resolving a global.

    `module_map` maps an OLD module path to a NEW one.  A key also
    matches any submodule of itself (``a.b`` remaps ``a.b.c`` too), so a
    whole package rename is one entry.
    """

    def __init__(self, file, module_map: dict[str, str], record: list | None = None):
        super().__init__(file)
        # longest key first so the most specific rule wins
        self._rules = sorted(module_map.items(), key=lambda kv: -len(kv[0]))
        self._record = record

    def remap(self, module: str) -> str:
        for old, new in self._rules:
            if module == old:
                return new
            if module.startswith(old + "."):
                return new + module[len(old):]
        return module

    def find_class(self, module, name):
        new_module = self.remap(module)
        if self._record is not None:
            self._record.append((module, name, new_module))
        return super().find_class(new_module, name)


def load_remapped(path, module_map, record=None):
    with open(path, "rb") as f:
        return RemappingUnpickler(f, module_map, record).load()


def migrate_file(src, dst, module_map, protocol=5, record=None):
    """Load `src` with old->new module remapping, re-dump to `dst`.

    The classes reached through the remap live in the NEW module, so the
    re-dumped stream embeds the NEW path by construction.
    """
    obj = load_remapped(src, module_map, record)
    with open(dst, "wb") as f:
        pickle.dump(obj, f, protocol=protocol)
    return obj


# ---------------------------------------------------------------------------
# deep structural comparison
# ---------------------------------------------------------------------------

def deep_compare(a, b, path="root", diffs=None):
    """Return list of human-readable differences ([] == identical)."""
    if diffs is None:
        diffs = []

    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            diffs.append(f"{path}: ndarray vs {type(b).__name__}/{type(a).__name__}")
            return diffs
        if a.dtype != b.dtype:
            diffs.append(f"{path}: dtype {a.dtype} != {b.dtype}")
        if a.shape != b.shape:
            diffs.append(f"{path}: shape {a.shape} != {b.shape}")
            return diffs
        if not np.array_equal(a, b, equal_nan=True):
            diffs.append(f"{path}: array_equal False "
                         f"(max abs diff {np.nanmax(np.abs(a - b)) if a.size else 0})")
        # bit-exactness (catches -0.0 / NaN-payload drift array_equal hides)
        if a.dtype == b.dtype and a.shape == b.shape:
            if np.ascontiguousarray(a).tobytes() != np.ascontiguousarray(b).tobytes():
                diffs.append(f"{path}: byte buffers differ (values compare equal)")
        return diffs

    ta, tb = type(a), type(b)
    a_is_nt = isinstance(a, tuple) and hasattr(a, "_fields")
    b_is_nt = isinstance(b, tuple) and hasattr(b, "_fields")
    if a_is_nt or b_is_nt:
        if not (a_is_nt and b_is_nt):
            diffs.append(f"{path}: namedtuple-ness mismatch {ta.__name__}/{tb.__name__}")
            return diffs
        if ta.__name__ != tb.__name__:
            diffs.append(f"{path}: namedtuple class {ta.__name__} != {tb.__name__}")
        if a._fields != b._fields:
            diffs.append(f"{path}: fields {a._fields} != {b._fields}")
            return diffs
        for fld in a._fields:
            deep_compare(getattr(a, fld), getattr(b, fld), f"{path}.{fld}", diffs)
        return diffs

    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if ta is not tb:
            diffs.append(f"{path}: container type {ta.__name__} != {tb.__name__}")
        if len(a) != len(b):
            diffs.append(f"{path}: len {len(a)} != {len(b)}")
            return diffs
        for i, (x, y) in enumerate(zip(a, b)):
            deep_compare(x, y, f"{path}[{i}]", diffs)
        return diffs

    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            diffs.append(f"{path}: dict keys differ {set(a) ^ set(b)}")
            return diffs
        for k in a:
            deep_compare(a[k], b[k], f"{path}[{k!r}]", diffs)
        return diffs

    if ta is not tb:
        diffs.append(f"{path}: type {ta.__name__} != {tb.__name__}")
        return diffs
    # arbitrary objects (dataclasses, plain classes): compare state dicts
    if hasattr(a, "__dict__") and not isinstance(a, (int, float, str, bytes, bool, complex)):
        deep_compare(vars(a), vars(b), f"{path}.__dict__", diffs)
        return diffs
    if isinstance(a, float):
        # bit-exact float comparison
        import struct
        if struct.pack("<d", a) != struct.pack("<d", b):
            diffs.append(f"{path}: float {a!r} != {b!r}")
        return diffs
    if a != b:
        diffs.append(f"{path}: {a!r} != {b!r}")
    return diffs


def count_nodes(o):
    if isinstance(o, np.ndarray):
        return 1 + o.size
    if isinstance(o, tuple) and hasattr(o, "_fields"):
        return 1 + sum(count_nodes(getattr(o, f)) for f in o._fields)
    if isinstance(o, (list, tuple)):
        return 1 + sum(count_nodes(x) for x in o)
    if isinstance(o, dict):
        return 1 + sum(count_nodes(k) + count_nodes(v) for k, v in o.items())
    return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_map(pairs):
    m = {}
    for p in pairs:
        old, _, new = p.partition("=")
        if not old or not new:
            raise SystemExit(f"bad --map entry {p!r}; expected OLD=NEW")
        m[old] = new
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    mg = sub.add_parser("migrate")
    mg.add_argument("files", nargs="+")
    mg.add_argument("--map", action="append", required=True)
    mg.add_argument("--out-dir", required=True)
    mg.add_argument("--protocol", type=int, default=5)

    vf = sub.add_parser("verify")
    vf.add_argument("old")
    vf.add_argument("new")
    vf.add_argument("--map", action="append", default=[])

    args = ap.parse_args(argv)

    if args.cmd == "migrate":
        module_map = _parse_map(args.map)
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for f in args.files:
            src = Path(f)
            dst = out / src.name
            rec = []
            migrate_file(src, dst, module_map, args.protocol, rec)
            remapped = sorted({(o, n) for o, _, n in rec if o != n})
            print(f"{src} -> {dst}  ({src.stat().st_size} -> {dst.stat().st_size} bytes)")
            for o, n in remapped:
                print(f"    remapped module: {o}  ->  {n}")
        return 0

    module_map = _parse_map(args.map)
    a = load_remapped(args.old, module_map)
    with open(args.new, "rb") as f:
        b = pickle.load(f)
    diffs = deep_compare(a, b)
    print(f"nodes compared: {count_nodes(a)}")
    if diffs:
        print("DIFFERENCES:")
        for d in diffs[:50]:
            print("  " + d)
        return 1
    print("IDENTICAL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
