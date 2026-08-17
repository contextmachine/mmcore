"""Layering gate for mmcore (docs/RESTRUCTURE.md section 5).

Layers:  L0 geom/ + numeric/ substrate (+ implicit/ after Step 13)
         L1 numeric/intersection/ solver families
         L2 construction/ topo/ compat/
         L3 extras/ (leaf integrations)

Rule: ``layer(importer) >= layer(imported)`` for every import edge inside
mmcore/.  Imports under ``if __name__ == "__main__":`` are script demos,
not library edges, and are exempt.

Exit 0: no upward edge.  Exit 1: violations listed.
Run from the repo root: ``python tools/check_layering.py``
"""
from __future__ import annotations

import ast
import subprocess
import sys


def layer_of(module: str) -> int | None:
    """Layer of a dotted mmcore module path, or None if not mmcore."""
    if module == "mmcore" or module.startswith("mmcore."):
        parts = module.split(".")
        sub = parts[1] if len(parts) > 1 else ""
        if sub == "extras":
            return 3
        if sub in ("construction", "topo", "compat"):
            return 2
        if sub == "numeric":
            return 1 if (len(parts) > 2 and parts[2] == "intersection") else 0
        # geom, implicit, root, anything else at the package base
        return 0
    return None


def module_name(path: str) -> str:
    mod = path[:-3].replace("/", ".")
    return mod[:-9] if mod.endswith(".__init__") else mod


def resolve_relative(importer: str, is_pkg: bool, node: ast.ImportFrom) -> str | None:
    base = importer.split(".")
    # level=1 in a package __init__ refers to the package itself
    drop = node.level - (1 if is_pkg else 0)
    if drop > 0:
        base = base[:-drop]
    if not base:
        return None
    return ".".join(base + ([node.module] if node.module else []))


def edges_of(path: str):
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src, filename=path)
    importer = module_name(path)
    is_pkg = path.endswith("__init__.py")

    def is_main_guard(node: ast.stmt) -> bool:
        return (isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id == "__name__")

    def walk(nodes):
        for n in nodes:
            if is_main_guard(n):
                continue
            if isinstance(n, ast.Import):
                for a in n.names:
                    yield a.name, n.lineno
            elif isinstance(n, ast.ImportFrom):
                if n.level:
                    m = resolve_relative(importer, is_pkg, n)
                    if m:
                        yield m, n.lineno
                elif n.module:
                    yield n.module, n.lineno
            else:
                yield from walk(ast.iter_child_nodes(n))

    return importer, list(walk(tree.body))


def main() -> int:
    files = subprocess.run(["git", "ls-files", "mmcore"],
                           capture_output=True, text=True, check=True).stdout.split()
    violations = []
    for f in files:
        if not f.endswith(".py"):
            continue
        importer, edges = edges_of(f)
        li = layer_of(importer)
        for imported, lineno in edges:
            lt = layer_of(imported)
            if lt is not None and li is not None and li < lt:
                violations.append((f, lineno, importer, li, imported, lt))

    if violations:
        print(f"{len(violations)} layering violation(s):")
        for f, ln, importer, li, imported, lt in sorted(violations):
            print(f"  {f}:{ln}: L{li} {importer} -> L{lt} {imported}")
        return 1
    print("layering OK: no upward import edges in mmcore/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
