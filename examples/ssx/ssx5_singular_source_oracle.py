"""Audit a singular fixture's exact binary height polynomial.

Run from the repository root, with the optional SymPy audit dependency:
    python examples/ssx/ssx5_singular_source_oracle.py _mexican_hat
    python examples/ssx/ssx5_singular_source_oracle.py _touch_plus_loop '[0.04]'

This independent oracle does not run SSX. For height graphs against z=0,
the unit Groebner basis of (P, P_s, P_t) proves that the supplied binary
control net has no singular zero over the complex numbers, hence no real
tangent contact. A non-unit basis by itself makes no existence claim.
"""
import argparse
import json
from math import comb
from pathlib import Path
import runpy


def audit(name, arguments):
    import sympy as sp

    root = Path(__file__).resolve().parents[2]
    fixtures = runpy.run_path(str(root/'tests/test_bez_ssx5_singular.py'))
    if not name.startswith('_') or name not in fixtures:
        raise ValueError('Expected a named surface-pair fixture helper')
    first, _ = fixtures[name](*arguments)
    if first.shape[-1] != 3:
        raise ValueError('This oracle expects a Cartesian height graph')
    s, t = sp.symbols('s t')
    m, n = first.shape[0]-1, first.shape[1]-1
    def component(axis):
        return sp.Poly(sum(
            sp.Rational(float(first[i, j, axis]))*comb(m, i)*s**i*(1-s)**(m-i)
            *comb(n, j)*t**j*(1-t)**(n-j)
            for i in range(m+1) for j in range(n+1)), s, t)
    # A height-graph singularity has precisely these three scalar equations.
    # Reject unsupported chart shapes rather than interpreting a degenerate
    # chart's parameter derivatives as graph derivatives.
    if component(0).as_expr() != s or component(1).as_expr() != t:
        raise ValueError('Height-graph coordinates must be exactly x=s,y=t')
    p = component(2)
    basis = sp.groebner([p.as_expr(), p.diff(s).as_expr(), p.diff(t).as_expr()], s, t)
    return {'helper': name, 'arguments': arguments, 'sympy': sp.__version__,
            'height': str(p.as_expr()),
            'center_height': str(p.eval({s: sp.Rational(1, 2), t: sp.Rational(1, 2)})),
            'groebner': [str(item) for item in basis],
            'no_singular_complex_roots': list(basis) == [1]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('helper')
    parser.add_argument('arguments', nargs='?', default='[]')
    args = parser.parse_args()
    print(json.dumps(audit(args.helper, json.loads(args.arguments)), indent=2))
