"""Exact rational univariate polynomial and Sturm helpers."""
from fractions import Fraction
from math import comb


def coefficient_build_work(control_count):
    """Vector coefficient updates in a Bernstein-to-power basis change."""
    n = int(control_count)
    return n*(n+1)//2


def isolate_root_intervals(polynomial, tick):
    """Square-free Sturm isolation of every distinct root in closed [0,1].

    Coefficients are ascending exact rational powers. Return
    (squarefree, repeated_factor, sturm_sequence, root_intervals).
    A rational endpoint root is represented by a degenerate interval;
    nondegenerate intervals contain exactly one root in their interior.
    The identically zero polynomial is a dimension change, not isolated
    roots, and is rejected. ``tick`` owns the caller's shared work budget.
    """
    polynomial = _trim(list(polynomial))
    if not any(polynomial):
        raise ValueError('the zero polynomial has a positive-dimensional root set')
    tick()
    repeated = _gcd(polynomial, _derivative(polynomial), tick)
    squarefree, _ = _divide(polynomial, repeated)
    if len(squarefree) == 1:
        return squarefree, repeated, [], []
    sequence = _sturm(squarefree, tick)
    zero, one = Fraction(0), Fraction(1)
    intervals = [(t, t) for t in (zero, one) if _value(squarefree, t) == 0]
    pending = [(zero, one)]
    if len(squarefree) == 2:
        root = -squarefree[0]/squarefree[1]
        intervals = [(root, root)] if zero <= root <= one else []
        pending = []
    while pending:
        lo, hi = pending.pop()
        count = _open_count(squarefree, sequence, lo, hi)
        if count == 0:
            continue
        if count == 1:
            intervals.append((lo, hi))
            continue
        tick()
        mid = (lo+hi)/2
        if _value(squarefree, mid) == 0:
            intervals.append((mid, mid))
        pending.extend(((lo, mid), (mid, hi)))
    return squarefree, repeated, sequence, sorted(intervals)

def _trim(p):
    while len(p) > 1 and not p[-1]:
        p.pop()
    return p


def _value(p, t):
    result = Fraction(0)
    for coefficient in reversed(p):
        result = result*t + coefficient
    return result


def _derivative(p):
    return [i*p[i] for i in range(1, len(p))] or [Fraction(0)]


def _divide(a, b):
    remainder = a.copy()
    quotient = [Fraction(0)]*max(1, len(a)-len(b)+1)
    while len(remainder) >= len(b) and any(remainder):
        index = len(remainder)-len(b)
        factor = remainder[-1]/b[-1]
        quotient[index] = factor
        for i, coefficient in enumerate(b):
            remainder[index+i] -= factor*coefficient
        _trim(remainder)
    return _trim(quotient), remainder


def _gcd(a, b, tick):
    while any(b):
        tick()
        _, remainder = _divide(a, b)
        a, b = b, remainder
        if any(b):
            b = [coefficient/abs(b[-1]) for coefficient in b]
    return [coefficient/a[-1] for coefficient in a]


def _sturm(p, tick):
    sequence = [p, _derivative(p)]
    while any(sequence[-1]):
        tick()
        _, remainder = _divide(sequence[-2], sequence[-1])
        if not any(remainder):
            break
        sequence.append([-coefficient/abs(remainder[-1]) for coefficient in remainder])
    return sequence if any(sequence[-1]) else sequence[:-1]


def _variations(sequence, t):
    signs = [1 if value > 0 else -1 for p in sequence if (value := _value(p, t))]
    return sum(a != b for a, b in zip(signs, signs[1:]))


def _open_count(p, sequence, lo, hi):
    return _variations(sequence, lo)-_variations(sequence, hi)-int(_value(p, hi) == 0)


def _power(bernstein):
    degree = len(bernstein)-1
    result = [Fraction(0)]*(degree+1)
    for i, value in enumerate(bernstein):
        for j in range(i, degree+1):
            result[j] += value*comb(degree, i)*comb(degree-i, j-i)*(-1)**(j-i)
    return _trim(result)


def _range(p, lo, hi):
    low = high = Fraction(0)
    for coefficient in reversed(p):
        products = (low*lo, low*hi, high*lo, high*hi)
        low, high = min(products)+coefficient, max(products)+coefficient
    return low, high


def _refine(p, sequence, interval, tick):
    lo, hi = interval
    if lo == hi:
        return interval
    tick()
    mid = (lo+hi)/2
    if _value(p, mid) == 0:
        return mid, mid
    if _open_count(p, sequence, lo, mid):
        return lo, mid
    return mid, hi


def _sign_at_root(p, sequence, interval, h, tick):
    lo, hi = interval
    if lo == hi:
        value = _value(h, lo)
        return (value > 0)-(value < 0), interval
    common = _gcd(p, h, tick)
    if len(common) > 1:
        common_sequence = _sturm(common, tick)
        if _open_count(common, common_sequence, lo, hi):
            return 0, interval
    while True:
        lower, upper = _range(h, *interval)
        if lower > 0:
            return 1, interval
        if upper < 0:
            return -1, interval
        interval = _refine(p, sequence, interval, tick)
        if interval[0] == interval[1]:
            value = _value(h, interval[0])
            return (value > 0)-(value < 0), interval
