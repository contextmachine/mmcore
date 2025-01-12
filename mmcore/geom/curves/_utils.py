import math


def fround(val: float, tol: float = 0.001):
    return round(val, int(abs(math.log10(tol))))
