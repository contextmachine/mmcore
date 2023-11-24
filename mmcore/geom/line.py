import numpy as np


def evaluate_line(start, end, t):
    return (end - start) * t + start


evaluate_line = np.vectorize(evaluate_line, signature='(i),(i),()->(i)')
