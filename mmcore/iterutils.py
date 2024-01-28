from itertools import tee, cycle


def rotate(iterable, n=1):
    it = cycle(iterable)
    for _ in range(len(iterable) + n if n < 0 else n):
        next(it)
    for _ in iterable:
        yield next(it)


def shift_circular(iterable, n=2):
    clit = tee(cycle(iterable), n)
    for i in range(n):
        for _ in range(i):
            clit[i].__next__()
    it = zip(*clit)
    for _ in iterable:
        yield next(it)
