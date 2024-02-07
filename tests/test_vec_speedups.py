import time
from unittest import TestCase
from mmcore.geom.vec.vec_speedups import dot as sdot
from mmcore.geom.vec.vec_speedups import norm as snorm
from mmcore.geom.vec.vec_speedups import unit as sunit
from mmcore.geom.vec import dot, norm, unit

import numpy as np


def test_vec_dot_speedups(count=1_000_000, components=3):
    print('dot')
    a = np.random.random((count, components))
    b = np.random.random((count, components))
    s = time.time()
    res1 = sdot(a, b)
    t1 = time.time() - s
    print('speedups', divmod(t1, 60))
    s = time.time()
    res2 = dot(a, b)
    t2 = time.time() - s
    print('python', divmod(t2, 60))

    print(f'speedups {t2 / t1} x faster on {count} items')
    return np.allclose(res1, res2)


def test_vec_unit_speedups(count=1_000_000, components=3):
    print('unit')
    a = np.random.random((count, components))

    s = time.time()
    res1 = sunit(a)
    t1 = time.time() - s
    print('speedups', divmod(t1, 60))
    s = time.time()
    res2 = unit(a)
    t2 = time.time() - s
    print('python', divmod(t2, 60))
    print(f'speedups {t2 / t1} x faster on {count} items')
    return np.allclose(res1, res2)


def test_vec_norm_speedups(count=1_000_000, components=3):
    print('norm')
    a = np.random.random((count, components))
    s = time.time()
    res1 = snorm(a)
    t1 = time.time() - s
    print('speedups', divmod(t1, 60))
    s = time.time()
    res2 = norm(a)
    t2 = time.time() - s
    print('python', divmod(t2, 60))
    print(f'speedups {t2 / t1} x faster on {count} items')
    return np.allclose(res1, res2)


class TestVecSpeedups(TestCase):
    def setUp(self):
        self.count = 3_000_000
        self.n_components = 3

    def test_dot(self):
        res = test_vec_dot_speedups(self.count, self.n_components)
        self.assertTrue(res)

    def test_norm(self):
        res = test_vec_norm_speedups(self.count, self.n_components)
        self.assertTrue(res)

    def test_unit(self):
        res = test_vec_unit_speedups(self.count, self.n_components)
        self.assertTrue(res)
