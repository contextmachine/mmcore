import unittest
from scipy.optimize import rosen,rosen_der
import numpy as np
from mmcore.numeric.fdm import Grad

def cassini_func(xy, a=1.1, c=1.0):
    x, y = xy
    return (x * x + y * y) * (x * x + y * y) - 2 * c * c * (x * x - y * y) - (a * a * a * a - c * c * c * c)


def cassini_grad(xy, c=1.0):
    x, y = xy
    return (float(4.0 * x * (x * x + y * y) - 4.0 * c * c * x), float(4.0 * y * (x * x + y * y) + 4.0 * c * c * y))


def testf(fun1, fun2, t):
    t = np.array(t)
    r1, r2 = fun1(t), fun2(t)
    if not np.allclose(r1, r2, rtol=1e-05, atol=1e-05):
        print(r1, r2, False)
        return False

    return True


class MyTestCase(unittest.TestCase):
    def setUp(self):

        self.cassini_grad = Grad(cassini_func
                                    )
        self.rosen_grad = Grad(rosen)


    def test_cassini_pde_grad(self):
        self.assertTrue(
            all([testf(self.cassini_grad, cassini_grad,
                       np.random.random(2) * np.random.randint(-100, 100)) for i in range(100)]))

    def test_rosen_pde_grad(self):
        self.assertTrue(
            all([testf(self.rosen_grad, rosen_der,
                       np.random.random(np.random.randint(3,9))) for i in range(100)])
        )

if __name__ == '__main__':
    unittest.main()
