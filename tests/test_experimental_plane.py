from unittest import TestCase

import numpy as np


class TestPlaneCase(TestCase):
    def setUp(self):
        self.PXYZ = Pln(lambda t: t, lambda t: np.array([1., 0., 0.]), lambda t: np.array([0., 1., 0.]),
                        lambda t: np.array([0., 0., 1.])
                        )
        self.origs = np.array([[1, 2, 3], [4, 1, 2]])


def tst():

    origs = np.array([[1, 2, 3], [4, 1, 2]])
    params = np.array([[5, 1, 2], [4, 1, 2]])

    return (np.allclose(params[0], P.local(origs[0], P(origs[0], params[0]))),
            np.allclose(params, P.local(origs, P(origs, params))))
