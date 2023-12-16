from enum import Enum

import numpy as np

from mmcore.func import vectorize


@vectorize(signature='(),(),()->(i,i)')
def ypr_matrix(t, w, p):
    Y = np.array([[np.cos(t), -np.sin(t), 0.], [np.sin(t), np.cos(t), 0.], [0., 0., 1.]])

    P = np.array([[np.cos(w), 0., np.sin(w)], [0., 1., 0.], [-np.sin(w), 0., np.cos(w)]])
    W = np.array([[1., 0., 0.], [0., np.cos(p), -np.sin(p)], [0., np.sin(p), np.cos(p)]])

    return Y.dot(P).dot(W)

class YPR:
    __slots__ = ['ypw']

    def __init__(self, yaw=(0., 2 * np.pi), pitch=(0., 0), roll=(0., 0)):
        self.ypw = yaw, pitch, roll

    @vectorize(excluded=[0], signature='(),(),()->(i,i)')
    def rotation_matrix(self, t, w, p):
        Y = np.array([
            [np.cos(t), -np.sin(t), 0.],
            [np.sin(t), np.cos(t), 0.],
            [0., 0., 1.]
        ])

        P = np.array(
            [
                [np.cos(w), 0., np.sin(w)],
                [0., 1., 0.],
                [-np.sin(w), 0., np.cos(w)]
            ])
        W = np.array([
            [1., 0., 0.],
            [0., np.cos(p), -np.sin(p)],
            [0., np.sin(p), np.cos(p)]
        ])

        return Y.dot(P).dot(W)

    @vectorize(excluded=[0], signature='(i),()->(j,i)')
    def rotate(self, pts, num):
        a, b, c = self.ypw
        return self.rotation_matrix(
            np.linspace(*a, num),
            np.linspace(*b, num),
            np.linspace(*c, num)).dot(np.array(pts).T)

    def __call__(self, points: np.ndarray, num=4):
        res = self.rotate(points, num)

        # print(res.shape)
        return res


Z90 = ypr_matrix(np.pi / 2, 0., 0.)
X90 = ypr_matrix(0., 0., np.pi / 2)
Y90 = ypr_matrix(0., np.pi / 2, 0.)
X90_Y90 = ypr_matrix(0., np.pi / 2, np.pi / 2)
