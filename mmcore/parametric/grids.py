#  Copyright (c) 2022. Computational Geometry, Digital Engineering and Optimizing your construction processe"
import copy

import numpy as np

from . import PrmGenerator, ParametricType
from mm.geom import Triangle
from mm.meta import MetaItem




class Grid(ParametricType):
    def evaluate(self, t):
        pass

    start = ()
    stop = ()
    step = ()
    item = None

    def __getitem__(self, item: slice):

        if isinstance(item, tuple):

            return self.evaluate(item)

        elif isinstance(item, slice):
            slf = copy.deepcopy(self)
            slf.__call__(start=item.start, stop=item.stop, step=item.step)

            return slf


class TriangleGrid(Grid):
    start = (0, 0, 0)
    stop = (1, 1, 1)
    step = (0.5, 0.5, 0.5)
    item = Triangle

    def evaluate(self, t):
        np.asarray(self.start)

        (np.eye(3) @ np.asarray(t) + np.eye(3) @ np.asarray(self.step)).T
        return Triangle((np.eye(3) @ np.asarray(t) + np.eye(3) @ np.asarray(self.step)).T)

    def __repr__(self):
        return "TriangleGrid item"
