#  Copyright (c) 2022. Computational Geometry, Digital Engineering and Optimizing your construction processe"
import copy

import numpy as np

from . import PrmGenerator, ParametricType


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
