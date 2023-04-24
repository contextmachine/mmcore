from mmcore.base.geom import LineObject
from mmcore.collections import DCLL
import numpy as np

class Shape(LineObject, DCLL):
    @property
    def points(self):
        return np.array(list(self))