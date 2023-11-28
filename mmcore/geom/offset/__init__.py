import numpy as np

from mmcore.func import vectorize
from mmcore.geom.parametric import polygon_variable_offset


@vectorize(excluded=[0], signature='(k,i)->(i,j)')
def offset(points, dists):
    return np.array(list(polygon_variable_offset(points, dists.T)))
