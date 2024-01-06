import numpy as np

from mmcore.func import vectorize
from mmcore.numeric import cartesian_product


@vectorize(signature='(i),(i)->(j,i)')
def box_from_intervals(start, end):
    return cartesian_product(*(np.dstack((start, end))[0]))
