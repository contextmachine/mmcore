#  Copyright (c) 2022. Computational Geometry, Digital Engineering and Optimizing your construction processe"

from functools import wraps

import compas.geometry as cg
from mmcore.geom.parametric.algorithms import *
from mmcore.geom.parametric.sketch import *


def to_cmp_point(func):
    @wraps(func)
    def wrp(*a, **kw):
        return cg.Point(*func(*a, **kw))

    return wrp


