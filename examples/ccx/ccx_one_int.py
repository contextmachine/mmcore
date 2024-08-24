import numpy as np

from mmcore.geom.curves import NURBSpline
from mmcore.numeric.intersection.ccx import ccx


aa, bb = NURBSpline(
    np.array(
        [
            (-13.654958030023677, -19.907874497194975, 0.0),
            (3.7576433265207765, -39.948793039632903, 0.0),
            (16.324284871574083, -18.018771519834026, 0.0),
            (44.907234268165922, -38.223959886390297, 0.0),
            (49.260384607302036, -13.419216444520401, 0.0),
        ]
    )
), NURBSpline(
    np.array(
        [
            (40.964758489325661, -3.8915666456564679, 0.0),
            (-9.5482124270650726, -28.039230791052990, 0.0),
            (4.1683178868166371, -58.264878428828240, 0.0),
            (37.268687446662931, -58.100608604709883, 0.0),
        ]
    )
)



res = ccx(aa, bb, 0.001)

print(res) # [(0.600738525390625, 0.371673583984375)]
