import numpy as np

from mmcore.geom.curves.bspline import NURBSpline

pts=np.array([[148.39347323205118, -60.435505127468502, 0.0], [148.39347323205118, -60.435505127468502, 312.99479852664467], [15.867952362691028, -60.435505127468502, 312.99479852664467], [15.867952362691028, 79.868450977375943, 312.99479852664467], [127.81520754187642, 79.868450977375943, 312.99479852664467], [127.81520754187642, 263.24739314874569, 312.99479852664467], [225.93283148956732, 263.24739314874569, 312.99479852664467], [225.93283148956732, 178.59891945435234, 312.99479852664467], [336.65283186482787, 178.59891945435234, 312.99479852664467], [336.65283186482787, 64.991782255356810, 312.99479852664467]]
)
nc = NURBSpline(pts)
pts0=nc(np.linspace(*nc.interval(),100)).tolist()
print(nc.interval(),nc.knots)
#nc.make_periodic()
#pts1=nc(np.linspace(*nc.interval(),100)).tolist()