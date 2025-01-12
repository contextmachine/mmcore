
import numpy as np

from mmcore.numeric.intersection.implicit_implicit import ImplicitIntersectionCurve, iterate_curves

from mmcore.geom.primitives import Tube


x, y, v, u, z = [[[12.359112840551504, -7.5948049557495425, 0.0], [2.656625109045951, 1.2155741170561933, 0.0]],
                 [[7.14384241216015, -6.934735074711716, -0.1073366304415263],
                  [7.0788761013028365, 10.016931402130641, 0.8727530304189204]],
                 [[8.072688942425103, -2.3061831591019826, 0.2615779273274319],
                  [7.173685617288537, -3.4427234423361512, 0.4324928834164773],
                  [7.683972288682133, -2.74630545102506, 0.07413871667321925],
                  [7.088944240699163, -4.61458155002528, -0.22460509818398067],
                  [7.304629277158477, -3.9462033818505433, 0.8955725109783643],
                  [7.304629277158477, -3.3362864951018985, 0.8955725109783643],
                  [7.304629277158477, -2.477065729786164, 0.7989970582016114],
                  [7.304629277158477, -2.0988672326949933, 0.7989970582016114]], 0.72648, 1.0]

aa = np.array(x)
bb = np.array(y)

t1 = Tube(aa[0],  aa[1],z,thickness=0.2)
t2 = Tube( bb[0],  bb[1] ,u,thickness=0.2)
vv = np.array(v)

import time




crv = ImplicitIntersectionCurve(t1, t2)
crv.build_tree()
s = time.time()
res = []
for item in iterate_curves(crv):
    res.append(item)


print(time.time() - s)

print(len(res))
