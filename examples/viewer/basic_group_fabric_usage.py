# !ipython
import numpy as np

from mmcore.base.sharedstate import serve
from mmcore.geom.box import Box
from mmcore.geom.vec import unit

vecs = unit(np.random.random((2, 4, 3)))
boxes = [Box(10, 20, 10), Box(5, 5, 5), Box(15, 5, 5), Box(25, 20, 2)]
for i in range(4):
    boxes[i].xaxis = vecs[0, i, :]
    boxes[i].origin = vecs[1, i, :] * np.random.randint(0, 20)
    boxes[i].refine(('y', 'z'))

from mmcore.common.viewer import DefaultGroupFabric

group = DefaultGroupFabric([bx.to_mesh() for bx in boxes], uuid='fabric-group')

serve.start()
