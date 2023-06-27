# This Api provides an extremely flexible way of updating data. You can pass any part of the parameter dictionary
# structure, the parameters will be updated recursively and only the part of the graph affected by the change
# will be recalculated.
import IPython
import numpy as np

import mmcore
from mmcore.base.params import param_graph_node_native
from mmcore.base.sharedstate import debug_properties, serve
from mmcore.geom.parametric.nurbs import NurbsCurve
from mmcore.geom.parametric.sweep import Sweep


@param_graph_node_native
def spiral(radius=500, high=500, pitches=12):
    def evaluate(t):
        return np.cos(t * pitches) * radius * np.cos(t), np.sin(t * pitches) * radius * np.cos(t), t * high

    return NurbsCurve([evaluate(t) for t in np.linspace(0, 1, 8 * pitches)], dimension=3)


grasshopper_default_material = dict(color=(200, 10, 10), opacity=0.4)

profile_outer = [[-23, 40, 0], [23, 40, 0], [25, 38, 0],
                 [25, -38, 0], [23, -40, 0], [-23, -40, 0],
                 [-25, -38, 0], [-25, 38, 0], [-23, 40, 0]]

profile_inner = [[-21.0, 36.5, 0], [21, 36.5, 0], [21.5, 36, 0],
                 [21.5, -36, 0], [21, -36.5, 0], [-21, -36.5, 0],
                 [-21.5, -36, 0], [-21.5, 36, 0], [-21.0, 36.5, 0]]

profile_outer.reverse()  # Don't mind me, I was just drunk and wrote down the original clockwise profile coordinates
profile_inner.reverse()  # Don't mind me, I was just drunk and wrote down the original clockwise profile coordinates

sweep = Sweep(name="test_sweep", uuid="test_sweep", path=spiral, profiles=(profile_outer, profile_inner),
              **grasshopper_default_material)
# If you are using a debug viewer, the line below will force it to broadcast exactly the object you want,
# this can be very useful for large scenes and more.
debug_properties["target"] = "test_sweep"

# Starting the mmcore backend
serve.start()

# Comment the lines below if using ipython by default (e.g. in Pycharm with a python console)
IPython.embed(header=f"[mmcore {mmcore.__version__()}]")

# Go to https://viewer.contextmachine.online/v2/scene/006ccbec-f07c-48b4-ac4d-b5b456d6e7d7
# or look "Sweep Local Example" scene in all scenes' menu.
