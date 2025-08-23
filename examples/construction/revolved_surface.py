import numpy as np
from mmcore.geom._nurbs_eval import NURBSCurveTuple
from mmcore.construction import revolved
from mmcore.geom._nurbs_knots import decompose_surface

control_points = np.array(
    [
        [72.0, -67.0, 0.0],
        [91.924766084067414, -67.0, 0.0],
        [91.924766084067414, 7.3602393546629514, 0.0],
        [72.0, 7.3602393546629514, 0.0]
    ]
)
profile = NURBSCurveTuple(
    order=2,
    knot=np.array([0.0, 0.0, 19.924766084067379, 94.285005438730337, 114.20977152279772, 114.20977152279772]),
    control_points=control_points,
    weights=np.ones((control_points.shape[0],), float),
)
axis = np.array([[60.0, -80.0, 0.0], [0.0, 0.0, 0.0]])


surf = revolved(profile, axis, (0.0, 2 * np.pi))


from mmcore.compat.step.step_writer import StepWriter

we = StepWriter()

ref1 = we.add_nurbs_surface(surf, (0.5, 0.5, 0.5), name="surface1")
with open("revolved_surfaces.step", "w") as f:
    we.step_file.write(f)
