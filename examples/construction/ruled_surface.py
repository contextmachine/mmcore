import numpy as np
from mmcore.geom.nurbs import NURBSCurve

from mmcore.construction import ruled

# 1. Ruled surface by curves of equal degree but with different number of control points:
#  Creating curves:
first_curve = NURBSCurve(
    np.array(
        [
            [-20.0, 0.0, -10.0],
            [-16.666666666666666, 0.0, -1.0],
            [-13.33333333333333, 0.0, -1.0],
            [-5.0, 0.0, -16.7],
            [5.0, 0.0, 8.3],
            [15.0, 0.0, -16.7],
            [23.33333333333333, 0.0, 0.0],
            [26.66666666666666, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ]
    ),
    degree=3,
)
second_curve = NURBSCurve(
    np.array(
        [
            [-20.0, 43.0, -0.03333333333333333],
            [-5.0, 43.0, 6.6666666666666666],
            [5.0, 43.0, -18.33333333333333],
            [15.0, 43.0, 6.666666666666666],
            [23.33333333333333, 43.0, -6.0],
            [30.0, 43.0, -5.0],
        ]
    ),
    degree=3,
)
#  Build ruled surface:
first_surface = ruled(first_curve, second_curve)


# 2. Ruled surface by curves of different degree and number of control points:
#  Create curves:


third_curve = NURBSCurve(np.array([[0.41003319883988076, -5.9558709997242776, -0.45524326627631317],
             [0.41003319883988076, -5.5445084274881866, 0.31289808372671224],
             [0.41003319883988076, -4.2689095570901747, 0.27335792945560905],
             [0.41003319883988054, -2.8275390970241014, 0.38227620969285792],
             [0.41003319883988071, -1.5384497736905611, -0.55192398184841063],
             [0.4100331988398806, -1.0649511609106423, 0.18024597033519471],
             [0.41003319883988076, -0.45939612773632504, 0.16055590176917547],
             [0.41003319883988065, -0.039150933363156781, 0.19313507758724083]])*10, 2
              )
fourth_curve=NURBSCurve(np.array([[-1.2431856590487269, -5.5356810246947985, 0.0],
             [-1.1678165416732957, -3.8992861465815305, 1.3621802986275990],
             [-1.9780801959755556, -3.9866785366038910, 0.0], [-1.0059685065682198, -0.38528326448094535, 0.0]])*10, 3)

second_surface =  ruled(third_curve, fourth_curve)



from mmcore.compat.step.step_writer import StepWriter
from pathlib import Path
current_example_dir=Path(__file__).parent
with (current_example_dir/'ruled_surfaces.step').open('w') as f:
    writer=StepWriter()
    writer.add_nurbs_surface(first_surface)
    writer.add_nurbs_surface(second_surface)
    writer.write(f)
    print(f"The geometry of the ruled surfaces is written to a {current_example_dir/'ruled_surfaces.step'}")