import numpy as np

from mmcore.construction import sweep1



from mmcore.nurbs._nurbs_eval import NURBSCurveTuple,_tuple_to_nurbs


rail=NURBSCurveTuple(
    order=6,
    knot=np.array([  0.       ,   0.       ,   0.       ,   0.       ,   0.       ,
             0.       , 140.3136521, 140.3136521, 140.3136521, 140.3136521,
           140.3136521, 140.3136521]),
    control_points=np.array([[  1., -11.,   0.],
           [-20., -23.,   2.],
           [-37.,  12., -18.],
           [ -9.,  21.,   0.],
           [ 16.,   3.,   0.],
           [ 16., -14.,   1.]]),
    weights=np.array([1., 1., 1., 1., 1., 1.])
)

prof1 = NURBSCurveTuple(
    order=2,
    knot=np.array([0.        , 0.        , 0.45454545, 0.63636364, 0.81818182,
           1.        , 1.        ]),
    control_points=np.array([[  1.        , -11.        ,   0.        ],
           [  1.        ,  -6.        ,   1.        ],
           [  1.51763809,  -6.        ,   1.93185165],
           [  1.64704761,  -8.        ,   2.41481457],
           [  1.77645714, -10.        ,   2.89777748]]),
    weights=np.array([1., 1., 1., 1., 1.])
)

prof2 = NURBSCurveTuple(
    order=2,
    knot=np.array([0.        , 0.        , 0.44951475, 0.62932064, 1.        ,
           1.        ]),
    control_points=np.array([[ 16.        , -14.        ,   1.        ],
           [ 11.93542154, -13.63780695,   2.70271723],
           [ 12.28369512, -13.35942638,   4.53519394],
           [ 15.70949467, -14.60999054,   4.08925852]]),
    weights=np.array([1., 1., 1., 1.])
)


surf= sweep1(rail, [prof1,prof2], anchors="first_cp", frame='RMF', sampler_tol=1.0)

from mmcore.compat.step.step_writer import StepWriter
from pathlib import Path
current_example_dir=Path(__file__).parent
with (current_example_dir/'ruled_surfaces.step').open('w') as f:
    writer=StepWriter()
    writer.add_nurbs_surface(surf)

    writer.write(f)
    print(f"The geometry of the sweep1 surface is written to a {current_example_dir/'sweep1_surface.step'}")


RENDER = True
if RENDER:
    try:
        from mmcore.extras.renderer import CADRenderer, Camera

        print(dir(Camera))
        centr = np.average(surf.control_points, axis=0)
        renderer = CADRenderer(camera=Camera(zoom=75.0))

        renderer.add_nurbs_surface(_tuple_to_nurbs(surf), color=(1.0, 1.0, 1.0))
      


        renderer.run()

    except ModuleNotFoundError as err:
        print("mmcore.renderer is not installed, skip preview.")
    except ImportError as err:
        print("mmcore.renderer is not installed, skip preview.")
    except Exception as err:
        raise err
