import time

import numpy as np

from mmcore.geom.nurbs import NURBSSurface
from mmcore.construction import cylinder_surface_2pt,torus
from mmcore.numeric.intersection.ssx import ssx

x, y, v, u, z = [
    [[12.359112840551504, -7.5948049557495425, 0.0], [2.656625109045951, 1.2155741170561933, 0.0]],
                 [[7.14384241216015, -6.934735074711716, -0.1073366304415263],
                  [7.0788761013028365, 4.016931402130641, 0.8727530304189204]],
    
                 [[8.072688942425103, -2.3061831591019826, 0.2615779273274319],
                  [7.173685617288537, -3.4427234423361512, 0.4324928834164773],
                  [7.683972288682133, -2.74630545102506, 0.07413871667321925],
                  [7.088944240699163, -4.61458155002528, -0.22460509818398067],
                  [7.304629277158477, -3.9462033818505433, 0.8955725109783643],
                  [7.304629277158477, -3.3362864951018985, 0.8955725109783643],
                  [7.304629277158477, -2.477065729786164, 0.7989970582016114],
                  [7.304629277158477, -2.0988672326949933, 0.7989970582016114]], 0.72648, 1.0]
from mmcore.geom.nurbs import NURBSCurve
from mmcore.geom._nurbs_eval import _tuple_to_nurbs
import numpy as np

st1=cylinder_surface_2pt(*np.array(x),radius=u)
st2=cylinder_surface_2pt(*np.array(y),radius=z)
s1=_tuple_to_nurbs(st1)
s2=_tuple_to_nurbs(st2)
s3=_tuple_to_nurbs(torus())
from mmcore.renderer.renderer3dv2 import CADRenderer, Camera



from mmcore.numeric.intersection.ssx import ssx
import logging
logging.basicConfig(level=logging.DEBUG)

s=time.time()
result=ssx(s1,s2,tol=1e-7,spt=0.001)


print(f'intersection computed at: {time.time() - s} sec.')


print(f'\n({s1} X \n\t{s2}):')

for i, (spatial, uv1, uv2) in enumerate(result[0]):
            print(f'\t{i + 1}. {spatial}, {uv1}, {uv2}')
            cpts=(spatial.control_points).tolist()
            cpts_repr = repr(cpts)
            #if len(cpts)>4:
            #    cpts_repr=f'[{cpts[1]}, {cpts[2]}, ... , {cpts[-2]}, {cpts[-1]}]'
            print(f'\t\tcontrol points: {cpts_repr}')
            print(f'\t\tdegree: {spatial.degree}')


try:
    from mmcore.renderer.renderer3dv2 import CADRenderer,Camera

    print(dir(Camera))
    centr=np.average(s1.control_points_flat, axis=0)
    renderer=CADRenderer(camera=Camera( zoom=75.
        )
    )

    renderer.add_nurbs_surface(s1,color=(1.,1.,1.))
    renderer.add_nurbs_surface(s2,color=(1.,1.,1.))

    for (crv,uv1,uv2) in result[0]:
        renderer.add_nurbs_curve(crv, color=(0.,1.,0.5))


    renderer.run()

except ModuleNotFoundError as err:
    print("mmcore.renderer is not installed, skip preview.")
except ImportError as err:
    print("mmcore.renderer is not installed, skip preview.")
except Exception as err:
    raise err
