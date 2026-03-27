import numpy as np

from mmcore.extras.renderer.renderer3d import OrbitCamera, Viewer
from mmcore.geom._nurbs_eval import NURBSCurveTuple
from mmcore.geom._nurbs_knots import trim_curve
from mmcore.numeric.intersection.ccx import nurbs_ccx

import numpy as np
from mmcore.geom._nurbs_eval import NURBSCurveTuple


curve1 = NURBSCurveTuple(
    order=4,
    knot=np.array([0., 0., 0., 0., 1., 1., 1., 1.]),
    control_points=np.array([[-0.92475126,  1.46166592,  0.        ],
           [-0.25992915,  0.7062367 ,  0.        ],
           [-1.41598441,  1.14500509,  0.        ],
           [-0.1356103 ,  1.30775672,  0.        ]]),
    weights=np.array([1., 1., 1., 1.])
)
import numpy as np
from mmcore.geom._nurbs_eval import NURBSCurveTuple


curve2 = NURBSCurveTuple(
    order=3,
    knot=np.array([0., 0., 0., 1., 1., 1.]),
    control_points=np.array([[-0.77783268,  1.2703046 ,  0.        ],
           [-0.17966058,  1.13222589,  0.        ],
           [-0.25365612,  1.65359905,  0.        ]]),
    weights=np.array([1., 1., 1.])
)


tol=0.001
translate_d=tol/10
translate_vec=np.array([-0.09709519,  0.9952751,0.])*translate_d

curve3=curve2._replace(control_points=curve2.control_points-(translate_vec[np.newaxis,:]))

isolated1,overs1=nurbs_ccx(curve1,curve2,tol)

# curve1 x curve3 is not overlap!
isolated2,overs2=nurbs_ccx(curve1,curve3,tol)
viewer=Viewer(camera=OrbitCamera(target=(-0.5 , 1.2,0.,),up=(0,1.,0.), ortho_half_height=1,distance=1,yaw= -3*np.pi/2,pitch= -np.pi/2))
viewer.cam.lock_orbit(True)
viewer.add(curve1, samples=512)
viewer.add(curve3, color=(0.0, 0.6, 1.0, 1.0), samples=512)
for o in overs1:
    u_range,v_range,xyz_range=o
    uv0,uv1=zip(u_range, v_range)

    viewer.add_point3d(xyz_range[0], color=(0.7, 0.9, 0.0, 1.0), size_px=13)
    viewer.add_point3d(xyz_range[1], color=(0.7, 0.9, 0.0, 1.0), size_px=13)
    viewer.add(trim_curve(curve1, min(uv0[0], uv1[0]), max(uv0[0], uv1[0])),color=(0.7, 0.9, 0.0, 1.0))
    viewer.add(trim_curve(curve2, min(uv0[1], uv1[1]), max(uv0[1], uv1[1])),color=(0.7, 0.9, 0.0, 1.0))
    viewer.add(trim_curve(curve1,  min(uv0[0],uv1[0]),max(uv0[0],uv1[0])),color=(0.7, 0.9, 0.0, 1.0))

#viewer.add(curve2,color=(0.6,  1.0,  0.0, 1.0), samples=512)

print(isolated1)
print(overs1)
print()
print(isolated2)
print(overs2)
print()
for pt in isolated1['point']:

    viewer.add(pt
                    , color=(0.7, 0.9, 0.0, 1.0),size_px=13)

#for pt in isolated2['point']:
#    viewer.add(pt, color=(0.0, 0.6, 1.0, 1.0),size_px=13)
viewer.run()
