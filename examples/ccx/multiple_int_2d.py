import numpy as np
from mmcore.geom._nurbs_eval import NURBSCurveTuple


val = [NURBSCurveTuple(
    order=3,
    knot=np.array([  0.   ,   0.   ,   0.   ,  60.684,  60.684, 121.368, 121.368,
           182.051, 182.051, 242.735, 242.735, 242.735]),
    control_points=np.array([[161.01 ,  95.097],
           [175.987, 130.709],
           [140.376, 145.685],
           [104.764, 160.662],
           [ 89.788, 125.051],
           [ 74.811,  89.439],
           [110.422,  74.463],
           [146.034,  59.486],
           [161.01 ,  95.097]]),
    weights=np.array([1.   , 0.707, 1.   , 0.707, 1.   , 0.707, 1.   , 0.707, 1.   ])
), NURBSCurveTuple(
    order=3,
    knot=np.array([0.   , 0.   , 0.   , 1.571, 1.571, 3.142, 3.142, 4.712, 4.712,
           6.283, 6.283, 6.283]),
    control_points=np.array([[141.707, 107.412],
           [114.416, 107.412],
           [114.416, 183.626],
           [114.416, 259.841],
           [141.707, 259.841],
           [168.998, 259.841],
           [168.998, 183.626],
           [168.998, 107.412],
           [141.707, 107.412]]),
    weights=np.array([1.   , 0.707, 1.   , 0.707, 1.   , 0.707, 1.   , 0.707, 1.   ])
),NURBSCurveTuple(
    order=3,
    knot=np.array([  0.   ,   0.   ,   0.   ,  76.691,  76.691, 153.382, 153.382,
           230.074, 230.074, 306.765, 306.765, 306.765]),
    control_points=np.array([[132.055, 206.591],
           [141.374, 254.516],
           [ 93.449, 263.835],
           [ 45.523, 273.154],
           [ 36.204, 225.229],
           [ 26.885, 177.303],
           [ 74.811, 167.984],
           [122.736, 158.665],
           [132.055, 206.591]]),
    weights=np.array([1.   , 0.707, 1.   , 0.707, 1.   , 0.707, 1.   , 0.707, 1.   ])
),NURBSCurveTuple(
    order=4,
    knot=np.array([  0.   ,   0.   ,   0.   ,   0.   , 123.163, 246.326, 369.489,
           492.652, 615.815, 615.815, 615.815, 615.815]),
    control_points=np.array([[  3.588, 128.046],
           [ 20.895, 198.27 ],
           [ 67.489, 121.723],
           [ 88.124, 203.595],
           [133.054, 164.656],
           [209.934, 231.552],
           [ 87.791, 295.453],
           [ 71.15 , 227.225]]),
    weights=np.array([1., 1., 1., 1., 1., 1., 1., 1.])
)
]
from mmcore.numeric.intersection.ccx import nurbs_ccx,nurbs_ccx_multiple
from mmcore.extras.renderer.renderer3d import Viewer,OrbitCamera

isolated,overlaps=nurbs_ccx_multiple(val)
print(isolated)

viewer=Viewer(camera=OrbitCamera(target=(106.97827 ,  167.56537,0),up=(0,1.,0.), ortho_half_height=110,distance=300,yaw= -3*np.pi/2,pitch= -np.pi/2))
viewer.cam.lock_orbit(True)
for curve in val:
    viewer.add(curve, color=(0.7, 0.9, 1.0, 1.0))
for pt in isolated['point']:
    viewer.add(pt, color=(0.0, 1.0, 0.5,1.0),size_px=13)

viewer.run()
print(isolated['point'])