import numpy as np
from mmcore.geom._nurbs_eval import NURBSCurveTuple, NURBSSurfaceTuple, _tuple_to_nurbs, evaluate_nurbs_curve

import time


import rich

from mmcore.geom._nurbs_knots import trim_curve
from mmcore.numeric import evaluate_curvature_vec
from mmcore.numeric.approx import adaptive_curve_sampler
from mmcore.numeric.intersection.csx import nurbs_csx_v2, nurbs_csx

curve = NURBSCurveTuple(
    order=3,
    knot=np.array([  0.        ,   0.        ,   0.        ,  91.83300275,
           148.16516477, 206.24102425, 296.33032955, 296.33032955,
           296.33032955]),
    control_points=np.array([[   0.64617093, -137.        ,  -24.        ],
           [   0.38629333,  -87.41587386,    0.        ],
           [  35.63100531,  -21.05150855,    0.        ],
           [  41.52768692,   23.7340877 ,    0.        ],
           [  20.09543746,   74.05471129,    0.        ],
           [  20.64617093,  101.40810621,   16.        ]]),
    weights=np.array([1., 1., 1., 1., 1., 1.])
)

surface = NURBSSurfaceTuple(
    order_u=4,
    order_v=3,
    knot_u=np.array([ 0.        ,  0.        ,  0.        ,  0.        , 40.22437072,
           40.22437072, 40.22437072, 40.22437072]),
    knot_v=np.array([  0.        ,   0.        ,   0.        , 153.56627554,
           307.13255109, 307.13255109, 307.13255109]),
    control_points=np.array([[[ -49.35382907, -137.        ,    0.        ],
            [  30.89785834,  -57.        ,    0.        ],
            [  46.17096231,   59.        ,    0.        ],
            [ -20.35382907,   97.40810621,    0.        ]],

           [[ -50.02049574, -137.        ,   11.        ],
            [  23.23119168,  -57.        ,   11.        ],
            [  38.50429565,   59.        ,   11.        ],
            [ -21.02049574,   97.40810621,   11.        ]],

           [[ -50.68716241, -137.        ,   22.        ],
            [  15.56452501,  -57.        ,   22.        ],
            [  30.83762898,   59.        ,   22.        ],
            [ -21.68716241,   97.40810621,   22.        ]],

           [[ -51.35382907, -137.        ,   33.        ],
            [   7.89785834,  -57.        ,   33.        ],
            [  23.17096231,   59.        ,   33.        ],
            [ -22.35382907,   97.40810621,   33.        ]]]),
    weights=np.array([[1., 1., 1., 1.],
           [1., 1., 1., 1.],
           [1., 1., 1., 1.],
           [1., 1., 1., 1.]])
)

#s = time.time()
#result = nurbs_csx(_tuple_to_nurbs(curve), _tuple_to_nurbs(surface))
#print(f"CSX v1 performed at: {time.time()-s} secs.")
#over=[]
#isol=[]
#print(result)
#for tp,item,uv in result:
#    if tp =='overlap':
#        over.append(item)
#    else:
#        isol.append(item)
#print('isolated:')
#rich.print(isol)
#print('overlaps:')
#rich.print(over)
s = time.time()
isolated,overlaps = nurbs_csx_v2(curve, surface)
print(f"CSX v2 performed at: {time.time()-s} secs.")


print('isolated:')

if isolated is not None:
    rich.print(isolated['point'].tolist())
print('overlaps:')
if overlaps is not None:
    rich.print(overlaps['point'].tolist())
RENDERER=False
if RENDERER:
    try:
        from mmcore.extras.renderer.renderer3d import Viewer,OrbitCamera
        viewer=Viewer(camera=OrbitCamera(near=1,far=1e+9))
        primary_color=(*(np.array([250, 102, 166])/255).tolist(),1)
        srf = viewer.add_nurbs_surface(surface, color=(0.7,0.7,0.7,1),surface_color=(0.5, 0.5, 0.9, 0.05),)
        if isolated is not None:
            for pt in isolated['point']:

                viewer.add(pt, color=(0.0, 1.0, 0.5,1.0),size_px=13)
        if overlaps is not None:

            for start,end in overlaps['point']:
                viewer.add(start, color=(0.0, 1.0, 0.5, 1.0), size_px=6)
                viewer.add(end, color=(0.0, 1.0, 0.5, 1.0), size_px=6)
            for o in overlaps['t']:
                #print(o)
                t0 = o[0]
                t1 = o[-1]
                viewer.add(trim_curve(curve,curve.interval()[0],t0),color=(0.9, 0.9, 0.9, 1.0))

                viewer.add(trim_curve(curve,t1, curve.interval()[1]), color=(0.9, 0.9, 0.9, 1.0))
                _c=trim_curve(curve, t0, t1)

                from mmcore.geom._nurbs_eval import evaluate_nurbs_curve_curvature
                points=[]
                ders=[]
                offset=1
                params,du_list,evals,s_list=adaptive_curve_sampler(_c)
                from mmcore.geom._nurbs_interp import hermite_interpolate_nurbs
                pts=np.linspace(*_c.interval(),500)
                for t in pts:

                    # uK=data["K"]/np.linalg.norm(data["K"])
                    evl=evaluate_nurbs_curve(curve,t,d_order=0)
                    viewer.add_point3d(evl['C'],color=(0.0, 1.0, 0.5, 1.0), size_px=3)
                    #evl["C1"]/=np.linalg.norm(evl['C1'])
                    #N=np.cross([0.,0.,1.],evl["C1"])
                    ##tnb=frenet_serret_frame_from_ders(   evl['C1'],
                    ##evl["C2"])
                    ##print(tnb)
                    #points.append(evl["C"] -      N/np.linalg.norm(N) * offset)
                    #ders.append(evl['C1'])
                #print(np.array(ders).tolist())
                #new_curve=hermite_interpolate_nurbs(np.array(points),np.array(ders),params,degree=3)

                #viewer.add(new_curve, color=(0.0, 1.0, 0.5, 1.0))

        viewer.run()


    except ModuleNotFoundError as err:
        print("mmcore.renderer is not installed, skip preview.")
    except ImportError as err:
        print("mmcore.renderer is not installed, skip preview.")
    except Exception as err:
        raise err
