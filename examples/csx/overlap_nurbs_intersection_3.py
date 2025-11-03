import time

from mmcore._test_data import csx as csx_cases

from mmcore.numeric.intersection.csx import nurbs_csx

from mmcore.geom._nurbs_eval import _tuple_to_nurbs, _nurbs_to_tuple
from mmcore.geom._nurbs_knots import split_curve_multiple
import numpy as np
from mmcore.geom._nurbs_eval import NURBSCurveTuple


curve1 = NURBSCurveTuple(
    order=2,
    knot=np.array([ 0.        ,  0.        , 20.65300965, 65.96260962, 69.30975481,
           69.30975481]),
    control_points=np.array([[ 66.01811696,  90.95353754,  45.81225139],
           [ 66.01811696,  90.95353754,  20.75799567],
           [111.32771693,  90.95353754,  20.75799567],
           [110.18292585,  87.8082499 ,  20.75799567]]),
    weights=np.array([1., 1., 1., 1.])
)



curve2 = NURBSCurveTuple(
    order=4,
    knot=np.array([0., 0., 0., 0., 2.10490611,
                   2.10490611, 2.10490611, 4.35578185, 4.35578185, 6.60665591,
                   6.60665591, 8.85752794, 8.85752794, 11.10839997, 11.10839997,
                   13.359272, 13.359272, 15.61014479, 15.61014479, 17.86102249,
                   17.86102249, 20.11190103, 20.11190103, 20.11190103, 31.3033027,
                   31.3033027, 31.3033027, 31.3033027]),
    control_points=np.array([[90.06507871, 81.18027761, -4.25499835],
                             [90.51406358, 81.58907357, -4.0387382],
                             [90.96304844, 81.99786953, -3.82247806],
                             [91.41203331, 82.4066655, -3.60621791],
                             [92.85239561, 83.71810012, -2.91244628],
                             [94.01988661, 85.30553815, -2.19782612],
                             [95.65026587, 88.77912871, -0.72689091],
                             [96.11300253, 90.66371219, 0.02942412],
                             [96.28502389, 94.43829065, 1.58374726],
                             [95.99448464, 96.32669496, 2.38175534],
                             [94.72680657, 99.82380743, 4.01946362],
                             [93.75013478, 101.43108127, 4.85916382],
                             [91.27842704, 104.13179098, 6.58025473],
                             [89.7844795, 105.2243694, 7.46164544],
                             [86.51953959, 106.74648291, 9.26611609],
                             [84.75002525, 107.1758931, 10.18919605],
                             [81.209659, 107.32794476, 12.07704534],
                             [79.44029924, 107.05076864, 13.04181475],
                             [76.16713956, 105.85446369, 15.01303942],
                             [74.66467989, 104.93579006, 16.01949469],
                             [73.40404256, 103.77584031, 17.04679174],
                             [67.13613493, 98.00855285, 22.15452797],
                             [60.8682273, 92.24126539, 27.26226419],
                             [54.60031967, 86.47397794, 32.37000042]]),
    weights=np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                      1., 1., 1., 1., 1., 1., 1.])
)

surface, curve3 = csx_cases[0]
curve3=_nurbs_to_tuple(curve3)
inters = []
overs = []
pts = []
s = time.time()
result1 = nurbs_csx(_tuple_to_nurbs(curve1), surface)
print(f"CSX 1 performed at: {time.time()-s} secs.")
s = time.time()
result2 = nurbs_csx(_tuple_to_nurbs(curve2), surface)
print(f"CSX 2 performed at: {time.time()-s} secs.")
s = time.time()
result3 = nurbs_csx(_tuple_to_nurbs(curve3), surface)

print(f"CSX 3 performed at: {time.time()-s} secs.")



try:
    from mmcore.extras.renderer import CADRenderer, Camera

    print(dir(Camera))

  
    renderer = CADRenderer(camera=Camera(zoom=50.0, near=0.1))
  
    tess=renderer.add_nurbs_surface(surface, color=(0.9,0.9,0.9),surface_color = (0.5, 0.5, 0.9, 0.1))
  
    def render_intersects(renderer,curve1,result1, curve_color=(0.0, 1.0, 0.5),inter_color=(1.0, 0.5, 0.0)):
        overlaps_prms=[]
        
        for item in result1:
            print(item)
            if item[0]=='overlap':
                tinter,uinter,vinter=item[2]
                
                renderer.add_point(item[1][0], np.array(inter_color), 4)
                
                renderer.add_point(item[1][1], np.array(inter_color), 4)
                overlaps_prms.extend((tinter.low,tinter.upp))
                
                
            else:
                renderer.add_point(item[1], np.array(inter_color), 4)
        if len(overlaps_prms)>0:
            overlaps_prms=np.unique(overlaps_prms).tolist()
           
            at_start=(overlaps_prms[0]-curve1.interval()[0])<=1e-6
            if at_start:
                del overlaps_prms[0]
            if (curve1.interval()[1]-overlaps_prms[-1] ) <= 1e-6:
                del overlaps_prms[-1]
          
                
            for  i,segm in enumerate(split_curve_multiple(curve1, overlaps_prms)):
                
                if ((i%2) ==0) :
                
                    renderer.add_nurbs_curve(
                    _tuple_to_nurbs(segm),
                    color=np.array(inter_color) if at_start else np.array(curve_color)
                    )
                else:
                    renderer.add_nurbs_curve(
                        _tuple_to_nurbs(segm),
                        color=np.array(curve_color) if at_start else np.array(inter_color)
                    )
        else:
            
            renderer.add_nurbs_curve(
                _tuple_to_nurbs(curve1),
                color=np.array(curve_color)
            )
    
    render_intersects(renderer,curve1,result1)
    render_intersects(renderer, curve2, result2)
    #render_intersects(renderer, curve3, result3)
    renderer.run()

except ModuleNotFoundError as err:
    print("mmcore.renderer is not installed, skip preview.")
except ImportError as err:
    print("mmcore.renderer is not installed, skip preview.")
except Exception as err:
    raise err
