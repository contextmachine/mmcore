"""
 
 This example demonstrates the intersection of two NURBS spheres.
"""
import time
from mmcore.geom._nurbs_eval import _tuple_to_nurbs

from mmcore.numeric.intersection.ssx import ssx

# Creating intersection objects
import numpy as np
from mmcore.geom._nurbs_eval import NURBSSurfaceTuple


st1 = NURBSSurfaceTuple(
    order_u=3,
    order_v=3,
    knot_u=np.array([ 0.        ,  0.        ,  0.        , 17.27875959, 17.27875959,
           34.55751919, 34.55751919, 51.83627878, 51.83627878, 69.11503838,
           69.11503838, 69.11503838]),
    knot_v=np.array([-17.27875959, -17.27875959, -17.27875959,  -0.        ,
            -0.        ,  17.27875959,  17.27875959,  17.27875959]),
    control_points=np.array([[[  6.,   0., -11.],
            [ 17.,   0., -11.],
            [ 17.,   0.,  -0.],
            [ 17.,   0.,  11.],
            [  6.,   0.,  11.]],

           [[  6.,   0., -11.],
            [ 17.,  11., -11.],
            [ 17.,  11.,  -0.],
            [ 17.,  11.,  11.],
            [  6.,   0.,  11.]],

           [[  6.,   0., -11.],
            [  6.,  11., -11.],
            [  6.,  11.,  -0.],
            [  6.,  11.,  11.],
            [  6.,   0.,  11.]],

           [[  6.,   0., -11.],
            [ -5.,  11., -11.],
            [ -5.,  11.,  -0.],
            [ -5.,  11.,  11.],
            [  6.,   0.,  11.]],

           [[  6.,   0., -11.],
            [ -5.,   0., -11.],
            [ -5.,   0.,  -0.],
            [ -5.,   0.,  11.],
            [  6.,   0.,  11.]],

           [[  6.,   0., -11.],
            [ -5., -11., -11.],
            [ -5., -11.,  -0.],
            [ -5., -11.,  11.],
            [  6.,   0.,  11.]],

           [[  6.,   0., -11.],
            [  6., -11., -11.],
            [  6., -11.,  -0.],
            [  6., -11.,  11.],
            [  6.,   0.,  11.]],

           [[  6.,   0., -11.],
            [ 17., -11., -11.],
            [ 17., -11.,  -0.],
            [ 17., -11.,  11.],
            [  6.,   0.,  11.]],

           [[  6.,   0., -11.],
            [ 17.,   0., -11.],
            [ 17.,   0.,  -0.],
            [ 17.,   0.,  11.],
            [  6.,   0.,  11.]]]),
    weights=np.array([[1.        , 0.70710678, 1.        , 0.70710678, 1.        ],
           [0.70710678, 0.5       , 0.70710678, 0.5       , 0.70710678],
           [1.        , 0.70710678, 1.        , 0.70710678, 1.        ],
           [0.70710678, 0.5       , 0.70710678, 0.5       , 0.70710678],
           [1.        , 0.70710678, 1.        , 0.70710678, 1.        ],
           [0.70710678, 0.5       , 0.70710678, 0.5       , 0.70710678],
           [1.        , 0.70710678, 1.        , 0.70710678, 1.        ],
           [0.70710678, 0.5       , 0.70710678, 0.5       , 0.70710678],
           [1.        , 0.70710678, 1.        , 0.70710678, 1.        ]])
)




st2 = NURBSSurfaceTuple(
    order_u=3,
    order_v=3,
    knot_u=np.array([ 0.        ,  0.        ,  0.        , 17.27875959, 17.27875959,
           34.55751919, 34.55751919, 51.83627878, 51.83627878, 69.11503838,
           69.11503838, 69.11503838]),
    knot_v=np.array([-17.27875959, -17.27875959, -17.27875959,  -0.        ,
            -0.        ,  17.27875959,  17.27875959,  17.27875959]),
    control_points=np.array([[[ 11.,   0., -13.],
            [ 22.,   0., -13.],
            [ 22.,   0.,  -2.],
            [ 22.,   0.,   9.],
            [ 11.,   0.,   9.]],

           [[ 11.,   0., -13.],
            [ 22.,  11., -13.],
            [ 22.,  11.,  -2.],
            [ 22.,  11.,   9.],
            [ 11.,   0.,   9.]],

           [[ 11.,   0., -13.],
            [ 11.,  11., -13.],
            [ 11.,  11.,  -2.],
            [ 11.,  11.,   9.],
            [ 11.,   0.,   9.]],

           [[ 11.,   0., -13.],
            [  0.,  11., -13.],
            [  0.,  11.,  -2.],
            [  0.,  11.,   9.],
            [ 11.,   0.,   9.]],

           [[ 11.,   0., -13.],
            [  0.,   0., -13.],
            [  0.,   0.,  -2.],
            [  0.,   0.,   9.],
            [ 11.,   0.,   9.]],

           [[ 11.,   0., -13.],
            [  0., -11., -13.],
            [  0., -11.,  -2.],
            [  0., -11.,   9.],
            [ 11.,   0.,   9.]],

           [[ 11.,   0., -13.],
            [ 11., -11., -13.],
            [ 11., -11.,  -2.],
            [ 11., -11.,   9.],
            [ 11.,   0.,   9.]],

           [[ 11.,   0., -13.],
            [ 22., -11., -13.],
            [ 22., -11.,  -2.],
            [ 22., -11.,   9.],
            [ 11.,   0.,   9.]],

           [[ 11.,   0., -13.],
            [ 22.,   0., -13.],
            [ 22.,   0.,  -2.],
            [ 22.,   0.,   9.],
            [ 11.,   0.,   9.]]]),
    weights=np.array([[1.        , 0.70710678, 1.        , 0.70710678, 1.        ],
           [0.70710678, 0.5       , 0.70710678, 0.5       , 0.70710678],
           [1.        , 0.70710678, 1.        , 0.70710678, 1.        ],
           [0.70710678, 0.5       , 0.70710678, 0.5       , 0.70710678],
           [1.        , 0.70710678, 1.        , 0.70710678, 1.        ],
           [0.70710678, 0.5       , 0.70710678, 0.5       , 0.70710678],
           [1.        , 0.70710678, 1.        , 0.70710678, 1.        ],
           [0.70710678, 0.5       , 0.70710678, 0.5       , 0.70710678],
           [1.        , 0.70710678, 1.        , 0.70710678, 1.        ]])
)


s1 = _tuple_to_nurbs(st1)
s2 = _tuple_to_nurbs(st2)

# Perform SSX
start_time = time.time()
result = ssx(s1, s2, tol=1e-7, spt=0.001)

print(f"intersection computed at: {time.time() - start_time} sec.")

# Printing and rendering results
print(f"\n({s1} X \n\t{s2}):")

for i, (spatial, uv1, uv2) in enumerate(result[0]):
    print(f"\t{i + 1}. {spatial}, {uv1}, {uv2}")
    cpts = (spatial.control_points).tolist()
    cpts_repr = repr(cpts)
    # if len(cpts)>4:
    #    cpts_repr=f'[{cpts[1]}, {cpts[2]}, ... , {cpts[-2]}, {cpts[-1]}]'
    print(f"\t\tcontrol points: {cpts_repr}")
    print(f"\t\tdegree: {spatial.degree}")


try:
    
    from mmcore.extras.renderer import CADRenderer, Camera
    def draw_ssx(s1,s2, result, renderer=None):
        
        renderer = renderer if renderer is not None else CADRenderer(camera=Camera(zoom=50.0, near=1.))
        renderer.add_nurbs_surface(s1, color=(1.0, 1.0, 1.0))
        renderer.add_nurbs_surface(s2, color=(1.0, 1.0, 1.0), )
        
        for crv, uv1, uv2 in result[0]:
            renderer.add_nurbs_curve(crv, color=(0.0, 1.0, 0.5))
        return renderer

    renderer=draw_ssx(s1, s2, result)

    renderer.run()

except ModuleNotFoundError as err:
    print("mmcore.renderer is not installed, skip preview.")
except ImportError as err:
    print("mmcore.renderer is not installed, skip preview.")
except Exception as err:
    raise err
