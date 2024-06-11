import json
import yappi
import profile
import numpy as np
from mmcore.geom.vec.vec_speedups import scalar_norm,scalar_unit,scalar_dot

import mmcore.geom.implicit.marching
import mmcore.geom.implicit.intersection_curve
import importlib

importlib.reload(mmcore.geom.implicit.intersection_curve )
from mmcore.geom.implicit.marching import marching_intersection_curve_points,surface_point,surface_plane
from mmcore.geom.implicit.intersection_curve import  ImplicitIntersectionCurve,iterate_curves

from mmcore.geom.primitives import Tube

import time

def test_bodies1():
    x, y, v, u, z = [[[12.359112840551504, -7.5948049557495425, 0.0], [2.656625109045951, 1.2155741170561933, 0.0]],
                     [[7.14384241216015, -6.934735074711716, -0.1073366304415263],
                      [7.0788761013028365, 10.016931402130641, 0.8727530304189204]],
                     [[8.072688942425103, -2.3061831591019826, 0.2615779273274319],
                      [7.173685617288537, -3.4427234423361512, 0.4324928834164773],
                      [7.683972288682133, -2.74630545102506, 0.07413871667321925],
                      [7.088944240699163, -4.61458155002528, -0.22460509818398067],
                      [7.304629277158477, -3.9462033818505433, 0.8955725109783643],
                      [7.304629277158477, -3.3362864951018985, 0.8955725109783643],
                      [7.304629277158477, -2.477065729786164, 0.7989970582016114],
                      [7.304629277158477, -2.0988672326949933, 0.7989970582016114]], 0.72648, 1.0]

    aa = np.array(x)
    bb = np.array(y)
    t11 = Tube(aa[0], aa[1], z, 0.2)
    t21 = Tube(bb[0], bb[1], u, 0.2)
    yappi.set_clock_type("wall")  # Use set_clock_type("wall") for wall time
    yappi.start(builtins=True)
    s = time.perf_counter_ns()
    vv = np.array(v)
    res1 = []
    for i in range(len(vv)):
        res1.append(
            marching_intersection_curve_points(
                t11.implicit,
                t21.implicit,
                t11.normal,
                t21.normal,
                vv[i],
                max_points=200,
                step=0.2,
                tol=1e-5
            )
        )
    ms=(time.perf_counter_ns() - s) * 1e-6
    yappi.stop()
    func_stats = yappi.get_func_stats()
    func_stats.save(f"test_perf_intersection_test_bodies1_stats_{int(time.time())}.pstat", type='pstat')
    #print("mmcore builtin primitives speed:", ms, 'ms.')
    rres1 = []
    for r in res1:
        rres1.append(r.tolist())
    #print(rres1)
    with open(__file__.replace('.py', '_test_bodies1_log.txt'), 'a') as f:
        f.writelines([f'\n{ms} ms. {len(res1)}'])


def test_bodies2():
    x, y, v, u, z = [[[12.359112840551504, -7.5948049557495425, 0.0], [2.656625109045951, 1.2155741170561933, 0.0]],
                     [[7.14384241216015, -6.934735074711716, -0.1073366304415263],
                      [7.0788761013028365, 10.016931402130641, 0.8727530304189204]],
                     [[8.072688942425103, -2.3061831591019826, 0.2615779273274319],
                      [7.173685617288537, -3.4427234423361512, 0.4324928834164773],
                      [7.683972288682133, -2.74630545102506, 0.07413871667321925],
                      [7.088944240699163, -4.61458155002528, -0.22460509818398067],
                      [7.304629277158477, -3.9462033818505433, 0.8955725109783643],
                      [7.304629277158477, -3.3362864951018985, 0.8955725109783643],
                      [7.304629277158477, -2.477065729786164, 0.7989970582016114],
                      [7.304629277158477, -2.0988672326949933, 0.7989970582016114]], 0.72648, 1.0]

    aa = np.array(x)
    bb = np.array(y)
    t11 = Tube(aa[0], aa[1], z, 0.2)
    t21 = Tube(bb[0], bb[1], u, 0.2)
    s1 = time.perf_counter_ns()
    yappi.set_clock_type("wall")  # Use set_clock_type("wall") for wall time
    yappi.start(builtins=True)
    crv = ImplicitIntersectionCurve(t11, t21, tol=0.001)
    crv.build_tree()
    it=iterate_curves(crv,step=0.2)
    s = time.perf_counter_ns()
    res2 = []
    for item in it:
        res2.append(item)
    e=time.perf_counter_ns()
    yappi.stop()
    func_stats = yappi.get_func_stats()
    func_stats.save(f"test_perf_intersection_test_bodies2_stats_{int(time.time())}.pstat", type='pstat')

    ms = (e- s) * 1e-6
    ms1 = (e - s1) * 1e-6
    #print("mmcore builtin primitives speed:", ms, 'ms.')

    #print(res2)
    with open(__file__.replace('.py', '_test_bodies2_log.txt'), 'a') as f:
        f.writelines([f'\n{ms1} ms. {ms} ms. {len(res2)}'])
if __name__ == '__main__':
    import time
    #yappi.set_clock_type("wall")  # Use set_clock_type("wall") for wall time
    #yappi.start(builtins=True)
    test_bodies1()
    #yappi.stop()
    #func_stats=yappi.get_func_stats()
    #thread_stats=yappi.get_thread_stats()
    #func_stats.save(f"test_perf_intersection_test_bodies1_stats_{int(time.time())}.pstat", "pstat")
    #thread_stats.print_all()

    #yappi.set_clock_type("wall")  # Use set_clock_type("wall") for wall time
    #yappi.start(builtins=True)
    test_bodies2()
    #yappi.stop()
    #func_stats=yappi.get_func_stats()
    #func_stats.save(f"test_perf_intersection_test_bodies2_stats_{int(time.time())}.pstat", "pstat")
    #thread_stats1 = yappi.get_thread_stats()
    #thread_stats1.print_all()


