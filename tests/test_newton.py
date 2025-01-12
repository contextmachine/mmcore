import numpy as np
from mmcore.numeric.newton.cnewton import newtons_method as cnewthon_method
from mmcore.numeric.fdm import newtons_method
from mmcore._test_data import ssx as ssx_cases
import time
def test_newthon():
    surf=ssx_cases[2][0]
    pt=np.array([-2.,0,10.])
    tolerance=1e-6
    def fun(x):
        nonlocal pt
        return np.sum(np.power(surf.evaluate(x)-pt,2))

    (umin,umax),(vmin,vmax)=surf.interval()
    s1=time.perf_counter_ns()
    res1=newtons_method(fun, np.array([(umax+umin)/2,(vmax+vmin)/2]),tol=tolerance)
    e1=time.perf_counter_ns()-s1
    print("time: ",e1*1e-9)

    dist1=np.linalg.norm(surf.evaluate(res1) - pt)
    print(res1,  dist1)
    s2=time.perf_counter_ns()

    res2=np.array(cnewthon_method(fun, np.array([(umax+umin)/2,(vmax+vmin)/2]),tol=tolerance))
    e2=time.perf_counter_ns()-s2
    print("time: ", e2 * 1e-9)
    if e1>e2:
        print(f"C Newthon is faster at: {e1/e2}")
    elif e1==e2:
        print(f"Similar speed: {e1},{e2}")
    else :
        print(f"Py Newthon is faster at: {e2/e1}")

    dist2 = np.linalg.norm(surf.evaluate(res1) - pt)
    print(res2, dist2)
    assert np.abs(dist1-dist2)<tolerance
    assert np.all(np.abs(np.array(res1)-np.array(res2))<tolerance)