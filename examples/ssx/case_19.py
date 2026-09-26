import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from _case_19_data import surf1,surf2

from mmcore.numeric.intersection.ssx import nurbs_ssx
if __name__ == "__main__":
    import time
    start=time.perf_counter()
    result= nurbs_ssx( surf1, surf2, atol=1e-3)
    end=time.perf_counter()
    print( result)
    print(f'{end-start} secs.')


