import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from _case16_data import surf0,surf11,surf12,surf21,surf22,surf31,expected_endpoints

def print_res_summary(res,expected_endpts):
    keys=[
        'branches',
        'points',
        'singularities',
        'overlap_regions',
        #'unresolved_regions',
    ]
    
    for k in keys:
        if k=='branches':
            print(k, len(res[k]), sep=':')
            if len(res[k])!=1:
                print(f'WARNING: number of branches ({len(res[k])}) does not match expected (1)')

            for j,b in enumerate(res['branches']):
                endpts_match=(np.linalg.norm(b.curve[1][0] - expected_endpts[0])<TOL and   np.linalg.norm(b.curve[1][-1] - expected_endpts[1])<TOL) or (np.linalg.norm(b.curve[1][-1] - expected_endpts[0])<TOL and   np.linalg.norm(b.curve[1][0] - expected_endpts[1])<TOL)

                print('endpts match:', endpts_match, end='  ')
                if not endpts_match:


                    print(min((np.linalg.norm(b.curve[1][0] - expected_endpts[0]), np.linalg.norm(b.curve[1][-1] - expected_endpts[0]))),min((np.linalg.norm(b.curve[1][0] - expected_endpts[1]), np.linalg.norm(b.curve[1][1] - expected_endpts[1]))))
                    print('expected_endpts:',expected_endpts)

                print(f'({j}, endpoints:{b.curve[1][0].tolist()}, {b.curve[1][-1].tolist()})',end=',')


            print(end='; ')
        else:
            print(k,len(res[k]),sep=':',end='; ')
    print()



from mmcore.numeric.intersection.ssx import nurbs_ssx
TOL=1e-3

res=nurbs_ssx(surf0,surf11, atol=TOL)
print_res_summary(res, expected_endpoints[0])


res=nurbs_ssx(surf0,surf12, atol=TOL)
print_res_summary(res,expected_endpoints[1])



res=nurbs_ssx(surf0,surf21, atol=TOL)
print_res_summary(res,expected_endpoints[2])



res=nurbs_ssx(surf0,surf22, atol=TOL)
print_res_summary(res,expected_endpoints[3])


res=nurbs_ssx(surf0,surf31, atol=TOL)
print_res_summary(res, expected_endpoints[4])


