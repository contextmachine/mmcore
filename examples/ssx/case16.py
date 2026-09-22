import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from _case16_data import surf0,surf11,surf12,surf21,surf22,surf31

def print_res_summary(res):
    keys=[
        'branches',
        'points',
        'singularities',
        'overlap_regions',
        'unresolved_regions',
    ]
    for k in keys:
        print(k,res[k],sep=':',end='; ')



from mmcore.numeric.intersection.ssx import nurbs_ssx
TOL=1e-3

res=nurbs_ssx(surf0,surf11, atol=TOL)
print_res_summary(res)


res=nurbs_ssx(surf0,surf12, atol=TOL)
print_res_summary(res)



res=nurbs_ssx(surf0,surf21, atol=TOL)
print_res_summary(res)



res=nurbs_ssx(surf0,surf22, atol=TOL)
print_res_summary(res)



res=nurbs_ssx(surf0,surf31, atol=TOL)
print_res_summary(res)