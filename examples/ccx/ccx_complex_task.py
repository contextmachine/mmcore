import itertools
import os
import pickle
import warnings
import multiprocessing as mp
from pathlib import Path

import numpy as np
from mmcore.geom.bvh import (
    Object3D,
    build_bvh,
    intersect_bvh_objects,
    intersect_bvh,
    BoundingBox, BVHNode,
)
from mmcore.geom.nurbs import NURBSCurve,split_curve,split_curve_multiple

from mmcore.geom.features import points_order,PointsOrder
from mmcore.numeric.intersection.ccx import ccx
from mmcore.numeric.aabb import aabb



from mmcore.geom.polygon import is_point_in_polygon
from mmcore.numeric.vectors import scalar_norm



class Set2Curve(Object3D):
    def __init__(self, geom):
        self.geom = np.array(geom)
        #print(geom)
        self.geom[..., -1] = 0
        degree = len( self.geom) - 1 if len(self.geom) < 4 else 3
        #print(degree)
        self.polyline = NURBSCurve(self.geom, degree=degree)

        #k=np.array(self.polyline.knots)
        #self.polyline.knots=k/(k.max()-k.min())
        super().__init__(BoundingBox(*aabb(np.array(self.geom))))

class  Set1Curve(Object3D):
    def __init__(self, geom,degree=1):

        self.boundary = np.array(geom)
        self.boundary[..., -1] = 0

        res=points_order(self.boundary[:-1,:-1])
        if res!=PointsOrder.CCW:
           self.boundary=np.flip(self.boundary,axis=0)



        self.boundary_curve = NURBSCurve(self.boundary, degree=1)




        super().__init__(BoundingBox(*aabb(np.array(self.boundary))))
    def cut(self, other: Set2Curve):
        #print(np.array(other.polyline.control_points),np.array(other.polyline.knots))
        current = other.polyline
        in_poly = is_point_in_polygon(current.evaluate(current.interval()[0])[:-1], self.boundary[:-1, :-1])
        if in_poly:
            rr=[is_point_in_polygon(pt[:-1], self.boundary[:-1, :-1]) for pt in current.control_points]
            if all(rr):
                return [current]



        pars1 = ccx(other.polyline, self.boundary_curve)


        res = list(zip(*pars1))
        if len(res) == 0:
            return []
        t, s = res



        t = np.unique(np.round(t,4))
        if len(t) > 0:
            #print(t)
            for s in current.interval():
                t=t[np.abs((t-s))>1e-9]

        if len(t) ==0:
            return [current]

        cuts = split_curve_multiple(current, t)

        inside = in_poly

        inside_crvs=[]
        for i in range(len(cuts)):
            cut=cuts[i]
            a,b=cut.interval()
            if scalar_norm(cut.evaluate(a)-cut.evaluate(b))<0.1:
                continue
            else:
                if inside:

                    inside_crvs.append(cut)

                    inside=False
                else:
                    inside=True
        #print(inside_crvs)


        return inside_crvs






def build_trees_v2(curves_set1, curves_set2):

    pn = build_bvh([Set1Curve(i) for i in curves_set1])
    curves_set2_bvh = build_bvh([Set2Curve(i) for i in curves_set2])

    return pn, curves_set2_bvh

def cut_intersections(objects_a,objects_b, print_progress=False):
    bvh_a,bvh_b=build_bvh(objects_a),build_bvh(objects_b)

    intersections=intersect_bvh_objects(bvh_a, bvh_b)
    l = len(intersections)
    all_cuts = dict()

    if print_progress:
        print(f'{l} potential intersections found.')

    for i, (a, b) in enumerate(intersections):
        if print_progress:
            print(f"Progress: {i}/{l}", flush=True, end='\r')
        if a.object not in all_cuts :
            all_cuts[a.object] = []
        cuts = a.object.cut(b.object)
        if len(cuts) > 0:
            all_cuts[a.object].extend(cuts)
    return [all_cuts.get(obj,[]) for obj in objects_a]
def cut_intersections_mp(objects_a,objects_b, print_progress=False, cpus=-1):
    bvh_a,bvh_b=build_bvh(objects_a),build_bvh(objects_b)
    if cpus == -1:
        cpus=os.cpu_count()
    intersections=intersect_bvh_objects(bvh_a, bvh_b)
    l = len(intersections)


    if print_progress:
        print(f'{l} potential intersections found.')
    with mp.Pool(cpus) as pool:
        return pool.map(process_pair, intersections)




def process_pair(ab):
    a,b=ab

    cuts = a.object.cut(b.object)
    return cuts

def cut(curves_set1, curves_set2, print_progress=True):
    panels_objects = [Set1Curve(i) for i in curves_set1]
    curves_set2_objects = [Set2Curve(i) for i in curves_set2]
    return cut_intersections(panels_objects,curves_set2_objects, print_progress=print_progress)



def nurbs_pipeline():

    import json
    import time
    import gc
    s = time.time()
    curves_set1 = Path(__file__).parent / "curves_set1.txt"
    curves_set2 =  Path(__file__).parent /"curves_set2.txt"
    with open(curves_set1, "r") as f:
        curves_set1_data = json.load(f)

    with open(curves_set2, "r") as f:
        curves_set2_data = json.load(f)

    print(f'read at {time.time() - s}')
    print(len(curves_set1_data), " curves in first set")
    print(len(curves_set2_data), " curves in second set")

    s = time.time()
    gc.disable() #for additional performance
    # m1 = cut_all(curves_set1_data, curves_set2_data, method=2)
    m1 = cut(curves_set1_data, curves_set2_data, print_progress=True)
    print('cut at ', time.time() - s)
    s = time.time()


    def find_nurb(dat):
        if isinstance(dat,NURBSCurve):
            return dat.astuple()

        else:

            return [find_nurb(item)for item in dat]



    with open(Path(__file__).parent/"result.txt", "w") as f:
        json.dump( find_nurb(m1), f)
    print('write at ', time.time() - s)


    gc.enable()
    gc.collect()
    print('clean up')






if __name__ == "__main__":
    # The first set contains closed curves, the second set contains open curves.
    # Task :
    #  1. break the curves from the second set at the intersection points with the curves from the first set
    #  2. to select only the segments that are inside the curves from the first set.
    nurbs_pipeline()