"""
This example illustrates how intersections can be handled on a geometric level,
i.e. without resorting to advanced topological algorithms and data structures.
Only points, curves, surfaces and trims are used.


At the end of executing this script you should see something like:
```
6 intersection computed at: 0.03678107261657715 sec.

(<mmcore.geom.surfaces.Ruled object at 0x109a071d0> X
    <mmcore.geom.surfaces.BiLinear object at 0x118184e00>):
    1. <mmcore.geom.curves.bspline.NURBSpline object at 0x10feffa50>, <mmcore.geom.surfaces.CurveOnSurface object at 0x104820d40>, <mmcore.geom.surfaces.CurveOnSurface object at 0x109f23e00>

(<mmcore.geom.surfaces.Ruled object at 0x118184e30> X
    <mmcore.geom.surfaces.BiLinear object at 0x118184e00>):
    1. <mmcore.geom.curves.bspline.NURBSpline object at 0x11f8f4250>, <mmcore.geom.surfaces.CurveOnSurface object at 0x11f8d7e60>, <mmcore.geom.surfaces.CurveOnSurface object at 0x11f8fc050>

...

(<mmcore.geom.surfaces.Ruled object at 0x118184e30> X
    <mmcore.geom.surfaces.BiLinear object at 0x118187620>):
    1. <mmcore.geom.curves.bspline.NURBSpline object at 0x11f8f4e50>, <mmcore.geom.surfaces.CurveOnSurface object at 0x11f8fff20>, <mmcore.geom.surfaces.CurveOnSurface object at 0x11f990050>

```


"""
import numpy as np

from mmcore.geom.bvh import intersect_bvh_objects
from mmcore.geom.curves.bspline import interpolate_nurbs_curve
from mmcore.geom.surfaces import Ruled, BiLinear
from mmcore.numeric.intersection.surface_surface import surface_intersection

# Define two test sets of surfaces to intersect with each other
pts1 = (
        np.array(
            [
                [8969.2391032358137, 36402.046050942292, 6292.8224532207423],
                [8303.3111021092336, 36256.949420894314, 6292.7876616253379],
                [7671.3283331387356, 36085.626212471994, 6297.5857751334588],
                [7101.6111784193927, 35844.637630011224, 6297.1096704469128],
                [6611.8779113112250, 35470.259327025233, 6265.4031398450425],
                [6241.5211691120639, 34961.150857322231, 6181.8319066840231],
                [6018.0884860818915, 34366.490073671725, 6031.1260584794763],
                [5950.4714543944428, 33728.547159776361, 5805.7174807852389],
                [6003.7916451886849, 33102.538853535443, 5501.4120469770096],
                [6163.1800719991443, 32536.070062824292, 5066.8201586184477],
                [6423.6862474074587, 32170.439637394782, 4448.2917266677359],
                [6693.2862046856608, 32076.209350652032, 3755.4779113889781],
                [6921.7675112245342, 32141.334848101738, 3079.6183776142680],
                [7113.4722683146974, 32245.522771637792, 2490.2727432253123],
            ]
        )

)
pts2 = (
        np.array(
            [
                [8891.4710897927580, 36691.791696208704, 6292.8241203500766],
                [8233.6040283262992, 36541.175235157461, 6287.2928093149203],
                [7588.5113214327139, 36344.336052763640, 6275.8754203989047],
                [6983.7287266775238, 36049.094855798547, 6248.5965202524203],
                [6475.1969193423138, 35614.085056450065, 6184.0976188852328],
                [6119.1945810210600, 35051.894754217254, 6066.5286491586703],
                [5927.5459295028268, 34429.017735234054, 5886.5092704965609],
                [5901.7851646732743, 33799.668387282749, 5639.2177055551547],
                [5994.0701299196080, 33205.413854076323, 5327.3242424203891],
                [6175.5316698708630, 32694.389813346912, 4924.3625114633578],
                [6436.7418832486292, 32401.250513832805, 4378.6395500375766],
                [6700.2626604665420, 32352.773208395840, 3758.3850334360141],
                [6927.2318490327161, 32427.760148593115, 3124.0275810434359],
                [7137.6116218915704, 32544.549833613004, 2489.9443907486457],
            ]
        )

)
pts3 = (
        np.array(
            [
                [8823.0264092137368, 36026.109703064103, 7757.6259785028269],
                [8073.8840691360965, 35809.841544881507, 7746.6472099496859],
                [7329.7809725639327, 35577.209630192745, 7733.6741229199706],
                [6586.4362825951248, 35343.131940568594, 7712.7269218637484],
                [5865.3169418177713, 35051.355130849493, 7661.9189689828891],
                [5146.9139813784423, 34705.682126232678, 7552.7118156625766],
                [4508.0076803997945, 34269.913604068803, 7335.8195732309359],
                [4052.4976475952135, 33763.792133403418, 6917.5799797250766],
                [3909.8186831829080, 33332.563528079154, 6263.9964531137484],
                [4010.4369016621204, 33038.359857972871, 5523.2280204965609],
                [4204.5192327377154, 32822.143043686774, 4772.4536369516391],
                [4452.3213028329774, 32676.826695022057, 4020.5653618051547],
                [4741.3811703451793, 32633.563246605328, 3271.2367485239047],
                [5068.5739190365857, 32711.634213989026, 2540.0655347380707],
            ]
        )

)
curves = [
    interpolate_nurbs_curve(pts1, degree=3),
    interpolate_nurbs_curve(pts2, degree=3),
    interpolate_nurbs_curve(pts3, degree=3),
]

surfaces = [Ruled(curves[i], curves[i + 1]) for i in range(len(curves) - 1)]

pts21 = (
        np.array(
            [
                [8244.4457332359016, 37797.652290137819, 6163.4993129667237],
                [8501.3083858613973, 37869.063748097011, 6549.8104081081437],
                [8204.9399109777296, 37836.444647952376, 6916.2446621014351],
                [8456.5918353917077, 37915.338122788868, 7307.7468303782480],
                [8137.4042592510377, 37887.045931933731, 7660.4649412348481],
                [8380.3384645676269, 37970.965459917032, 8062.3970266759243],
            ]
        )

)

pts22 = (
        np.array(
            [
                [8642.2210429773131, 35281.063571088758, 6154.9743033489431],
                [8951.8311067289651, 35322.260974302349, 6643.7920771095132],
                [8586.7314310010242, 35327.310453723738, 6924.0872515320898],
                [8852.5642766267993, 35371.898030035103, 7324.2765755943656],
                [8512.2008921595407, 35388.180007850933, 7680.1304486542240],
                [8777.1529917496664, 35439.142134817441, 8089.3449105725776],
            ]
        )

)

surfaces2 = [
    BiLinear(pts21[i], pts21[i + 1], pts22[i + 1], pts22[i])
    for i in range(len(pts21) - 1)
]

# Build BVH for each surface

for surf in surfaces:
    surf.build_tree(10, 10)

for surf2 in surfaces2:
    surf2.build_tree(10, 10)
import time

# Find all intersection curves

s = time.time()
intersections = dict()
for i, surf1 in enumerate(surfaces):
    for j, surf2 in enumerate(surfaces2):
        if len(intersect_bvh_objects(surf1._tree, surf2._tree)) > 0:
            res = surface_intersection(surf1, surf2, tol=0.01)
            if res is None:
                pass
            else:
                intersections[(i, j)] = res
print(f'{len(intersections)} intersection computed at: {time.time() - s} sec.')

ptss = []
degs = []
for keys, crvs in intersections.items():
    print(f'\n({surfaces[keys[0]]} X \n\t{surfaces2[keys[1]]}):')

    for i, (spatial, uv1, uv2) in enumerate(crvs):
        print(f'\t{i + 1}. {spatial}, {uv1}, {uv2}')
        ptss.append((spatial.control_points).tolist())
        degs.append(spatial.degree)

# Tesselation

from mmcore.topo.mesh.tess import tessellate_surface, as_polygons

TT = dict()
for (i, j), v in intersections.items():
    if i not in TT:
        TT[i] = []
    for spatial, uv1, uv2 in v:
        TT[i].append(uv1)
meshes = []
for i, trms in TT.items():
    meshes.append(tessellate_surface(surfaces[i], trims=trms, v_count=8, calculate_density=True))

# This function will simply return all polygons as lists of points
# (this is handy for debugging me here and now, the original data structure can give you much more)
mshs = [(as_polygons(msh)).tolist() for msh in meshes]

