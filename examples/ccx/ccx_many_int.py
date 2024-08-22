import numpy as np

from mmcore.geom.curves import NURBSpline
from mmcore.numeric.intersection.ccx import ccx


aa, bb = NURBSpline(
    np.array(
        [
            [22.158641527416805, -41.265945906519704, 0.0],
            [38.290860468167494, -12.153299366618626, 0.0],
            [-4.3376334258665850, -26.514161725191443, 0.0],
            [15.051938537620600, 19.088976634204428, 0.0],
            [-32.020429607822194, -4.4209634623771734, 0.0],
            [-14.840038397145449, 35.287157805129340, 0.0],
            [-44.548538735168648, 25.263858719823133, 0.0],
        ]
    )
), NURBSpline(
    np.array(
        [
            [-6.1666326988875966, 37.891976021922630, 0.0],
            [14.215544129727249, 34.295146382508435, 0.0],
            [40.163941122493227, 14.352347571166774, 0.0],
            [43.540592939134157, -9.4157170022721530, 0.0],
            [16.580183934742877, -30.210021970020129, 0.0],
            [-10.513217234303696, -21.362760866641814, 0.0],
            [-26.377549521918183, -1.2133457261141416, 0.0],
            [-9.3086771658378353, 19.974390832869219, 0.0],
            [6.3667708626935706, 27.313795735872205, 0.0],
            [22.990902897521764, 11.683487552065344, 0.0],
            [26.711915155435108, 0.6064494223866177, 0.0],
            [19.374509602616740, -11.611372227389872, 0.0],
            [8.2582629104155956, -16.234752290968999, 0.0],
            [-3.0903039985573031, -11.940646639020102, 0.0],
            [-10.739472285742522, -2.2469933680379199, 0.0],
            [2.2509778312197994, 7.9168038191384795, 0.0],
            [14.498391690318186, -0.17203316116128065, 0.0],
        ]
    )
)


res = ccx(aa, bb, 0.001)
print(f"Intersections count: {len(res)}")
print(res)  # [(0.393707275390625, 2.604217529296875), (1.043701171875, 10.961181640625), (1.78436279296875, 13.72467041015625), (2.875030517578125, 5.431243896484375)]

pt1 = [[22.158641527416805, -41.265945906519704, 0.0], [38.290860468167494, -12.153299366618626, 0.0],
       [-4.3376334258665850, -26.514161725191443, 0.0], [15.051938537620600, 19.088976634204428, 0.0],
       [-32.020429607822194, -4.4209634623771734, 0.0], [-14.840038397145449, 35.287157805129340, 0.0],
       [-44.548538735168648, 25.263858719823133, 0.0]]
pt2 = [[-6.1666326988875966, 37.891976021922630, 0.0], [14.215544129727249, 34.295146382508435, 0.0],
       [40.163941122493227, 14.352347571166774, 0.0], [43.540592939134157, -9.4157170022721530, 0.0],
       [16.580183934742877, -30.210021970020129, 0.0], [-10.513217234303696, -21.362760866641814, 0.0],
       [-26.377549521918183, -1.2133457261141416, 0.0], [-9.3086771658378353, 19.974390832869219, 0.0],
       [6.3667708626935706, 27.313795735872205, 0.0], [22.990902897521764, 11.683487552065344, 0.0],
       [26.711915155435108, 0.6064494223866177, 0.0], [19.374509602616740, -11.611372227389872, 0.0],
       [8.2582629104155956, -16.234752290968999, 0.0], [-3.0903039985573031, -11.940646639020102, 0.0],
       [-10.739472285742522, -2.2469933680379199, 0.0], [2.2509778312197994, 7.9168038191384795, 0.0],
       [14.498391690318186, -0.17203316116128065, 0.0]]

from mmcore.geom.curves import NURBSpline

cc, dd = NURBSpline(np.array(pt1)), NURBSpline(np.array(pt2))

from mmcore.numeric.intersection.ccx import ccx

res = ccx(cc, dd)

from mmcore.renderer import Renderer2D

r = Renderer2D()
r.add_marker(np.array([cc.evaluate(s) for s, t in res]), color="red", size=5)
res = r([cc, dd], display_ctrlpts=False)