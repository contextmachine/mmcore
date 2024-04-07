import unittest
from pathlib import Path

import numpy as np

from mmcore.geom.bspline import NURBSpline
import time
from mmcore.numeric.closest_point import closest_point_on_curve


class MyTestCase(unittest.TestCase):
    def setUp(self):
        tests_data_dir = Path(__file__).parent.resolve()
        with open(tests_data_dir / "data/closest_points_dataset.json") as f:
            import json

            self.data = json.load(f)

    def test_closest_point(self):
        vals = []
        for j, item in enumerate(self.data):
            spl = NURBSpline(
                np.array(item["input"]["curve"]["control_points"]),
                degree=item["input"]["curve"]["degree"],
            )
            pts = np.array(item["input"]["points"])
            result = item["result"]
            for i, pt in enumerate(pts):
                s = time.time()
                res1, d = closest_point_on_curve(spl, pt, 1e-3)
                res2 = result[i]
                if not ((res1 - res2) <= 1e-3):
                    print(j, i, res1, res2)
                vals.append(time.time() - s)
                self.assertTrue((res1 - res2) <= 1e-3)
        print(
            f"{len(vals)} calls,  total: {np.sum(vals)}, ~ {np.average(vals)} per call"
        )

    def test_workers_closest_point(self):
        vals = []

        for j, item in enumerate(self.data):
            spl = NURBSpline(
                np.array(item["input"]["curve"]["control_points"]),
                degree=item["input"]["curve"]["degree"],
            )
            pts = np.array(item["input"]["points"])
            result = item["result"]
            s = time.time()
            res=closest_point_on_curve(spl, pts, tol=1e-3, workers=10)
            vals.append(time.time() - s)
            self.assertTrue(np.all((np.array(res)[...,0] - result) <= 1e-3))
        print(
            f"{len(vals)} calls,  total: {np.sum(vals)}, ~{np.average(vals)} per call"
        )

    # add assertion here


if __name__ == "__main__":
    unittest.main()
