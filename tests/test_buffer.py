#  Copyright (c) 2022. Computational Geometry, Digital Engineering and Optimizing your construction processe"
import os
import unittest
from unittest import TestCase

from ..geom.buffer import TrimmingCone


class TestTrimmingCone(TestCase):

    def run(self, result: unittest.result.TestResult | None = ...) -> unittest.result.TestResult | None:
        self.tc = TrimmingCone([1, 2, 3], [12, 15, 8], 5, 7)
        return super(TestTrimmingCone, self).run(result)

    def test_to_json(self):
        self.assertIsNotNone(self.tc.to_json())
        print(self.tc.to_json(indent=2))

    def test_to_dict(self):
        self.assertIsNotNone(self.tc.to_dict())

    def test_write(self):
        with open(f"{os.getenv('HOME')}/mmodel/mm/tests/{self.__class__.__name__}.json", "w") as f:
            f.write(self.tc.to_json())
