#  Copyright (c) 2022. Computational Geometry, Digital Engineering and Optimizing your construction processe"
import copy
import json
import os
import unittest

import dotenv

dotenv.load_dotenv(dotenv.find_dotenv('.env', True))
CHECK_VAL = 11


class TestRemoteObject(metaclass=RemoteType, bucket=os.getenv("BUCKET"), storage=os.getenv("STORAGE"),
                       prefix="cxm/tests/", suffix=""):

    def __gethook__(self, hook):
        ##print(f"{self} get hook")
        return json.loads(hook["Body"].read())

    def __sethook__(self, hook):
        ##print(f"{self} set hook")
        return json.dumps(hook)


class RemoteTestCase(unittest.TestCase):
    test_object = TestRemoteObject()

    def test0(self):
        self.assertTrue(hasattr(self.test_object, "test_attribute"))

    def test1(self):
        global CHECK_VAL

        self.assertEqual(self.test_object.test_attribute["check_val"], CHECK_VAL)

    def test2(self):
        global CHECK_VAL

        self.test_object.test_attribute = {"name": "test_attribute", "check_val": 22}

        cached = copy.deepcopy(self.test_object.test_attribute["check_val"])
        self.test_object.test_attribute = {"name": "test_attribute", "check_val": 11}

        self.assertTrue((self.test_object.test_attribute["check_val"] == CHECK_VAL) and (cached == 22))
