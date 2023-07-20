import unittest
from dataclasses import is_dataclass

import strawberry

from mmcore.base.basic import Group, Object3D
from mmcore.typegen.dict_schema import DictSchema

A=Object3D(name="A")
B = Group(name="B")
B.add(A)
dct = strawberry.asdict(B.get_child_three())
##print(dct)
ds=DictSchema(dct).generate_schema()
class MyTestCase(unittest.TestCase):
    def test_base(self):


        self.assertEqual(B.children, [A])  # add assertion here

    def test_dict(self):

        self.assertTrue(isinstance(dct, dict))
    def test_gen(self):


        self.assertTrue(is_dataclass(ds))

if __name__ == '__main__':
    unittest.main()
