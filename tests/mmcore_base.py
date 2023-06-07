import unittest
from mmcore.base.basic import Object3D, Group, DictSchema
import strawberry
from dataclasses import is_dataclass, asdict
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
