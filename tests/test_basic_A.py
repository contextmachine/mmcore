from unittest import TestCase

from mmcore.base.basic import A


class TestA(TestCase):
    def test_A(self):
        """
        Test A object
        @return:
        """
        try:
            b = A()
            a = A()

            c = A()
            d = A()

            b.cc = c
            a.first = b
            ##print(a.idict)
        except Exception as err:
            self.fail(f"Test {__file__}]'{self.test_A.__doc__}' fail with:\n\t{err}")



