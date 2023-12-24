import functools
import unittest
from unittest.mock import MagicMock, patch
from collections import namedtuple

CNT = []


# Assuming the ExecutableNode class provided by you
class ExecutableNode:
    def __init__(self):
        self.inputs = "Inputs"

    def __hash__(self):
        return hash(self.inputs)

    @functools.lru_cache(maxsize=None)
    def solve(self):
        global CNT

        CNT.append(self.inputs)

    @property
    def output(self):
        self.solve()
        return CNT[-1]


# Test case for ExecutableNode
class TestExecutableNode(unittest.TestCase):
    def setUp(self):
        self.node = ExecutableNode()

    def test_output(self):
        self.node.solve()
        self.node.solve()

        print(self.node.__hash__())
        self.assertTrue(len(CNT) == 1)
        self.assertTrue(self.node.output == "Inputs")
        print(self.node.__hash__())
        self.node.inputs = "NewInputs"

        self.node.solve()
        print(self.node.__hash__())
        print(CNT)
        self.assertTrue(len(CNT) == 2)
        self.assertTrue(self.node.output == 'NewInputs')


# Run the tests
if __name__ == '__main__':
    unittest.main()
