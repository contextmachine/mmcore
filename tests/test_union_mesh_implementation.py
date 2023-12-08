import pickle
from unittest import TestCase

import time

from mmcore.geom.mesh import mesh_comparison, union_mesh


def test_union_mesh_performance(fun):
    """
    Тест производительности для разных реализаций функций объединения mesh.
    Тестируемая функция должна принимать список MeshTuple и словарь extras, опционально.
    Возвращать должна MeshTuple с аттрибутом _objectid.
      :param fun: union_mesh function
    :type fun:
    :return:
    :rtype:
    """

    s = time.time()
    with open('../tests/data/union_mesh_test_data_rects.pkl', 'rb') as f:
        meshes = pickle.load(f)
    print('load data', divmod(time.time() - s, 60))
    s = time.time()
    b = fun(meshes)
    print(fun.__name__, divmod(time.time() - s, 60))
    return b


import multiprocessing as mp


def union_mesh_candidate(candidate):
    def test(prints=False):
        return test_union_mesh_implementation(candidate, prints=prints)

    return test


def test_union_mesh_implementation(candidate, prints=False):
    """
     При тестировании реализации необходимо сравнить ее со стабильной по производительности. И соответствию
     результирующих данных с помощью mesh_comparison.

    :param fun:
    :type fun:
    :param prints:
    :type prints:
    :return:
    :rtype:
    """

    with mp.Pool(2) as p:
        res = p.map(test_union_mesh_performance, [union_mesh, candidate])

    r, diffs = mesh_comparison(*res)
    if prints:
        if not r:

            print("Fail")
            print(diffs)

        else:
            print('Done')
    return r


def my_union_mesh_implementation1(meshes, extras=None):
    return union_mesh(meshes, extras)


class TestUmeshMeshCandidate(TestCase):
    def test_case1(self):
        self.assertTrue(test_union_mesh_implementation(my_union_mesh_implementation1, prints=True))

    def test_case2(self):
        """
        Comparison with self :)
        :return:
        :rtype:
        """
        self.assertTrue(test_union_mesh_implementation(union_mesh, prints=True))
