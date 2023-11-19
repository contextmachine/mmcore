import numpy as np
import time

from mmcore.geom.mesh import MeshTuple, extract_mesh_attrs_union_keys_with_counter, \
    extract_mesh_attrs_union_keys_with_set

MESH_ATTRIBUTE_NAME_CASES = 'position', 'normal', 'uv', 'color'


def attr_match_test_data(count, cases, min_count=1):
    i = 0
    while i < count:
        i += 1
        yield MeshTuple(dict.fromkeys(cases[:np.random.randint(min_count, len(cases) + 1)]), None, None)


def performance(count, cases, min_count=1):
    *data, = attr_match_test_data(count, cases, min_count=min_count)
    s = time.time()
    res1 = extract_mesh_attrs_union_keys_with_counter(data)
    time1 = time.time() - s
    s = time.time()
    res2 = extract_mesh_attrs_union_keys_with_set(data)
    time2 = time.time() - s
    perf = sorted([('case1', time1), ('case2', time2)], key=lambda x: x[1])
    (best, best_time), (second, second_time) = perf

    message = f'{best} is {second_time / best_time} fastest! ({count} meshes)'
    return res1, res2, divmod(time1, 60), divmod(time2, 60), message


def test(count):
    min_count = 2

    result = tuple(MESH_ATTRIBUTE_NAME_CASES[:min_count]), set(MESH_ATTRIBUTE_NAME_CASES[:min_count])

    a, b, t1, t2, message = performance(count, MESH_ATTRIBUTE_NAME_CASES, min_count=min_count)

    print(f'performance: \n\tcase1: {t1}\n\tcase2: {t2}\n\t{message}')
    print(f'result: \n\tcase1: {a}\n\tcase2: {b}')

    if (a, b) == result:
        print("pass!")
    else:
        print(f'Fail: {result} != {(a, b)}')
    assert a, b == result


if __name__ == '__main__':
    test(10)
    test(100)
    test(500)
    test(1000)
    test(10_000)
    test(100_000)
    test(1_000_000)
