import numpy as np
def power_of_two(x):
    # Get the closest power of two to a given number
    return 1 << (x - 1).bit_length()

def convert_lengths(dim, step=0.6):
    # Convert initial dims to power of 2 representations
    pow = power_of_two(round(dim / step))
    return pow * step


# functions to find optimal gradient for segments in pointcloud with increased density
def generate_array(start_val, constant_val, total_len):
    half_segment = np.geomspace(start_val, constant_val, 10)
    const_len = total_len - sum(half_segment)*2

    const_count = round(const_len/constant_val)
    const_val = const_len / const_count

    constant = np.ones(const_count) * const_val
    return np.concatenate([half_segment, constant, np.flip(half_segment)])




def divide_xyz(b_min, b_max, c):
    return [

        [b_min, c],
        [[c[0], b_min[1], b_min[2]], [b_max[0], c[1], c[2]]],
        [[c[0], c[1], b_min[2]], [b_max[0], b_max[1], c[2]]],
        [[b_min[0], c[1], b_min[2]], [c[0], b_max[1], c[2]]],
        [[b_min[0], b_min[1], c[2]], [c[0], c[1], b_max[2]]],
        [[c[0], b_min[1], c[2]], [b_max[0], c[1], b_max[2]]],
        [c, b_max],
        [[b_min[0], c[1], c[2]], [c[0], b_max[1], b_max[2]]]
    ]


def divide(dimtype, b_min, b_max, c):
    op_map = {
        "x": [[c[0], b_max[1], b_max[2]], [c[0], b_min[1], b_min[2]]],
        "y": [[b_max[0], c[1], b_max[2]], [b_min[0], c[1], b_min[2]]],
        "z": [[b_max[0], b_max[1], c[2]], [b_min[0], b_min[1], c[2]]]
    }
    return [[b_min, op_map[dimtype][0]],
            [op_map[dimtype][1], b_max]]

def divide_two(dimtype, b_min, b_max, c):
    op_map = {
        "xy": [[c[0], c[1], b_max[2]], [c[0], b_min[1], b_min[2]], [b_max[0], c[1], b_max[2]],
               [c[0], c[1], b_min[2]], [b_min[0], c[1], b_min[2]], [c[0], b_max[1], b_max[2]]],
        "xz": [[c[0], b_max[1], c[2]],[c[0], b_min[1], b_min[2]], [b_max[0], b_max[1], c[2]],
               [c[0], b_min[1], c[2]], [b_min[0], b_min[1], c[2]], [c[0], b_max[1], b_max[2]]],
        "yz": [[b_max[0], c[1], c[2]], [b_min[0], b_min[1], c[2]], [b_max[0], c[1], b_max[2]],
               [b_min[0], c[1], c[2]], [b_min[0], c[1], b_min[2]], [b_max[0], b_max[1], c[2]]]
    }
    return [[b_min, op_map[dimtype][0]],
            [op_map[dimtype][1], op_map[dimtype][2]],
            [op_map[dimtype][3], b_max],
            [op_map[dimtype][4], op_map[dimtype][5]]]

def find_division_option(b_min, b_max, c, dims, target):
    dim_checks = [dims[0] > target[0], dims[1] > target[1], dims[2] > target[2]]
    dim_count = sum(map(int, dim_checks))
    dim_type = "".join([dimtype for cond, dimtype in zip(dim_checks, 'xyz') if cond])
    if dim_count == 1:
        # divide 'x', 'y' or 'z'
        return divide(dim_type, b_min, b_max, c)
    elif dim_count == 2:
        # divide 'xy', 'xz' or 'yz'
        return divide_two(dim_type, b_min, b_max, c)
    elif dim_count == 3:
        # divide 'xyz'
        return divide_xyz(b_min, b_max, c)



def divide_2d(dimtype, b_min, b_max, c):
    op_map = {
        "x": [[b_min, [c[0], b_max[1]]], [[c[0], b_min[1]], b_max]],
        "y": [[b_min, [b_max[0], c[1]]], [[b_min[0], c[1]], b_max]],
        "xy": [[b_min, c], [[c[0], b_min[1]], [b_max[0], c[1]]], [c, b_max], [[b_min[0], c[1]], [c[0], b_max[1]]]]}
    return op_map[dimtype]


def find_division_option_2d(b_min, b_max, c, dims, target):
    dim_checks = [dims[0] > target[0], dims[1] > target[1]]
    dim_type = "".join([dimtype for cond, dimtype in zip(dim_checks, 'xy') if cond])

    return divide_2d(dim_type, b_min, b_max, c)
