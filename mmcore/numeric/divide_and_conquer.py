import math

import numpy as np
from scipy.optimize import bisect
from mmcore.numeric.newton.cnewton import  newtons_method

from mmcore.numeric.fdm import classify_critical_point_2d, CriticalPointType





def recursive_divide_and_conquer_min(fun, bounds, tol):
    """
    Find the minimum value of a function within a given range using a recursive divide and conquer approach.

    :param fun: A callable function that takes a single parameter and returns a numeric value.
    :param bounds: A tuple representing the lower and upper bounds of the range to search for the minimum value.
    :param tol: The tolerance level for stopping the search when the range becomes smaller than this value.
    :return: A tuple containing the x-coordinate and the minimum value found within the given range.

    Algorithm Steps:
    1. Decompose the range into two parts, m1 and m2.
    2. Check if the precision level is achieved (high - low < tol). If yes, return the midpoint (x_min) and the corresponding function value.
    3. If not, compare the function values at m1 and m2.
    4. Recurse on the range with the lower function value.
    5. Repeat steps 2-4 until the minimum value is found within the desired precision level.

    Complexity Analysis:
    - The time complexity of this algorithm is O(log(1/tol)), where tol is the desired precision level.
    - The space complexity is O(log(1/tol)) due to the recursive calls.

    Example Usage:
    >>> def square(x):
    ...     return x ** 2
    >>> recursive_divide_and_conquer_min(square, (0, 10), 0.01)
    (0.0, 0.0)
    """
    low, high = bounds

    # Check if the precision level is achieved.
    if high - low < tol:
        x_min = (low + high) / 2

        return x_min, fun(x_min)
    else:
        # Divide the range into four parts.
        m1 = low + (high - low) / 4
        m2 = high - (high - low) / 4

        # Recurse on the half where the function value is lower.
        if fun(m1) < fun(m2):
            return recursive_divide_and_conquer_min(fun, (low, m2), tol)
        else:
            return recursive_divide_and_conquer_min(fun, (m1, high), tol)


def recursive_divide_and_conquer_max(fun, bounds, tol):
    """
    Find the maximum value of a function within a given range using a recursive divide and conquer approach.

    :param fun: A callable function that takes a single parameter and returns a numeric value.
    :param bounds: A tuple representing the lower and upper bounds of the range to search for the minimum value.
    :param tol: The tolerance level for stopping the search when the range becomes smaller than this value.
    :return: A tuple containing the x-coordinate and the minimum value found within the given range.

    Algorithm Steps:
    1. Decompose the range into two parts, m1 and m2.
    2. Check if the precision level is achieved (high - low < tol). If yes, return the midpoint (x_min) and the corresponding function value.
    3. If not, compare the function values at m1 and m2.
    4. Recurse on the range with the lower function value.
    5. Repeat steps 2-4 until the minimum value is found within the desired precision level.

    Complexity Analysis:
    - The time complexity of this algorithm is O(log(1/tol)), where tol is the desired precision level.
    - The space complexity is O(log(1/tol)) due to the recursive calls.

    Example Usage:
    >>> def square(x):
    ...     return x ** 2
    >>> recursive_divide_and_conquer_min(square, (0, 10), 0.01)
    (0.0, 0.0)
    """
    low, high = bounds

    # Check if the precision level is achieved.
    if high - low < tol:
        x_max = (low + high) / 2

        return x_max, fun(x_max)
    else:
        # Divide the range into four parts.
        m1 = low + (high - low) / 4
        m2 = high - (high - low) / 4

        # Recurse on the half where the function value is lower.
        if fun(m1) > fun(m2):
            return recursive_divide_and_conquer_max(fun, (low, m2), tol)
        else:
            return recursive_divide_and_conquer_max(fun, (m1, high), tol)


def iterative_divide_and_conquer_min(fun, bounds, tol):
    """
    Find the minimum value of a function within a given range using an iterative divide and conquer approach.

    :param fun: A callable function that takes a single parameter and returns a numeric value.
    :param bounds: A tuple representing the lower and upper bounds of the range to search for the minimum value.
    :param tol: The tolerance level for stopping the search when the range becomes smaller than this value.
    :return: A tuple containing the x-coordinate and the minimum value found within the given range.

    Complexity Analysis:
    - The time complexity of this algorithm is O(log(1/tol)), where tol is the desired precision level.
    - The space complexity is O(1) as we're using iteration now.

    Example Usage:
    >>> def square(x):
    ...     return x ** 2
    >>> iterative_divide_and_conquer_min(square, (0, 10), 0.01)
    (0.0, 0.0)
    """

    low, high = bounds

    while abs(high - low) >= tol:
        m1 = low + (high - low) / 4
        m2 = high - (high - low) / 4

        if fun(m1) < fun(m2):
            high = m2
        else:
            low = m1

    x_min = (low + high) / 2

    return x_min, fun(x_min)




import itertools




def recursive_divide_and_conquer_roots(fun, bounds, tol=0.01):
    low, high = bounds
    roots = []
    # Check if the precision level is achieved.
    if abs((high - low)) < tol:
        x_root = (low + high) / 2
        roots.append([x_root, fun(x_root)])
        return roots
    else:
        # Divide the range into four parts.
        m1 = low + (high - low) / 2
        m2 = high - (high - low) / 2

        # Recurse on the half where the function value is lower.
        if np.all(abs(fun(m1)) < abs(fun(m2))):
            roots.extend(recursive_divide_and_conquer_roots(fun, (low, m2), tol))
        else:
            roots.extend(recursive_divide_and_conquer_roots(fun, (m1, high), tol))

        return roots

def sign(val):
    return math.copysign(1, val)
def divide_and_conquer_min_2d(f, x_range, y_range, tol=1e-6):
    """
    :param f: The function to minimize. Should take two arguments, f(x, y), and return a scalar value.
    :param x_range: The range of x values to search for the minimum. Should be a tuple (x_min, x_max).
    :param y_range: The range of y values to search for the minimum. Should be a tuple (y_min, y_max).
    :param tol: The tolerance for the search. The search will stop when the range of x and y values is smaller than this tolerance.
    :return: The coordinates (x, y) of the minimum point found.

    This method implements a divide and conquer approach to find the minimum point of a 2D function within a given range. It starts with the entire range and iteratively divides it into smaller subranges until the range becomes smaller than the specified tolerance. At each iteration, the method evaluates the function at corners and midpoints of the edges, and selects the subrange that contains the minimum value. The method continues to divide this selected subrange until the range becomes smaller than the tolerance.

    The method returns the coordinates of the minimum point found, which is the midpoint of the final subrange.
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    while (x_max - x_min) > tol or (y_max - y_min) > tol:
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        # Evaluate the function at the corners and midpoints of the edges
        f00 = f(x_min, y_min)
        f10 = f(x_max, y_min)
        f01 = f(x_min, y_max)
        f11 = f(x_max, y_max)
        f_mid_x_min = f(x_mid, y_min)
        f_mid_x_max = f(x_mid, y_max)
        f_mid_y_min = f(x_min, y_mid)
        f_mid_y_max = f(x_max, y_mid)
        f_mid_xy = f(x_mid, y_mid)
        # Create a list of (value, coordinates) pairs
        candidates = [
            (f00, (x_min, y_min)),
            (f10, (x_max, y_min)),
            (f01, (x_min, y_max)),
            (f11, (x_max, y_max)),
            (f_mid_x_min, (x_mid, y_min)),
            (f_mid_x_max, (x_mid, y_max)),
            (f_mid_y_min, (x_min, y_mid)),
            (f_mid_y_max, (x_max, y_mid)),
            (f_mid_xy, (x_mid, y_mid))
        ]
        # Find the minimum value and its coordinates
        min_val, min_coords = min(candidates, key=lambda item: item[0])
        x_min, x_max = min_coords[0] - (x_max - x_min) / 4, min_coords[0] + (x_max - x_min) / 4
        y_min, y_max = min_coords[1] - (y_max - y_min) / 4, min_coords[1] + (y_max - y_min) / 4
        # Ensure the search space does not collapse below tolerance
        x_min = max(x_min, x_range[0])
        x_max = min(x_max, x_range[1])
        y_min = max(y_min, y_range[0])
        y_max = min(y_max, y_range[1])
    return (x_min + x_max) / 2, (y_min + y_max) / 2

def divide_and_conquer_min_2d_vectorized(f, x_range, y_range, tol=1e-6):
    """
    :param f: The function to minimize. Should take two arguments, f(x, y), and return a scalar value.
    :param x_range: The range of x values to search for the minimum. Should be a tuple (x_min, x_max).
    :param y_range: The range of y values to search for the minimum. Should be a tuple (y_min, y_max).
    :param tol: The tolerance for the search. The search will stop when the range of x and y values is smaller than this tolerance.
    :return: The coordinates (x, y) of the minimum point found.

    This method implements a divide and conquer approach to find the minimum point of a 2D function within a given range. It starts with the entire range and iteratively divides it into smaller subranges until the range becomes smaller than the specified tolerance. At each iteration, the method evaluates the function at corners and midpoints of the edges, and selects the subrange that contains the minimum value. The method continues to divide this selected subrange until the range becomes smaller than the tolerance.

    The method returns the coordinates of the minimum point found, which is the midpoint of the final subrange.
    """
    tol=tol/2
    x_min, x_max = x_range
    y_min, y_max = y_range
    while np.all((x_max - x_min) > tol) or np.all((y_max - y_min) > tol):
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        # Evaluate the function at the corners and midpoints of the edges
        f00 = f(x_min, y_min)
        f10 = f(x_max, y_min)
        f01 = f(x_min, y_max)
        f11 = f(x_max, y_max)
        f_mid_x_min = f(x_mid, y_min)
        f_mid_x_max = f(x_mid, y_max)
        f_mid_y_min = f(x_min, y_mid)
        f_mid_y_max = f(x_max, y_mid)
        f_mid_xy = f(x_mid, y_mid)
        # Create a list of (value, coordinates) pairs
        min_val=np.zeros(x_range.shape[1])
        min_coords=np.zeros((2,x_range.shape[1]))

        for i in range(x_range.shape[1]):
            candidates = [
                (f00[i], (x_min[i], y_min[i])),
                (f10[i], (x_max[i], y_min[i])),
                (f01[i], (x_min[i], y_max[i])),
                (f11[i], (x_max[i], y_max[i])),
                (f_mid_x_min[i], (x_mid[i], y_min[i])),
                (f_mid_x_max[i], (x_mid[i], y_max[i])),
                (f_mid_y_min[i], (x_min[i], y_mid[i])),
                (f_mid_y_max[i], (x_max[i], y_mid[i])),
                (f_mid_xy[i], (x_mid[i], y_mid[i]))
            ]

            min_val[i], min_coords[:,i]= min(candidates, key=lambda item: item[0])



        # Find the minimum value and its coordinates

        x_min, x_max = min_coords[0] - (x_max - x_min) / 4, min_coords[0] + (x_max - x_min) / 4
        y_min, y_max = min_coords[1] - (y_max - y_min) / 4, min_coords[1] + (y_max - y_min) / 4

        # Ensure the search space does not collapse below tolerance
        x_min = np.maximum(x_min, x_range[0])
        x_max = np.minimum(x_max, x_range[1])
        y_min = np.maximum(y_min, y_range[0])
        y_max = np.minimum(y_max, y_range[1])

    return (x_min + x_max) / 2, (y_min + y_max) / 2


def divide_and_conquer_min_3d(f, x_range, y_range, z_range, tol=1e-3):
    """
    :param f: The function to minimize. Should take three arguments, f(x, y, z), and return a scalar value.
    :param x_range: The range of x values to search for the minimum. Should be a tuple (x_min, x_max).
    :param y_range: The range of y values to search for the minimum. Should be a tuple (y_min, y_max).
    :param z_range: The range of z values to search for the minimum. Should be a tuple (z_min, z_max).
    :param tol: The tolerance for the search. The search will stop when the range of x, y, and z values is smaller than this tolerance.
    :return: The coordinates (x, y, z) of the minimum point found.

    This method implements a divide-and-conquer approach to find the minimum point of a 3D function within a given range.
    It starts with the entire range and iteratively divides it into smaller subranges until the range becomes smaller than the specified tolerance.
    At each iteration, the method evaluates the function at corners and midpoints of the edges, and selects the subrange that contains the minimum value.
    The method continues to divide this selected subrange until the range becomes smaller than the tolerance.
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range

    while (x_max - x_min) > tol or (y_max - y_min) > tol or (z_max - z_min) > tol:
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        z_mid = (z_min + z_max) / 2

        # Evaluate the function at the corners of the cube and midpoints of the edges
        f000 = f(x_min, y_min, z_min)
        f100 = f(x_max, y_min, z_min)
        f010 = f(x_min, y_max, z_min)
        f110 = f(x_max, y_max, z_min)
        f001 = f(x_min, y_min, z_max)
        f101 = f(x_max, y_min, z_max)
        f011 = f(x_min, y_max, z_max)
        f111 = f(x_max, y_max, z_max)

        f_mid_x_min_y_min = f(x_mid, y_min, z_min)
        f_mid_x_max_y_min = f(x_mid, y_max, z_min)
        f_mid_x_min_y_max = f(x_min, y_mid, z_min)
        f_mid_x_max_y_max = f(x_max, y_mid, z_min)
        f_mid_z_min = f(x_mid, y_mid, z_min)

        f_mid_x_min_z_max = f(x_min, y_min, z_mid)
        f_mid_x_max_z_max = f(x_max, y_min, z_mid)
        f_mid_y_min_z_max = f(x_min, y_max, z_mid)
        f_mid_y_max_z_max = f(x_max, y_max, z_mid)
        f_mid_z_max = f(x_mid, y_mid, z_max)

        f_mid_xyz = f(x_mid, y_mid, z_mid)

        # Create a list of (value, coordinates) pairs
        candidates = [
            (f000, (x_min, y_min, z_min)),
            (f100, (x_max, y_min, z_min)),
            (f010, (x_min, y_max, z_min)),
            (f110, (x_max, y_max, z_min)),
            (f001, (x_min, y_min, z_max)),
            (f101, (x_max, y_min, z_max)),
            (f011, (x_min, y_max, z_max)),
            (f111, (x_max, y_max, z_max)),
            (f_mid_x_min_y_min, (x_mid, y_min, z_min)),
            (f_mid_x_max_y_min, (x_mid, y_max, z_min)),
            (f_mid_x_min_y_max, (x_min, y_mid, z_min)),
            (f_mid_x_max_y_max, (x_max, y_mid, z_min)),
            (f_mid_z_min, (x_mid, y_mid, z_min)),
            (f_mid_x_min_z_max, (x_min, y_min, z_mid)),
            (f_mid_x_max_z_max, (x_max, y_min, z_mid)),
            (f_mid_y_min_z_max, (x_min, y_max, z_mid)),
            (f_mid_y_max_z_max, (x_max, y_max, z_mid)),
            (f_mid_z_max, (x_mid, y_mid, z_max)),
            (f_mid_xyz, (x_mid, y_mid, z_mid))
        ]

        # Find the minimum value and its coordinates
        min_val, min_coords = min(candidates, key=lambda item: item[0])
        x_min, x_max = min_coords[0] - (x_max - x_min) / 4, min_coords[0] + (x_max - x_min) / 4
        y_min, y_max = min_coords[1] - (y_max - y_min) / 4, min_coords[1] + (y_max - y_min) / 4
        z_min, z_max = min_coords[2] - (z_max - z_min) / 4, min_coords[2] + (z_max - z_min) / 4

        # Ensure the search space does not collapse below tolerance
        x_min = max(x_min, x_range[0])
        x_max = min(x_max, x_range[1])
        y_min = max(y_min, y_range[0])
        y_max = min(y_max, y_range[1])
        z_min = max(z_min, z_range[0])
        z_max = min(z_max, z_range[1])

    return (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2


def divide_and_conquer_min_nd(f, bounds, tol=1e-6):
    """
    Generalized divide-and-conquer method to find the minimum of a function in n dimensions, with the function returning
    2 curves for 2D and 4 surfaces for 3D.

    :param f: The function to minimize. Should take n arguments and return a scalar value.
    :param bounds: A list of tuples [(x_min, x_max), (y_min, y_max), ..., (z_min, z_max)] defining the range for each variable.
    :param tol: The tolerance for the search. The search will stop when the range of all variables is smaller than this tolerance.
    :return: A list of coordinates representing the minimum point found.
    """

    n_vars = len(bounds)  # Number of variables
    mins, maxs = zip(*bounds)  # Separate mins and maxs for each variable

    while any((maxs[i] - mins[i]) > tol for i in range(n_vars)):
        mids = [(mins[i] + maxs[i]) / 2 for i in range(n_vars)]

        # Evaluate function depending on the dimensionality of the problem
        candidates = []

        # For the 2D case, consider dividing along one dimension to generate two curves
        if n_vars == 2:
            for i in range(n_vars):
                for corner in [0, 1]:
                    point = [mins[j] if j != i else (mins[i] + maxs[i]) / 2 for j in range(n_vars)]
                    candidates.append((f(*point), point))

        # For the 3D case, consider dividing along two dimensions to generate four surfaces
        elif n_vars == 3:
            for corner in itertools.product([0, 1], repeat=2):  # Generate four surfaces
                point = [mins[i] if corner[i % 2] == 0 else maxs[i] for i in range(n_vars)]
                candidates.append((f(*point), point))

        # For higher dimensions, we consider more flexible subdivision strategies
        else:
            for corner in itertools.product([0, 1], repeat=n_vars):
                # Generate all corner points
                point = [mins[i] if corner[i] == 0 else maxs[i] for i in range(n_vars)]
                candidates.append((f(*point), point))

        # Find the minimum value and its coordinates
        min_val, min_coords = min(candidates, key=lambda item: item[0])

        # Update the bounds around the minimum point
        new_bounds = []
        for i in range(n_vars):
            range_size = (maxs[i] - mins[i]) / 4
            new_min = min_coords[i] - range_size
            new_max = min_coords[i] + range_size
            new_bounds.append((max(new_min, bounds[i][0]), min(new_max, bounds[i][1])))

        # Unpack the new bounds
        mins, maxs = zip(*new_bounds)

    # Return the midpoint of the final subrange
    return [(mins[i] + maxs[i]) / 2 for i in range(n_vars)]



def find_all_minima(f, x_range, y_range, grid_density=11, tol=1e-6):
    """

    :param f: The function to find minima for. It should be a function of two variables.
    :param x_range: The range of x-values to search for minima in, specified as a tuple (x_min, x_max).
    :param y_range: The range of y-values to search for minima in, specified as a tuple (y_min, y_max).
    :param grid_density: The number of grid points to generate along each axis. Higher grid density increases accuracy but also increases computational cost. Default is 10.
    :param tol: The tolerance for convergence criteria of the local minimization algorithm. Default is 1e-6.
    :return: A list of tuples (x, y, value) representing the minima found in the specified range.

    Example
    ---------
    >>> import numpy as np
    >>> def f(x, y):
    ...     # Example function, replace with your actual function
    ...     return (x - 1) ** 2 + (y - 2) ** 2 + np.sin(3 * x) * np.cos(3 * y)

    >>> x_range = (-2, 4)
    ... y_range = (-2, 4)
    ... tolerance = 1e-6

    >>> minima = find_all_minima(f, x_range, y_range, grid_density=20, tol=tolerance)
    >>> minima
    [(-0.20955570735465462, 2.068488218049512, 0.8814236284722828), (0.6239490890366713, 1.2349839216539054, -0.08077762630018981), (0.6319616334813937, 2.9100055589078364, 0.23559259074534356), (1.4654659299114376, 2.0765007624783474, -0.7266086640277386)]

    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    # Generate grid points
    x_vals = np.linspace(x_min, x_max, grid_density)
    y_vals = np.linspace(y_min, y_max, grid_density)
    minima = []
    tol2 = 10 ** (np.log10(tol) / 2)

    for x in x_vals:
        for y in y_vals:
            # Perform local minimization using scipy's minimize
            result, *other = newtons_method(lambda coords: f(coords[0], coords[1]), [x, y], tol=tol2, max_iter=8,
                                            no_warn=True, full_return=True)

            if result is not None:
                # if np.linalg.det(other[1])<0: seadle
                critical_point_type = classify_critical_point_2d(other[1])

                if critical_point_type is CriticalPointType.LOCAL_MINIMUM:

                    min_point = tuple(result)
                    min_value = f(*min_point)

                    # Check if this minimum is already in the list (within tolerance)
                    if all(
                            np.linalg.norm(np.array(min_point) - np.array(existing_min[:2]))
                            > tol2
                            for existing_min in minima
                    ):
                        minima.append((min_point[0], min_point[1], min_value))
    # Refine the minima using divide-and-conquer method
    refined_minima = []
    for min_point in minima:
        local_min = divide_and_conquer_min_2d(
            f,
            (min_point[0] - tol, min_point[0] + tol),
            (min_point[1] - tol, min_point[1] + tol),
            tol,
        )
        #print(min_point, local_min)
        refined_minima.append(
            (local_min[0], local_min[1], f(local_min[0], local_min[1]))
        )
    # Remove duplicates from the refined minima
    final_minima = []
    for min_point in refined_minima:
        if all(
                np.linalg.norm(np.array(min_point[:2]) - np.array(existing_min[:2])) > tol
                for existing_min in final_minima
        ):
            final_minima.append(min_point)
    return final_minima


def test_all_roots(fun, bounds, tol):
    t0, t1 = bounds
    if t1 - t0 <= tol:
        return []


    t_max, y_max = iterative_divide_and_conquer_min(lambda t: -fun(t), (t0, t1), tol)

    t_min, y_min = iterative_divide_and_conquer_min(fun, (t0, t1), tol)
    (t11, y11, typ11), (t21, y21, typ21) = sorted([(t_min, y_min, 0), (t_max, -y_max, 1)], key=lambda x: x[0])

    if sign(y11) != sign(y21):

        bs = bisect(fun, t11, t21)


        if t21 - t11 <= tol:
            return [bs]

        return [*test_all_roots(fun, (t0, t11-tol), tol), bs, *test_all_roots(fun, (t21+tol, t1), tol)]

    else:

        if t21 - t11 <= tol:
            return []
        return [*test_all_roots(fun, (t21 + tol, t1), tol)]




