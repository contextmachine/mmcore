
def genus2(x):
    """
    sources:
        1. https://en.wikipedia.org/wiki/Implicit_surface
        2. https://en.wikipedia.org/wiki/File:Impl-flaeche-geschl2.svg
    :param x: point in space
    :type x: ArrayLike
    :return: signed distance as float
    """
    return 2 * x[1] * (x[1] ** 2 - 3 * x[0] ** 2) * (1 - x[2] ** 2) + (x[0] ** 2 + x[1] ** 2) ** 2 - (
                9 * x[2] ** 2 - 1) * (1 - x[2] ** 2)
